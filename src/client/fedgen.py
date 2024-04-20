from copy import deepcopy
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from src.utils.tools import count_labels
from src.client.fedavg import FedAvgClient


class FedGenClient(FedAvgClient):
    def __init__(self, generator: torch.nn.Module, **commons):
        super().__init__(**commons)
        self.generator = deepcopy(generator).to(self.device)
        self.label_counts: list[list[int]] = [
            count_labels(self.dataset, indices["train"], min_value=1)
            for indices in self.data_indices
        ]
        self.regularization: bool
        self.available_labels: torch.Tensor
        self.current_global_epoch: int

    def package(self):
        client_package = super().package()
        client_package["label_counts"] = self.label_counts[self.client_id]
        return client_package

    def set_parameters(self, package: dict[str, Any]):
        super().set_parameters(package)
        self.current_global_epoch = package["current_global_epoch"]
        self.regularization = package["regularization"]
        self.generator.load_state_dict(package["generator_params"])
        self.available_labels = torch.unique(
            self.dataset.targets[self.trainset.indices]
        ).tolist()

    def fit(self):
        self.model.train()
        self.generator.train()
        self.dataset.train()
        for _ in range(self.local_epoch):
            for x, y in self.trainloader:
                if len(y) <= 1:
                    continue
                x, y = x.to(self.device), y.to(self.device)

                logits = self.model(x)
                loss = self.criterion(logits, y)

                if self.regularization:
                    alpha = self.exp_coef_scheduler(self.args.fedgen.generative_alpha)
                    beta = self.exp_coef_scheduler(self.args.fedgen.generative_beta)
                    generator_output, _ = self.generator(y)
                    logits_gen = self.model.classifier(generator_output).detach()

                    latent_loss = beta * F.kl_div(
                        F.log_softmax(logits, dim=1),
                        F.softmax(logits_gen, dim=1),
                        reduction="batchmean",
                    )

                    sampled_y = torch.tensor(
                        np.random.choice(
                            self.available_labels, self.args.fedgen.gen_batch_size
                        ),
                        dtype=torch.long,
                        device=self.device,
                    )
                    generator_output, _ = self.generator(sampled_y)
                    logits = self.model.classifier(generator_output)
                    teacher_loss = alpha * self.criterion(logits, sampled_y)

                    gen_ratio = (
                        self.args.fedgen.gen_batch_size / self.args.common.batch_size
                    )

                    loss += gen_ratio * teacher_loss + latent_loss

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            if self.lr_scheduler is not None:
                self.lr_scheduler.step()

    def exp_coef_scheduler(self, init_coef):
        return max(
            1e-4,
            init_coef
            * (
                self.args.fedgen.coef_decay
                ** (self.current_global_epoch // self.args.fedgen.coef_decay_epoch)
            ),
        )
