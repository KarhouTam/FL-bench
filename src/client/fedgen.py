from collections import Counter
from copy import deepcopy
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from src.client.fedavg import FedAvgClient


class FedGenClient(FedAvgClient):
    def __init__(self, generator: torch.nn.Module, **commons):
        super().__init__(**commons)
        self.generator = deepcopy(generator).to(self.device)
        self.label_counts = []
        self.clients_available_labels = []
        for indices in self.data_indices:
            counter = Counter(np.array(self.dataset.targets)[indices["train"]])
            self.label_counts.append(
                [counter.get(i, 0) for i in range(len(self.dataset.classes))]
            )
            self.clients_available_labels.append(
                np.array([i for i in counter.keys() if counter[i] > 0])
            )
        self.numpy_targets = np.array(self.dataset.targets)
        self.regularization: bool
        self.alpha: float
        self.beta: float

    def package(self):
        client_package = super().package()
        client_package["label_counts"] = self.label_counts[self.client_id]
        return client_package

    def set_parameters(self, package: dict[str, Any]):
        super().set_parameters(package)
        self.alpha = package["alpha"]
        self.beta = package["beta"]
        self.regularization = package["regularization"]
        self.generator.load_state_dict(package["generator_params"])

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
                    generator_output, _ = self.generator(y)
                    logits_gen = self.model.classifier(generator_output).detach()

                    latent_loss = self.beta * F.kl_div(
                        F.log_softmax(logits, dim=1),
                        F.softmax(logits_gen, dim=1),
                        reduction="batchmean",
                    )

                    sampled_y = torch.tensor(
                        np.random.choice(
                            self.clients_available_labels[self.client_id],
                            self.args.fedgen.gen_batch_size,
                        ),
                        dtype=torch.long,
                        device=self.device,
                    )
                    generator_output, _ = self.generator(sampled_y)
                    logits = self.model.classifier(generator_output)
                    teacher_loss = self.alpha * self.criterion(logits, sampled_y)

                    gen_ratio = (
                        self.args.fedgen.gen_batch_size / self.args.common.batch_size
                    )

                    loss += gen_ratio * teacher_loss + latent_loss

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            if self.lr_scheduler is not None:
                self.lr_scheduler.step()
