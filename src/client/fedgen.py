from collections import OrderedDict
from copy import deepcopy
from typing import List

import numpy as np
import torch
import torch.nn.functional as F

from src.utils.tools import count_labels, trainable_params
from fedavg import FedAvgClient


class FedGenClient(FedAvgClient):
    def __init__(self, model, args, logger, device):
        super().__init__(model, args, logger, device)
        self.label_counts: List[List[int]] = [
            count_labels(self.dataset, indices["train"], min_value=1)
            for indices in self.data_indices
        ]
        self.lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(
            optimizer=self.optimizer, gamma=0.99
        )

    def train(
        self,
        client_id: int,
        local_epoch: int,
        current_global_epoch: int,
        new_parameters: OrderedDict[str, torch.Tensor],
        generator: torch.nn.Module,
        regularization: bool,
        verbose=False,
    ):
        self.client_id = client_id
        self.local_epoch = local_epoch
        self.current_global_epoch = current_global_epoch
        self.generator = generator
        self.regularization = regularization
        self.load_dataset()
        self.set_parameters(new_parameters)

        eval_stats = self.train_and_log(verbose)

        return (
            trainable_params(self.model, detach=True),
            len(self.trainset),
            self.label_counts[self.client_id],
            eval_stats,
        )

    def set_parameters(self, new_parameters: OrderedDict[str, torch.Tensor]):
        super().set_parameters(new_parameters)
        self.available_labels = torch.unique(
            self.dataset.targets[self.trainset.indices]
        ).tolist()

    def fit(self):
        self.model.train()
        self.generator.train()
        for _ in range(self.local_epoch):
            for x, y in self.trainloader:
                if len(y) <= 1:
                    continue
                x, y = x.to(self.device), y.to(self.device)

                logits = self.model(x)
                loss = self.criterion(logits, y)

                if self.regularization:
                    alpha = self.exp_coef_scheduler(self.args.generative_alpha)
                    beta = self.exp_coef_scheduler(self.args.generative_beta)
                    generator_output, _ = self.generator(y)
                    logits_gen = self.model.classifier(generator_output).detach()

                    latent_loss = beta * F.kl_div(
                        F.log_softmax(logits, dim=1),
                        F.softmax(logits_gen, dim=1),
                        reduction="batchmean",
                    )

                    sampled_y = torch.tensor(
                        np.random.choice(
                            self.available_labels, self.args.gen_batch_size
                        ),
                        dtype=torch.long,
                        device=self.device,
                    )
                    generator_output, _ = self.generator(sampled_y)
                    logits = self.model.classifier(generator_output)
                    teacher_loss = alpha * self.criterion(logits, sampled_y)

                    gen_ratio = self.args.gen_batch_size / self.args.batch_size

                    loss += gen_ratio * teacher_loss + latent_loss

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

    def exp_coef_scheduler(self, init_coef):
        return max(
            1e-4,
            init_coef
            * (
                self.args.coef_decay
                ** (self.current_global_epoch // self.args.coef_decay_epoch)
            ),
        )
