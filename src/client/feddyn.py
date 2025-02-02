from typing import Any

import torch

from src.client.fedavg import FedAvgClient
from src.utils.functional import vectorize


class FedDynClient(FedAvgClient):
    def __init__(self, **commons):
        super().__init__(**commons)

        self.delta: torch.Tensor

    def set_parameters(self, package: dict[str, Any]):
        super().set_parameters(package)
        self.delta = package["local_dual_correction"].to(self.device)

    def fit(self):
        self.model.train()
        self.dataset.train()
        for _ in range(self.local_epoch):
            for x, y in self.trainloader:
                if len(x) <= 1:
                    continue

                x, y = x.to(self.device), y.to(self.device)
                logit = self.model(x)
                loss_ce = self.criterion(logit, y)
                flatten_curr_params = vectorize(self.model.parameters(), detach=False)
                loss_correct = torch.sum(flatten_curr_params * self.delta)
                loss = loss_ce + self.args.feddyn.alpha * loss_correct
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), max_norm=self.args.feddyn.max_grad_norm
                )
                self.optimizer.step()

            if self.lr_scheduler is not None:
                self.lr_scheduler.step()
