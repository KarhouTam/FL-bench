from typing import Any

import torch

from src.client.fedavg import FedAvgClient
from src.utils.tools import vectorize


class FedDynClient(FedAvgClient):
    def __init__(self, **commons):
        super().__init__(**commons)

        self.nabla: torch.Tensor
        self.alpha: float

    def set_parameters(self, package: dict[str, Any]):
        super().set_parameters(package)
        self.nabla = package["nabla"].to(self.device)
        self.alpha = package["alpha"]

    def fit(self):
        self.model.train()
        self.dataset.train()
        flatten_global_params = vectorize(self.model.parameters(), detach=True)
        for _ in range(self.local_epoch):
            for x, y in self.trainloader:
                if len(x) <= 1:
                    continue

                x, y = x.to(self.device), y.to(self.device)
                logit = self.model(x)
                loss_ce = self.criterion(logit, y)
                flatten_curr_params = vectorize(self.model.parameters(), detach=False)
                loss_algo = self.alpha * torch.sum(
                    flatten_curr_params * (-flatten_global_params + self.nabla)
                )
                loss = loss_ce + loss_algo
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad.clip_grad_norm_(
                    self.model.parameters(), max_norm=self.args.feddyn.max_grad_norm
                )
                self.optimizer.step()

            if self.lr_scheduler is not None:
                self.lr_scheduler.step()
