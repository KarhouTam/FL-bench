from typing import Any

import torch

from src.client.fedavg import FedAvgClient
from src.utils.tools import trainable_params, vectorize


class FedDynClient(FedAvgClient):
    def __init__(self, **commons):
        super().__init__(**commons)

        self.nabla: torch.Tensor
        self.flatten_global_params: torch.Tensor
        self.alpha: float

    def set_parameters(self, package: dict[str, Any]):
        super().set_parameters(package)
        self.flatten_global_params = vectorize(
            package["regular_model_params"], detach=True
        )
        self.nabla = package["nabla"]
        self.alpha = package["alpha"]

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
                flatten_curr_params = vectorize(
                    trainable_params(self.model), detach=False
                )
                loss_algo = self.alpha * torch.sum(
                    flatten_curr_params
                    * (-self.flatten_global_params + self.nabla).to(self.device)
                )
                loss = loss_ce + loss_algo
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad.clip_grad_norm_(
                    trainable_params(self.model),
                    max_norm=self.args.feddyn.max_grad_norm,
                )
                self.optimizer.step()

            if self.lr_scheduler is not None:
                self.lr_scheduler.step()
