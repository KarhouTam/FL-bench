from typing import OrderedDict

import torch

from fedavg import FedAvgClient
from src.utils.tools import trainable_params, vectorize


class FedDynClient(FedAvgClient):
    def __init__(self, model, args, logger, device):
        super().__init__(model, args, logger, device)

        self.nabla: torch.Tensor = None
        self.flatten_global_params: torch.Tensor = None
        self.alpha: float = None

    def train(
        self,
        client_id: int,
        local_epoch: int,
        new_parameters: OrderedDict[str, torch.Tensor],
        nabla: torch.Tensor,
        alpha: float,
        return_diff=False,
        verbose=False,
    ):
        self.flatten_global_params = vectorize(new_parameters, detach=True)
        self.nabla = nabla
        self.alpha = alpha
        res = super().train(
            client_id, local_epoch, new_parameters, return_diff, verbose
        )
        return res

    def fit(self):
        self.model.train()
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
                    flatten_curr_params * (-self.flatten_global_params + self.nabla)
                )
                loss = loss_ce + loss_algo
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad.clip_grad_norm_(
                    trainable_params(self.model), max_norm=self.args.max_grad_norm
                )
                self.optimizer.step()
