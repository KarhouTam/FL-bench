from typing import OrderedDict

import torch

from fedavg import FedAvgClient
from src.config.utils import trainable_params


class FedDynClient(FedAvgClient):
    def __init__(self, model, args, logger):
        super().__init__(model, args, logger)

        self.nabla = self.vectorize(self.model).detach().clone().zero_()
        self.vectorized_global_params: torch.Tensor = None
        self.vectorized_curr_params: torch.Tensor = None

    def train(
        self,
        client_id: int,
        new_parameters: OrderedDict[str, torch.nn.Parameter],
        verbose=False,
    ):
        self.vectorized_global_params = self.vectorize(new_parameters).detach().clone()
        res = super().train(
            client_id, new_parameters, return_diff=False, verbose=verbose
        )
        with torch.no_grad():
            self.nabla = self.nabla - self.args.alpha * (
                self.vectorized_curr_params - self.vectorized_global_params
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
                loss = self.criterion(logit, y)
                self.vectorized_curr_params = self.vectorize(self.model)
                loss -= -torch.dot(self.nabla, self.vectorized_global_params)
                loss += (self.args.alpha / 2) * torch.norm(
                    self.vectorized_curr_params - self.vectorized_global_params
                )
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

    def vectorize(self, src):
        return torch.cat([param.flatten() for param in trainable_params(src)]).to(
            self.device
        )
