from collections import OrderedDict
from copy import deepcopy
from typing import Any

import torch
from torch.optim import Optimizer

from src.client.fedavg import FedAvgClient


class pFedMeClient(FedAvgClient):
    def __init__(self, **commons):
        super(pFedMeClient, self).__init__(**commons)
        self.local_parameters: list[torch.Tensor] = None
        self.personalized_params_dict: dict[str, OrderedDict[str, torch.Tensor]] = {}
        self.optimzier = pFedMeOptimizer(
            self.model.parameters(),
            self.args.pfedme.pers_lr,
            self.args.pfedme.lamda,
            self.args.pfedme.mu,
        )
        if self.lr_scheduler_cls is not None:
            self.lr_scheduler = self.lr_scheduler_cls(self.optimizer)

    def set_parameters(self, package: dict[str, Any]):
        super().set_parameters(package)
        self.local_parameters = deepcopy(list(package["regular_model_params"].values()))
        self.personalized_params_dict = package["personalized_model_params"]

    def package(self):
        client_package = super().package()
        client_package["personalized_model_params"] = {
            key: param.cpu() for key, param in self.personalized_params_dict.items()
        }
        client_package["local_model_params"] = self.local_parameters
        return client_package

    def fit(self):
        self.model.train()
        self.dataset.train()
        for _ in range(self.args.common.local_epoch):
            for x, y in self.trainloader:
                if len(x) <= 1:
                    continue

                x, y = x.to(self.device), y.to(self.device)
                for _ in range(self.args.pfedme.k):
                    logit = self.model(x)
                    loss = self.criterion(logit, y)
                    self.optimzier.zero_grad()
                    loss.backward()
                    self.optimzier.step(self.local_parameters)

                for param_p, param_l in zip(
                    self.model.parameters(), self.local_parameters
                ):
                    param_l.data = (
                        param_l.data
                        - self.args.pfedme.lamda
                        * self.args.optimizer.lr
                        * (param_l.data - param_p.data.cpu())
                    )

    @torch.no_grad()
    def evaluate(self):
        frz_model_params = deepcopy(self.model.state_dict())
        self.model.load_state_dict(self.personalized_params_dict)
        res = super().evaluate()
        self.model.load_state_dict(frz_model_params)
        return res


class pFedMeOptimizer(Optimizer):
    def __init__(self, params, lr=0.01, lamda=0.1, mu=0.001):
        defaults = dict(lr=lr, lamda=lamda, mu=mu)
        super(pFedMeOptimizer, self).__init__(params, defaults)

    @torch.no_grad()
    def step(self, local_parameters: list[torch.Tensor]):
        for group in self.param_groups:
            for param_p, param_l in zip(group["params"], local_parameters):
                if param_p.requires_grad:
                    param_p.data = param_p.data - group["lr"] * (
                        param_p.grad.data
                        + group["lamda"]
                        * (param_p.data - param_l.data.to(param_p.device))
                        + group["mu"] * param_p.data
                    )
