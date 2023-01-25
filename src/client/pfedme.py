from collections import OrderedDict
from copy import deepcopy
from typing import Dict, OrderedDict, Tuple

import torch
from torch.optim import Optimizer

from .fedavg import FedAvgClient
from config.utils import trainable_params


class pFedMeClient(FedAvgClient):
    def __init__(self, model, args, logger):
        super(pFedMeClient, self).__init__(model, args, logger)
        self.local_parameters = deepcopy(trainable_params(self.model))
        self.personalized_params_dict: Dict[str, OrderedDict[str, torch.Tensor]] = {}
        self.optimzier = pFedMeOptimizer(
            trainable_params(self.model),
            self.args.pers_lr,
            self.args.lamda,
            self.args.mu,
        )

    def train(
        self,
        client_id: int,
        new_parameters: OrderedDict[str, torch.Tensor],
        evaluate=True,
        verbose=False,
    ):
        self.client_id = client_id
        self.get_client_local_dataset()
        self.set_parameters(new_parameters)
        self.local_parameters = deepcopy(trainable_params(new_parameters))
        # self.iter_trainloader = iter(self.trainloader)
        stats = self.log_while_training(evaluate, verbose)
        return (deepcopy(self.local_parameters), 1.0, stats)

    def save_state(self):
        super().save_state()
        self.personalized_params_dict[self.client_id] = deepcopy(
            self.model.state_dict()
        )

    def _train(self):
        self.model.train()
        for _ in range(self.args.local_epoch):
            # x, y = self.get_data_batch()
            for x, y in self.trainloader:
                x, y = x.to(self.device), y.to(self.device)
                for _ in range(self.args.k):
                    logit = self.model(x)
                    loss = self.criterion(logit, y)
                    self.optimzier.zero_grad()
                    loss.backward()
                    self.optimzier.step(self.local_parameters)

                for param_p, param_l in zip(
                    trainable_params(self.model), self.local_parameters
                ):
                    param_l.data = param_l.data - self.args.lamda * self.local_lr * (
                        param_l.data - param_p.data
                    )

    @torch.no_grad()
    def evaluate(self) -> Tuple[float, float, int]:
        frz_model_params = deepcopy(self.model.state_dict())
        if self.client_id in self.personalized_params_dict.keys():
            self.model.load_state_dict(self.personalized_params_dict[self.client_id])
        self.model.eval()
        loss = 0
        correct = 0
        criterion = torch.nn.CrossEntropyLoss(reduction="sum")
        for x, y in self.testloader:
            x, y = x.to(self.device), y.to(self.device)
            logits = self.model(x)
            loss += criterion(logits, y)
            pred = torch.argmax(logits, dim=-1)
            correct += (pred == y).int().sum()
        self.model.load_state_dict(frz_model_params)
        return loss.item(), correct.item(), len(self.testset)


class pFedMeOptimizer(Optimizer):
    def __init__(self, params, lr=0.01, lamda=0.1, mu=0.001):
        defaults = dict(lr=lr, lamda=lamda, mu=mu)
        super(pFedMeOptimizer, self).__init__(params, defaults)

    @torch.no_grad()
    def step(self, local_parameters):
        group = None
        for group in self.param_groups:
            for param_p, param_l in zip(group["params"], local_parameters):
                param_p.data = param_p.data - group["lr"] * (
                    param_p.grad.data
                    + group["lamda"] * (param_p.data - param_l.data)
                    + group["mu"] * param_p.data
                )
