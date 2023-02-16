from collections import OrderedDict
from typing import Dict, List, OrderedDict

import torch

from .fedavg import FedAvgClient
from config.utils import trainable_params


class SCAFFOLDClient(FedAvgClient):
    def __init__(self, model, args, logger):
        super().__init__(model, args, logger)
        self.c_local: Dict[List[torch.Tensor]] = {}
        self.c_diff = []

    def train(
        self,
        client_id: int,
        new_parameters: OrderedDict[str, torch.Tensor],
        c_global,
        evaluate=True,
        verbose=False,
    ):
        self.client_id = client_id
        self.load_dataset()
        self.set_parameters(new_parameters)
        if self.client_id not in self.c_local.keys():
            self.c_diff = c_global
        else:
            self.c_diff = []
            for c_l, c_g in zip(self.c_local[self.client_id], c_global):
                self.c_diff.append(-c_l + c_g)
        stats = self.log_while_training(evaluate, verbose)

        # update local control variate
        with torch.no_grad():

            if self.client_id not in self.c_local.keys():
                self.c_local[self.client_id] = [
                    torch.zeros_like(param, device=self.device)
                    for param in trainable_params(self.model)
                ]

            y_delta = []
            c_plus = []
            c_delta = []

            # compute y_delta (difference of model before and after training)
            y_delta = OrderedDict()
            for (name, param_g), param_l in zip(
                new_parameters.items(), trainable_params(self.model)
            ):
                y_delta[name] = param_l - param_g

            # compute c_plus
            coef = 1 / (self.local_epoch * self.args.local_lr)
            for c_diff, y_del in zip(self.c_diff, y_delta.values()):
                c_plus.append(-c_diff - coef * y_del)

            # compute c_delta
            for c_p, c_l in zip(c_plus, self.c_local[self.client_id]):
                c_delta.append(c_p - c_l)

            self.c_local[self.client_id] = c_plus

        return y_delta, c_delta, stats

    def _train(self):
        self.model.train()
        self.iter_trainloader = iter(self.trainloader)
        for _ in range(self.args.local_epoch):
            for x, y in self.trainloader:
                if len(x) <= 1:
                    continue

                x, y = x.to(self.device), y.to(self.device)
                logits = self.model(x)
                loss = self.criterion(logits, y)
                self.optimizer.zero_grad()
                loss.backward()
                for param, c_d in zip(trainable_params(self.model), self.c_diff):
                    param.grad.add_(c_d.data)
                self.optimizer.step()
