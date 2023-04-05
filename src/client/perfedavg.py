from collections import OrderedDict
from copy import deepcopy
from typing import Iterator

import torch
from torch.utils.data import DataLoader

from fedavg import FedAvgClient
from src.config.utils import clone_params, trainable_params


class PerFedAvgClient(FedAvgClient):
    def __init__(self, model, args, logger):
        super(PerFedAvgClient, self).__init__(model, args, logger)
        self.iter_trainloader: Iterator[DataLoader] = None
        self.meta_optimizer = torch.optim.SGD(
            trainable_params(self.model), lr=self.args.beta
        )
        if self.args.version == "hf":
            self.model_plus = deepcopy(self.model)
            self.model_minus = deepcopy(self.model)

    def train(
        self,
        client_id: int,
        new_parameters: OrderedDict[str, torch.nn.Parameter],
        return_diff=True,
        verbose=False,
    ):
        delta, _, stats = super().train(
            client_id, new_parameters, return_diff=return_diff, verbose=verbose
        )
        # Per-FedAvg's model aggregation doesn't need weight.
        return delta, 1.0, stats

    def load_dataset(self):
        super().load_dataset()
        self.iter_trainloader = iter(self.trainloader)

    def fit(self):
        self.model.train()
        for _ in range(self.args.local_epoch):
            for _ in range(len(self.trainloader) // (2 + (self.args.version == "hf"))):
                x0, y0 = self.get_data_batch()
                frz_params = clone_params(self.model)
                logit = self.model(x0)
                loss = self.criterion(logit, y0)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                x1, y1 = self.get_data_batch()
                logit = self.model(x1)
                loss = self.criterion(logit, y1)
                self.meta_optimizer.zero_grad()
                loss.backward()

                if self.args.version == "hf":
                    self.model_plus.load_state_dict(frz_params)
                    self.model_minus.load_state_dict(frz_params)

                    x2, y2 = self.get_data_batch()

                    for param_p, param_m, param_cur in zip(
                        trainable_params(self.model_plus),
                        trainable_params(self.model_minus),
                        trainable_params(self.model),
                    ):
                        param_p.data += self.args.delta * param_cur.grad
                        param_m.data -= self.args.delta * param_cur.grad

                    logit_plus = self.model_plus(x2)
                    logit_minus = self.model_minus(x2)

                    loss_plus = self.criterion(logit_plus, y2)
                    loss_minus = self.criterion(logit_minus, y2)

                    loss_plus.backward()
                    loss_minus.backward()

                    for param_cur, param_plus, param_minus in zip(
                        trainable_params(self.model),
                        trainable_params(self.model_plus),
                        trainable_params(self.model_minus),
                    ):
                        param_cur.grad = param_cur.grad - self.args.local_lr / (
                            2 * self.args.delta
                        ) * (param_plus.grad - param_minus.grad)
                        param_plus.grad.zero_()
                        param_minus.grad.zero_()

                self.model.load_state_dict(frz_params)
                self.meta_optimizer.step()

    def get_data_batch(self):
        try:
            x, y = next(self.iter_trainloader)
            # neglect the size 1 data batches
            if len(x) <= 1:
                x, y = next(self.iter_trainloader)
        except StopIteration:
            self.iter_trainloader = iter(self.trainloader)
            x, y = next(self.iter_trainloader)
        return x.to(self.device), y.to(self.device)
