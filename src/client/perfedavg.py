from copy import deepcopy
from typing import Iterator

import torch
from torch.utils.data import DataLoader

from src.client.fedavg import FedAvgClient


class PerFedAvgClient(FedAvgClient):
    def __init__(self, **commons):
        super(PerFedAvgClient, self).__init__(**commons)
        self.iter_trainloader: Iterator[DataLoader] = None
        self.meta_optimizer = torch.optim.SGD(
            self.model.parameters(), lr=self.args.perfedavg.beta
        )
        if self.args.perfedavg.version == "hf":
            self.model_plus = deepcopy(self.model)
            self.model_minus = deepcopy(self.model)

    def package(self):
        client_package = super().package()
        client_package["weight"] = 1
        return client_package

    def fit(self):
        self.model.train()
        self.dataset.train()
        self.iter_trainloader = iter(self.trainloader)
        for _ in range(self.local_epoch):
            for _ in range(
                len(self.trainloader) // (2 + (self.args.perfedavg.version == "hf"))
            ):
                x0, y0 = self.get_data_batch()
                frz_params = deepcopy(self.model.state_dict())
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

                if self.args.perfedavg.version == "hf":
                    self.model_plus.load_state_dict(frz_params)
                    self.model_minus.load_state_dict(frz_params)

                    x2, y2 = self.get_data_batch()

                    for param_p, param_m, param_cur in zip(
                        self.model_plus.parameters(),
                        self.model_minus.parameters(),
                        self.model.parameters(),
                    ):
                        param_p.data += self.args.perfedavg.delta * param_cur.grad
                        param_m.data -= self.args.perfedavg.delta * param_cur.grad

                    logit_plus = self.model_plus(x2)
                    logit_minus = self.model_minus(x2)

                    loss_plus = self.criterion(logit_plus, y2)
                    loss_minus = self.criterion(logit_minus, y2)

                    loss_plus.backward()
                    loss_minus.backward()

                    for param_cur, param_plus, param_minus in zip(
                        self.model.parameters(),
                        self.model_plus.parameters(),
                        self.model_minus.parameters(),
                    ):
                        param_cur.grad = param_cur.grad - self.args.optimizer.lr / (
                            2 * self.args.perfedavg.delta
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
