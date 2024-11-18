from typing import Any, Iterator

import torch
from torch.utils.data import DataLoader

from src.client.fedavg import FedAvgClient


class SCAFFOLDClient(FedAvgClient):
    def __init__(self, **commons):
        super().__init__(**commons)
        self.iter_trainloader: Iterator[DataLoader]
        self.c_local: list[torch.Tensor]
        self.c_global: list[torch.Tensor]
        self.y_delta: list[torch.Tensor]
        self.c_delta: list[torch.Tensor]

    def set_parameters(self, package: dict[str, Any]):
        super().set_parameters(package)
        self.iter_trainloader = iter(self.trainloader)
        self.c_global = package["c_global"]
        self.c_local = package["c_local"]

    def train(self, server_package: dict[str, Any]):
        self.set_parameters(server_package)
        self.train_with_eval()

        with torch.no_grad():
            self.y_delta = []
            c_plus = []
            self.c_delta = []

            model_params = self.model.state_dict()
            for key in server_package["regular_model_params"].keys():
                x, y_i = server_package["regular_model_params"][key], model_params[key]
                self.y_delta.append(y_i.cpu() - x)

            coef = 1 / (self.local_epoch * self.args.optimizer.lr)
            for c, c_i, y_del in zip(self.c_global, self.c_local, self.y_delta):
                c_plus.append(c_i - c - coef * y_del)

            for c_p, c_l in zip(c_plus, self.c_local):
                self.c_delta.append(c_p - c_l)

            self.c_local = c_plus

        return self.package()

    def package(self):
        client_package = super().package()
        client_package["c_delta"] = self.c_delta
        client_package["y_delta"] = self.y_delta
        return client_package

    def fit(self):
        self.model.train()
        self.dataset.train()
        for _ in range(self.args.common.local_epoch):
            x, y = self.get_data_batch()
            logits = self.model(x)
            loss = self.criterion(logits, y)
            self.optimizer.zero_grad()
            loss.backward()
            for param, c, c_i in zip(
                self.model.parameters(), self.c_global, self.c_local
            ):
                if param.requires_grad:
                    param.grad.data += (c - c_i).to(self.device)
            self.optimizer.step()

            if self.lr_scheduler is not None:
                self.lr_scheduler.step()

    def get_data_batch(self):
        try:
            x, y = next(self.iter_trainloader)
            if len(x) <= 1:
                x, y = next(self.iter_trainloader)
        except StopIteration:
            self.iter_trainloader = iter(self.trainloader)
            x, y = next(self.iter_trainloader)
        return x.to(self.device), y.to(self.device)
