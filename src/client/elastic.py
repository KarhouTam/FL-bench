from copy import deepcopy
import random
from typing import Any

import torch
from torch.utils.data import DataLoader, Subset

from src.client.fedavg import FedAvgClient
from src.utils.tools import trainable_params


class ElasticClient(FedAvgClient):
    def __init__(self, **commons):
        super().__init__(**commons)
        self.sampled_trainset = Subset(self.dataset, indices=[])
        self.sampled_trainloader = DataLoader(
            self.sampled_trainset, self.args.common.batch_size
        )

    def load_data_indices(self):
        train_data_indices = deepcopy(self.data_indices[self.client_id]["train"])
        random.shuffle(train_data_indices)
        sampled_size = int(len(train_data_indices) * self.args.elastic.sample_ratio)
        self.sampled_trainset.indices = train_data_indices[:sampled_size]
        self.trainset.indices = train_data_indices[sampled_size:]
        self.valset.indices = self.data_indices[self.client_id]["val"]
        self.testset.indices = self.data_indices[self.client_id]["test"]

    def set_parameters(self, package: dict[str, Any]):
        super().set_parameters(package)
        self.sensitivity = package["sensitivity"].to(self.device)

    def train(self, server_package: dict[str, Any]):
        self.set_parameters(server_package)

        self.model.eval()
        for x, y in self.sampled_trainloader:
            x, y = x.to(self.device), y.to(self.device)
            logits = self.model(x)
            loss = self.criterion(logits, y)
            grads_norm = [
                torch.norm(layer_grad[0]) ** 2
                for layer_grad in torch.autograd.grad(
                    loss, trainable_params(self.model)
                )
            ]
            for i in range(len(grads_norm)):
                self.sensitivity[i] = (
                    self.args.elastic.mu * self.sensitivity[i]
                    + (1 - self.args.elastic.mu) * grads_norm[i].abs()
                )

        self.train_with_eval()

        client_package = self.package()

        return client_package

    def package(self):
        client_package = super().package()
        client_package["sensitivity"] = self.sensitivity.cpu().clone()
        return client_package
