import random
from copy import deepcopy
from typing import Any

import torch
from torch.utils.data import DataLoader, Subset

from src.client.fedavg import FedAvgClient


class ElasticClient(FedAvgClient):
    def __init__(self, **commons):
        super().__init__(**commons)
        self.sampled_trainset = Subset(self.dataset, indices=[])
        self.sampled_trainloader = DataLoader(
            self.sampled_trainset, self.args.common.batch_size
        )
        self.init_sensitivity = torch.zeros(
            len(list(self.model.parameters())), device=self.device
        )

    def load_data_indices(self):
        train_data_indices = deepcopy(self.data_indices[self.client_id]["train"])
        random.shuffle(train_data_indices)
        sampled_size = int(len(train_data_indices) * self.args.elastic.sample_ratio)
        self.sampled_trainset.indices = train_data_indices[:sampled_size]
        self.trainset.indices = train_data_indices[sampled_size:]
        self.valset.indices = self.data_indices[self.client_id]["val"]
        self.testset.indices = self.data_indices[self.client_id]["test"]

    def train(self, server_package: dict[str, Any]):
        self.set_parameters(server_package)

        sensitivity = self.init_sensitivity
        self.model.eval()
        for x, y in self.sampled_trainloader:
            x, y = x.to(self.device), y.to(self.device)
            logits = self.model(x)
            loss = self.criterion(logits, y)
            loss.backward()
            grad_norms = []
            for param in self.model.parameters():
                if param.requires_grad:
                    grad_norms.append(torch.norm(param.grad.data) ** 2)
                else:
                    grad_norms.append(None)
            for i in range(len(grad_norms)):
                if grad_norms[i]:
                    sensitivity[i] = (
                        self.args.elastic.mu * sensitivity[i]
                        + (1 - self.args.elastic.mu) * grad_norms[i].abs()
                    )
                else:
                    sensitivity[i] = 1.0

        self.train_with_eval()

        client_package = self.package()
        client_package["sensitivity"] = sensitivity.cpu()
        return client_package
