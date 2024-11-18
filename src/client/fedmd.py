from typing import Any

import torch

from src.client.fedavg import FedAvgClient


class FedMDClient(FedAvgClient):
    def __init__(self, **commons):
        super(FedMDClient, self).__init__(**commons)
        self.mse_criterion = torch.nn.MSELoss()
        self.consensus: list[torch.Tensor]
        self.public_data: list[torch.Tensor]

    def set_parameters(self, package: dict[str, Any]):
        super().set_parameters(package)
        self.consensus = package["consensus"]
        self.public_data = package["public_data"]

    def train(self, server_package: dict[str, Any]):
        self.set_parameters(server_package)
        self.digest()
        self.train_with_eval()
        return self.package()

    def digest(self):
        self.model.train()
        for _ in range(self.args.fedmd.digest_epoch):
            for i in range(self.args.fedmd.public_batch_num):
                logit = self.model(self.public_data[i].to(self.device))
                loss = self.mse_criterion(logit, self.consensus[i].to(self.device))
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
