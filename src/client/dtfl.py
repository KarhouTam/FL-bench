import time
from typing import Any

import torch

from src.client.fedavg import FedAvgClient


class DTFLClient(FedAvgClient):
    """Client implementation for Dynamic Tiering Federated Learning (DTFL).

    Reference:
        Seyed Mahmoud Sajjadi Mohammadabadi, Sydneyuzorert Huffman, Lei Yang, and Feng Yan,
        "Speed Up Federated Learning in Heterogeneous Environments: A Dynamic Tiering Approach,"
        IEEE Internet of Things Journal (IoT-J), 2024.
        DOI: https://doi.org/10.1109/JIOT.2024.3468511
    """

    def __init__(self, **commons):
        super().__init__(**commons)
        self.tier: int = 0
        self.step_time: float = 0.0
        self.avg_loss: float = 0.0

    def set_parameters(self, package: dict[str, Any]):
        super().set_parameters(package)
        self.tier = package.get("tier", 0)

    def fit(self):
        self.model.train()
        self.dataset.train()

        start_time = time.perf_counter()
        total_loss = 0.0
        total_batches = 0

        for _ in range(self.local_epoch):
            for x, y in self.trainloader:
                # Discard batch size <= 1 to avoid BatchNorm2d errors
                if len(x) <= 1:
                    continue

                x, y = x.to(self.device), y.to(self.device)
                logit = self.model(x)
                loss = self.criterion(logit, y)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                total_loss += loss.item()
                total_batches += 1

            if self.lr_scheduler is not None:
                self.lr_scheduler.step()

        elapsed = max(time.perf_counter() - start_time, 1e-6)
        effective_epochs = max(self.local_epoch, 1)
        self.step_time = elapsed / effective_epochs
        self.avg_loss = total_loss / max(total_batches, 1)

    def package(self):
        client_package = super().package()
        client_package["step_time"] = self.step_time
        client_package["avg_loss"] = self.avg_loss
        client_package["tier"] = self.tier
        return client_package