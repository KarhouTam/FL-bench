from copy import deepcopy
from typing import Any

import torch
import torch.nn.functional as F

from src.client.fedavg import FedAvgClient
from src.utils.constants import NUM_CLASSES
from src.utils.models import DecoupledModel


class FedASClient(FedAvgClient):
    def __init__(self, **commons):
        super().__init__(**commons)
        self.prev_model: DecoupledModel = deepcopy(self.model)

    def get_fim_trace_sum(self) -> float:
        self.model.eval()
        self.dataset.eval()

        fim_trace_sum = 0

        for x, y in self.trainloader:
            x, y = x.to(self.device), y.to(self.device)
            logits = self.model(x)
            loss = (
                -F.log_softmax(logits, dim=1).gather(dim=1, index=y.unsqueeze(1)).mean()
            )

            self.model.zero_grad()
            loss.backward()

            for param in self.model.parameters():
                if param.grad is not None:
                    fim_trace_sum += (param.grad.data**2).sum().item()

        return fim_trace_sum

    def package(self):
        client_package = super().package()
        # FedAS uses the sum of FIM traces as the weight
        client_package["weight"] = self.get_fim_trace_sum()
        client_package["prev_model_state"] = deepcopy(self.model.state_dict())
        return client_package

    def set_parameters(self, package: dict[str, Any]) -> None:
        super().set_parameters(package)
        if package["prev_model_state"] is not None:
            self.prev_model.load_state_dict(package["prev_model_state"])
        else:
            self.prev_model.load_state_dict(self.model.state_dict())
        if not self.testing:
            self.align_federated_parameters()
        else:
            # FedAS evaluates clients' personalized models
            self.model.load_state_dict(self.prev_model.state_dict())

    def align_federated_parameters(self):
        self.prev_model.eval()
        self.prev_model.to(self.device)
        self.model.train()
        self.dataset.train()

        prototypes = [[] for _ in range(NUM_CLASSES[self.args.dataset.name])]

        with torch.no_grad():
            for x, y in self.trainloader:
                x, y = x.to(self.device), y.to(self.device)
                features = self.prev_model.get_last_features(x)

                for y, feat in zip(y, features):
                    prototypes[y].append(feat)

        mean_prototypes = [
            torch.stack(prototype).mean(dim=0) if prototype else None
            for prototype in prototypes
        ]

        alignment_optimizer = torch.optim.SGD(
            self.model.base.parameters(), lr=self.args.fedas.alignment_lr
        )

        for _ in range(self.args.fedas.alignment_epoch):
            for x, y in self.trainloader:
                x, y = x.to(self.device), y.to(self.device)
                features = self.model.get_last_features(x, detach=False)
                loss = 0
                for label in y.unique().tolist():
                    if mean_prototypes[label] is not None:
                        loss += F.mse_loss(
                            features[y == label].mean(dim=0), mean_prototypes[label]
                        )

                alignment_optimizer.zero_grad()
                loss.backward()
                alignment_optimizer.step()

        self.prev_model.cpu()
