from typing import Any

import torch
import torch.nn.functional as F

from src.client.fedavg import FedAvgClient
from src.utils.constants import NUM_CLASSES


class FedProtoClient(FedAvgClient):
    def __init__(self, **commons):
        super().__init__(**commons)
        shape = (
            NUM_CLASSES[self.args.dataset.name],
            self.model.classifier.in_features,
        )
        self.global_prototypes = torch.zeros(shape, device=self.device)
        self.accumulated_features = torch.zeros(shape, device=self.device)
        self.personal_params_name = list(self.model.state_dict().keys())
        self.label_counts = torch.zeros(
            NUM_CLASSES[self.args.dataset.name], 1, device=self.device
        )

    def set_parameters(self, package: dict[str, Any]):
        super().set_parameters(package)
        self.global_prototypes = {
            i: proto.to(self.device)
            for i, proto in package["global_prototypes"].items()
        }
        self.accumulated_features.zero_()
        self.label_counts.zero_()
        self.client_prototypes = {}

    def train(self, server_package: dict[str, Any]):
        self.set_parameters(server_package)
        self.train_with_eval()

        for i in range(NUM_CLASSES[self.args.dataset.name]):
            if self.label_counts[i] > 0:
                self.client_prototypes[i] = (
                    self.accumulated_features[i] / self.label_counts[i]
                ).cpu()

        return self.package()

    def package(self):
        client_package = super().package()
        client_package["prototypes"] = self.client_prototypes

        return client_package

    def fit(self):
        self.model.train()
        self.dataset.train()
        for _ in range(self.local_epoch):
            for x, y in self.trainloader:
                if len(x) <= 1:
                    continue

                x, y = x.to(self.device), y.to(self.device)
                features = F.relu(self.model.get_last_features(x, detach=False))
                logits = self.model.classifier(features)
                target_prototypes = self.process_features(features, y)

                prototype_loss = 0
                if len(self.global_prototypes) > 0:
                    prototype_loss = F.mse_loss(features, target_prototypes)
                ce_loss = self.criterion(logits, y)
                loss = ce_loss + self.args.fedproto.lamda * prototype_loss

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            if self.lr_scheduler is not None:
                self.lr_scheduler.step()

    @torch.no_grad()
    def process_features(self, features: torch.Tensor, y: torch.Tensor):
        labels = torch.unique(y).tolist()
        target_prototypes = features.clone()
        for i in labels:
            idxs = torch.where(y == i)[0]
            self.accumulated_features[i] += features[idxs].sum(dim=0)
            self.label_counts[i] += len(idxs)
            if i in self.global_prototypes.keys():
                target_prototypes[idxs] = self.global_prototypes[i]

        return target_prototypes
