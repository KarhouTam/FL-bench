from collections import Counter
from copy import deepcopy
from typing import Any

import numpy as np
import torch

from src.client.fedavg import FedAvgClient
from src.utils.constants import NUM_CLASSES


class FedPACClient(FedAvgClient):
    def __init__(self, **commons):
        super().__init__(**commons)
        self.global_prototypes = {}
        self.label_distribs = {}
        for client_id, indices in enumerate(self.data_indices):
            counter = Counter(np.array(self.dataset.targets)[indices["train"]])
            self.label_distribs[client_id] = torch.tensor(
                [counter.get(i, 0) for i in range(len(self.dataset.classes))],
                dtype=torch.float,
            )

        self.v = None
        self.h_ref = None
        self.optimizer = commons["optimizer_cls"](
            params=[
                {"params": self.model.base.parameters(), "lr": self.args.optimizer.lr},
                {
                    "params": self.model.classifier.parameters(),
                    "lr": self.args.fedpac.classifier_lr,
                },
            ]
        )
        self.init_optimizer_state = deepcopy(self.optimizer.state_dict())

    @torch.no_grad()
    def extract_stats(self):
        feature_length = self.model.classifier.in_features
        features = self.calculate_prototypes()

        distrib1 = self.label_distribs[self.client_id]
        distrib1 = distrib1 / distrib1.sum()
        distrib2 = distrib1.mul(distrib1)
        self.v = 0
        self.h_ref = torch.zeros(
            (NUM_CLASSES[self.args.dataset.name], feature_length), device=self.device
        )
        for i in range(NUM_CLASSES[self.args.dataset.name]):
            if isinstance(features[i], torch.Tensor):
                size = features[i].shape[0]
                mean = features[i].mean(dim=0)
                self.h_ref[i] = distrib1[i] * mean
                self.v += (
                    distrib1[i] * torch.trace((features[i].t() @ features[i]) / size)
                ).item()
                self.v -= (distrib2[i] * (mean**2)).sum().item()

        self.v /= len(self.trainset.indices)

    @torch.no_grad()
    def calculate_prototypes(self, mean=False):
        prototypes = [[] for _ in self.dataset.classes]
        for x, y in self.trainloader:
            if len(y) <= 1:
                continue
            features = self.model.get_last_features(x.to(self.device))
            for i, label in enumerate(y.tolist()):
                prototypes[label].append(features[i])

        for i, features in enumerate(prototypes):
            if len(features) > 0:
                prototypes[i] = torch.stack(features)
                if mean:
                    prototypes[i] = prototypes[i].mean(dim=0)

        return prototypes

    def set_parameters(self, package: dict[str, Any]):
        super().set_parameters(package)
        self.global_prototypes = package["global_prototypes"]
        self.extract_stats()

    def package(self):
        client_package = super().package()
        prototypes = []
        for proto in self.calculate_prototypes(mean=True):
            if isinstance(proto, torch.Tensor):
                prototypes.append(proto.detach().cpu().clone())
            elif isinstance(proto, list) and len(proto) == 0:  # void prototype
                prototypes.append(proto)
        client_package["prototypes"] = prototypes
        client_package["label_distrib"] = self.label_distribs[self.client_id]
        client_package["v"] = deepcopy(self.v)
        client_package["h_ref"] = self.h_ref.cpu().clone()
        return client_package

    def fit(self):
        self.model.train()
        self.dataset.train()
        local_prototypes = self.calculate_prototypes(mean=True)
        for E in range(self.local_epoch):
            if E < self.args.fedpac.train_classifier_round:
                for x, y in self.trainloader:
                    if len(y) <= 1:
                        continue
                    x, y = x.to(self.device), y.to(self.device)
                    logits = self.model(x)
                    loss = self.criterion(logits, y)
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.model.base.zero_grad()
                    self.optimizer.step()
            else:
                for x, y in self.trainloader:
                    if len(y) <= 1:
                        continue
                    x, y = x.to(self.device), y.to(self.device)
                    features = self.model.get_last_features(x, detach=False)
                    logits = self.model.classifier(features)
                    loss_ce = self.criterion(logits, y)

                    loss_mse = 0
                    target_prototypes = features.clone().detach()
                    if self.global_prototypes is not None:
                        for i, label in enumerate(y.cpu().tolist()):
                            target_prototypes[i] = self.global_prototypes.get(
                                label, local_prototypes[label]
                            )
                    loss_mse = torch.nn.functional.mse_loss(features, target_prototypes)
                    loss = loss_ce + self.args.fedpac.lamda * loss_mse
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.model.classifier.zero_grad()
                    self.optimizer.step()
