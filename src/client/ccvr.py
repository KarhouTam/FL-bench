from typing import Any

import torch

from src.client.fedavg import FedAvgClient


class CCVRClient(FedAvgClient):
    def __init__(self, **commons):
        super().__init__(**commons)

    def get_classwise_feature_means_and_covs(self, server_package: dict[str, Any]):
        self.set_parameters(server_package)
        self.model.eval()
        features = []
        targets = []
        feature_length = None
        for x, y in self.trainloader:
            x, y = x.to(self.device), y.to(self.device)
            features.append(self.model.get_last_features(x))
            targets.append(y)

        targets = torch.cat(targets)
        features = torch.cat(features)
        feature_length = features.shape[-1]
        indices = [
            torch.where(targets == i)[0] for i in range(len(self.dataset.classes))
        ]
        classes_features = [features[idxs] for idxs in indices]
        classes_means, classes_covs = [], []
        for fea in classes_features:
            if fea.shape[0] > 0:
                classes_means.append(fea.mean(dim=0))
                classes_covs.append(fea.t().cov(correction=0))
            else:
                classes_means.append(torch.zeros(feature_length, device=self.device))
                classes_covs.append(
                    torch.zeros(feature_length, feature_length, device=self.device)
                )

        return dict(
            means=classes_means,
            covs=classes_covs,
            counts=[len(idxs) for idxs in indices],
        )
