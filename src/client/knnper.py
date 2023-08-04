import random
from typing import Dict, List

import torch
import numpy as np
import faiss

from fedavg import FedAvgClient


class kNNPerClient(FedAvgClient):
    def __init__(self, model, args, logger, device):
        super().__init__(model, args, logger, device)
        self.datastore = DataStore(args, self.model.classifier.in_features)
        self.test_flag = False

    def test(self, client_id, new_paramters):
        self.test_flag = True
        res = super().test(client_id, new_paramters)
        self.test_flag = False
        return res

    @torch.no_grad()
    def evaluate(self) -> Dict[str, float]:
        if self.test_flag:
            self.dataset.enable_train_transform = False
            self.model.eval()
            criterion = torch.nn.CrossEntropyLoss(reduction="sum")
            model_logits = []
            train_features = []
            train_targets = []
            test_features = []
            test_targets = []

            for x, y in self.trainloader:
                x, y = x.to(self.device), y.to(self.device)

                train_features.append(self.model.get_final_features(x))
                train_targets.append(y)

            for x, y in self.testloader:
                x, y = x.to(self.device), y.to(self.device)

                feature = self.model.get_final_features(x)
                test_features.append(feature)
                model_logits.append(self.model.classifier(torch.relu(feature)))
                test_targets.append(y)

            model_logits = torch.cat(model_logits, dim=0)
            test_features = torch.cat(test_features, dim=0)
            test_targets = torch.cat(test_targets, dim=0)

            self.datastore.clear()
            self.datastore.build(train_features, train_targets)
            knn_logits = self.get_knn_logits(test_features)
            self.datastore.clear()

            logits = (
                self.args.weight * knn_logits + (1 - self.args.weight) * model_logits
            )
            pred = torch.argmax(logits, dim=-1)
            loss = criterion(logits, test_targets).item()
            correct = (pred == test_targets).sum().item()

            self.dataset.enable_train_transform = True
            # kNN-Per only do kNN trick in the test phase. So stats about evaluation on train data are not offered.
            return {
                "train_loss": 0,
                "test_loss": loss,
                "train_correct": 0,
                "test_correct": correct,
                "train_size": 1.0,
                "test_size": float(max(1, len(test_targets))),
            }
        else:
            return super().evaluate()

    def get_knn_logits(self, features: torch.Tensor):
        distances, indices = self.datastore.index.search(
            features.cpu().numpy(), self.args.k
        )
        similarities = np.exp(-distances / (features.shape[-1] * self.args.scale))
        neighbors_targets = self.datastore.targets[indices]

        masks = np.zeros(((len(self.dataset.classes),) + similarities.shape))

        for class_id in list(range(len(self.dataset.classes))):
            masks[class_id] = neighbors_targets == class_id

        knn_logits = (similarities * masks).sum(axis=2) / (
            similarities.sum(axis=1) + 1e-9
        )

        return torch.tensor(knn_logits.T, device=self.device)


class DataStore:
    def __init__(self, args, dimension):
        self.args = args
        self.dimension = dimension
        self.index = faiss.IndexFlatL2(dimension)
        self.targets = None

    def build(self, features: List[torch.Tensor], targets: List[torch.Tensor]):
        num_samples = len(features)
        features_ = torch.cat(features, dim=0).cpu().numpy()
        targets_ = torch.cat(targets, dim=0).cpu().numpy()
        if num_samples <= self.args.capacity:
            self.index.add(features_)
            self.targets = targets_
        else:
            indices = random.sample(list(range(num_samples)), self.args.capacity)
            self.index.add(features_[indices])
            self.targets = targets_[indices]

    def clear(self):
        self.index = faiss.IndexFlatL2(self.dimension)
        self.targets = None
