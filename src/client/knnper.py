import random

import faiss
import numpy as np
import torch

from src.client.fedavg import FedAvgClient
from src.utils.metrics import Metrics


class kNNPerClient(FedAvgClient):
    def __init__(self, **commons):
        super().__init__(**commons)
        self.datastore = DataStore(self.args, self.model.classifier.in_features)

    @torch.no_grad()
    def evaluate(self, model=None):
        if self.testing:
            self.dataset.eval()
            target_model = self.model if model is None else model
            target_model.eval()
            criterion = torch.nn.CrossEntropyLoss(reduction="sum")
            train_features, train_targets = [], []
            val_metrics = Metrics()
            test_metrics = Metrics()
            for x, y in self.trainloader:
                x, y = x.to(self.device), y.to(self.device)

                train_features.append(target_model.get_last_features(x))
                train_targets.append(y)

            def _knnper_eval(dataloader):
                nonlocal target_model, criterion, train_features, train_targets
                model_logits = []
                features, targets = [], []
                for x, y in dataloader:
                    x, y = x.to(self.device), y.to(self.device)

                    feature = target_model.get_last_features(x)
                    features.append(feature)
                    model_logits.append(target_model.classifier(torch.relu(feature)))
                    targets.append(y)

                model_logits = torch.cat(model_logits, dim=0)
                features = torch.cat(features, dim=0)
                targets = torch.cat(targets, dim=0)

                self.datastore.clear()
                self.datastore.build(train_features, train_targets)
                knn_logits = self.get_knn_logits(features)
                self.datastore.clear()

                logits = (
                    self.args.knnper.weight * knn_logits
                    + (1 - self.args.knnper.weight) * model_logits
                )
                pred = torch.argmax(logits, dim=-1)
                loss = criterion(logits, targets).item()
                return Metrics(loss, pred, targets)

            if len(self.testset) > 0 and self.args.common.eval_test:
                test_metrics = _knnper_eval(self.testloader)
            if len(self.valset) > 0 and self.args.common.eval_val:
                val_metrics = _knnper_eval(self.valloader)

            self.dataset.enable_train_transform = True
            # kNN-Per only do kNN trick in model test phase. So stats on training data are not offered.
            return {"train": Metrics(), "val": val_metrics, "test": test_metrics}
        else:
            return super().evaluate(model)

    def get_knn_logits(self, features: torch.Tensor):
        distances, indices = self.datastore.index.search(
            features.cpu().numpy(), self.args.knnper.k
        )
        similarities = np.exp(
            -distances / (features.shape[-1] * self.args.knnper.scale)
        )
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

    def build(self, features: list[torch.Tensor], targets: list[torch.Tensor]):
        num_samples = len(features)
        features_ = torch.cat(features, dim=0).cpu().numpy()
        targets_ = torch.cat(targets, dim=0).cpu().numpy()
        if num_samples <= self.args.knnper.capacity:
            self.index.add(features_)
            self.targets = targets_
        else:
            indices = random.sample(list(range(num_samples)), self.args.knnper.capacity)
            self.index.add(features_[indices])
            self.targets = targets_[indices]

    def clear(self):
        self.index = faiss.IndexFlatL2(self.dimension)
        self.targets = None
