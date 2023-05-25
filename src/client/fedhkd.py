from collections import OrderedDict, Counter
from copy import deepcopy
from typing import Dict, List, Tuple

import torch
from torch.distributions import Normal

from fedavg import FedAvgClient
from src.config.utils import trainable_params


class FedHKDClient(FedAvgClient):
    def __init__(self, model, args, logger):
        super().__init__(model, args, logger)
        self.Q: List[torch.Tensor] = None
        self.H: List[torch.Tensor] = None
        self.data_count = []
        for indices in self.data_indices:
            counter = Counter(self.dataset.targets[indices["train"]].tolist())
            data_cnt = [0] * len(self.dataset.classes)
            for i, _ in enumerate(self.dataset.classes):
                data_cnt[i] = counter.get(i, 1)
            self.data_count.append(data_cnt)

    def train(
        self,
        client_id: int,
        new_parameters: OrderedDict[str, torch.nn.Parameter],
        global_hyper_knowledge: Tuple[torch.Tensor, torch.Tensor],
        return_diff=True,
        verbose=False,
    ) -> Tuple[List[torch.nn.Parameter], int, Dict]:
        self.client_id = client_id
        H, Q = global_hyper_knowledge
        if H is not None and Q is not None:
            self.Q = Q.to(self.device)
            self.H = H.to(self.device)
        self.set_parameters(new_parameters)
        self.load_dataset()
        eval_stats = self.train_and_log(verbose)
        hyper_knowledge = self.update_hyper_knowledge()

        if return_diff:
            delta = OrderedDict()
            for (name, p0), p1 in zip(
                new_parameters.items(), trainable_params(self.model)
            ):
                delta[name] = p0.to(self.device) - p1

            return delta, hyper_knowledge, self.data_count[self.client_id], eval_stats
        else:
            return (
                deepcopy(trainable_params(self.model)),
                hyper_knowledge,
                self.data_count[self.client_id],
                eval_stats,
            )

    def fit(self):
        self.model.train()
        for _ in range(self.local_epoch):
            for x, y in self.trainloader:
                if len(x) <= 1:
                    continue
                x, y = x.to(self.device), y.to(self.device)
                h = self.model.get_final_features(x, detach=False)
                logits = self.model.classifier(torch.relu(h))
                ce_loss = self.criterion(logits, y)
                hyper_knowledge_loss = 0
                if self.H is not None and self.Q is not None:
                    hyper_knowledge_loss = (
                        self.args.lamda
                        * (
                            self.soft_predict(self.model.classifier(torch.relu(self.H)))
                            - self.Q
                        )
                        .norm(dim=1)
                        .mean()
                        + self.args.gamma * (h - self.H[y]).norm()
                    )
                loss = ce_loss + hyper_knowledge_loss
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

    @torch.no_grad()
    def update_hyper_knowledge(self):
        self.model.eval()
        threshold = int(len(self.trainset) * self.args.threshold)
        class_ids = range(len(self.dataset.classes))
        features = []
        targets = []
        preds = []
        for x, y in self.trainloader:
            x, y = x.to(self.device), y.to(self.device)
            h = self.model.get_final_features(x)
            q = self.soft_predict(self.model.classifier(torch.relu(h)))
            features.append(h)
            targets.append(y)
            preds.append(q)

        features = torch.cat(features)
        targets = torch.cat(targets)
        preds = torch.cat(preds)

        feature_shape = features.shape[1:]
        pred_shape = preds.shape[1:]

        aligned_features = [features[torch.where(targets == i)] for i in class_ids]
        aligned_preds = [preds[torch.where(targets == i)] for i in class_ids]

        # prune
        for i in class_ids:
            if len(aligned_preds[i]) < threshold:
                aligned_features[i] = None
                aligned_preds[i] = None

        zeta = []
        for feature in aligned_features:
            if feature is not None:
                zeta.append(torch.max(feature, 0).values)
            else:
                zeta.append(None)

        avg_features = [
            torch.zeros(*feature_shape, device=self.device)
            if f is None
            else f.mean(dim=0)
            for f in aligned_features
        ]
        avg_preds = [
            torch.zeros(*pred_shape, device=self.device) if p is None else p.mean(dim=0)
            for p in aligned_preds
        ]

        for i, (z, feature) in enumerate(zip(zeta, avg_features)):
            if z is not None and feature is not None:
                sensitivity = 2 * z / self.data_count[self.client_id][i]
                feature.add_(
                    Normal(
                        torch.zeros_like(feature),
                        torch.abs(sensitivity * self.args.sigma),
                    )
                    .sample()
                    .to(self.device)
                )

        return (avg_features, avg_preds)

    def soft_predict(self, z):
        soft_z = z / self.args.temperature
        return torch.exp(soft_z) / torch.sum(soft_z)
