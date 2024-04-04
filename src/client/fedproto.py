from copy import deepcopy
from typing import Dict

import torch
import torch.nn.functional as F

from fedavg import FedAvgClient
from src.utils.models import DecoupledModel
from src.utils.tools import Logger, NestedNamespace
from src.utils.constants import NUM_CLASSES


class FedProtoClient(FedAvgClient):
    def __init__(
        self,
        model: DecoupledModel,
        args: NestedNamespace,
        logger: Logger,
        device: torch.device,
    ):
        super().__init__(model, args, logger, device)
        shape = (
            NUM_CLASSES[self.args.common.dataset],
            self.model.classifier.in_features,
        )
        self.global_prototypes = torch.zeros(shape, device=self.device)
        self.accumulated_features = torch.zeros(shape, device=self.device)
        self.personal_params_name = list(self.model.state_dict().keys())
        self.init_personal_params_dict = deepcopy(self.model.state_dict())
        self.label_counts = torch.zeros(
            NUM_CLASSES[self.args.common.dataset], 1, device=self.device
        )

    def train(
        self,
        client_id: int,
        local_epoch: int,
        global_prototypes: Dict[int, torch.Tensor],
        verbose=False,
    ):
        self.client_id = client_id
        self.local_epoch = local_epoch
        self.global_prototypes = global_prototypes
        self.accumulated_features.zero_()
        self.label_counts.zero_()
        self.load_dataset()
        self.model.load_state_dict(
            self.personal_params_dict.get(
                self.client_id, self.init_personal_params_dict
            )
        )
        self.optimizer.load_state_dict(
            self.opt_state_dict.get(self.client_id, self.init_opt_state_dict)
        )

        eval_results = self.train_and_log(verbose=verbose)

        client_prototypes = {}
        for i in range(NUM_CLASSES[self.args.common.dataset]):
            if self.label_counts[i] > 0:
                client_prototypes[i] = (
                    self.accumulated_features[i] / self.label_counts[i]
                )

        return (client_prototypes, eval_results)

    def fit(self):
        self.model.train()
        self.dataset.train()
        for _ in range(self.local_epoch):
            for x, y in self.trainloader:
                if len(x) <= 1:
                    continue

                x, y = x.to(self.device), y.to(self.device)
                features = F.relu(self.model.get_final_features(x, detach=False))
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

    @torch.no_grad()
    def process_features(self, features: torch.Tensor, y: torch.Tensor):
        labels = torch.unique(y)
        target_prototypes = features.clone()
        for i in labels:
            idxs = torch.where(y == i)[0]
            self.accumulated_features[i] += features[idxs].sum(dim=0)
            self.label_counts[i] += len(idxs)
            if i in self.global_prototypes.keys():
                target_prototypes[idxs] = self.global_prototypes[i]

        return target_prototypes
