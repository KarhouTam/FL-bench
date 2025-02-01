from typing import List, Optional, Union

import numpy as np
import torch
from omegaconf import DictConfig
from torch import Tensor, nn
from torch.utils.data import DataLoader, Dataset, Subset

from src.client.fedavg import FedAvgClient
from src.utils.models import DecoupledModel


class FedEMClient(FedAvgClient):
    def __init__(self, **commons):
        super().__init__(**commons)
        self.data_sample_weights = None

    def package(self):
        client_package = super().package()
        client_package["data_sample_weights"] = self.data_sample_weights
        client_package["learner_weights"] = self.model.learners_weights.clone().cpu()
        return client_package

    def set_parameters(self, package):
        super().set_parameters(package)
        if package["data_sample_weights"] is not None:
            self.data_sample_weights = package["data_sample_weights"]
        else:
            self.data_sample_weights = torch.ones(
                (len(self.trainset), self.args.fedem.n_learners), device=self.device
            )
        if package["learner_weights"] is not None:
            self.model.learners_weights = package["learner_weights"].to(self.device)
        else:
            self.model.learners_weights = (
                torch.ones(self.args.fedem.n_learners, device=self.device)
                / self.args.fedem.n_learners
            )

    @torch.no_grad
    def update_data_sample_weights(self):
        self.dataset.eval()
        self.model.eval()
        all_losses = torch.zeros(
            len(self.trainset), self.args.fedem.n_learners, device=self.device
        )
        indices_dataset = IndiceDataset(self.trainset.dataset)
        new_trainset = Subset(indices_dataset, self.trainset.indices)
        # Don't need to shuffle here, so the sampler is not specified.
        new_trainloader = DataLoader(
            new_trainset, batch_size=self.trainloader.batch_size
        )
        for idxs, x, y in new_trainloader:
            x, y = x.to(self.device), y.to(self.device)
            logits = torch.stack(self.model(x, aggregated=False), dim=-1)
            # Get the indices of the samples that are in the new trainset.
            all_losses[np.where(np.isin(new_trainset.indices, idxs))[0]] = (
                torch.nn.functional.cross_entropy(
                    logits,
                    y.unsqueeze(1).broadcast_to((len(y), self.args.fedem.n_learners)),
                    reduction="none",
                )
            )

        self.data_sample_weights = torch.softmax(
            (torch.log(self.model.learners_weights) - all_losses), dim=-1
        )

    def update_learner_weights(self):
        self.model.learners_weights = self.data_sample_weights.mean(dim=0)

    def fit(self):
        self.update_data_sample_weights()
        self.update_learner_weights()
        self.model.train()
        self.dataset.train()
        indices_dataset = IndiceDataset(self.trainset.dataset)
        new_trainset = Subset(indices_dataset, self.trainset.indices)
        new_trainloader = DataLoader(
            new_trainset,
            batch_size=self.trainloader.batch_size,
            sampler=self.trainloader.sampler,
            drop_last=self.trainloader.drop_last,
        )
        for _ in range(self.local_epoch):
            for idxs, x, y in new_trainloader:
                if len(x) <= 1:
                    continue

                x, y = x.to(self.device), y.to(self.device)
                logits = torch.stack(self.model(x, aggregated=False), dim=-1)
                loss_no_reduction = nn.functional.cross_entropy(
                    logits,
                    y.unsqueeze(1).broadcast_to((len(y), self.args.fedem.n_learners)),
                    reduction="none",
                )
                loss = (
                    loss_no_reduction
                    * self.data_sample_weights[
                        np.where(np.isin(new_trainset.indices, idxs))[0]
                    ]
                ).sum() / len(y)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            if self.lr_scheduler is not None:
                self.lr_scheduler.step()


class IndiceDataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __getitem__(self, index):
        x, y = self.dataset[index]
        return index, x, y

    def __len__(self):
        return len(self.dataset)


class EnsembleModel(DecoupledModel):
    def __init__(self, learners: List[DecoupledModel], learners_weights: torch.Tensor):
        super().__init__()
        self.learners = nn.ModuleList(learners)
        self.base = nn.ModuleList([learner.base for learner in self.learners])
        self.classifier = nn.ModuleList(
            [learner.classifier for learner in self.learners]
        )
        self.dropout = []
        for learner in self.learners:
            for module in learner.modules():
                if isinstance(module, nn.Dropout):
                    self.dropout.append(module)
        self.learners_weights = learners_weights

    def forward(
        self, x: torch.Tensor, aggregated=True
    ) -> Union[torch.Tensor, List[torch.Tensor]]:
        outputs = [model(x) for model in self.learners]
        if aggregated:
            outputs = (self.learners_weights * torch.stack(outputs, dim=-1)).sum(dim=-1)
        return outputs

    def check_and_preprocess(self, args: DictConfig):
        for learner in self.learners:
            learner.check_and_preprocess(args)

    def get_last_features(self, x: torch.Tensor, detach=True) -> List[torch.Tensor]:
        return [learner.get_last_features(x, detach) for learner in self.learners]

    def get_all_features(self, x: torch.Tensor) -> Optional[List[Tensor]]:
        return [learner.get_all_features(x) for learner in self.learners]
