from argparse import ArgumentParser, Namespace
from copy import deepcopy
from collections import OrderedDict
from typing import List

import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader

from fedavg import FedAvgServer
from src.utils.tools import trainable_params, NestedNamespace
from src.utils.constants import NUM_CLASSES


def get_ccvr_args(args_list=None) -> Namespace:
    parser = ArgumentParser()
    parser.add_argument("--sample_per_class", type=int, default=200)
    return parser.parse_args(args_list)


class CCVRServer(FedAvgServer):
    def __init__(
        self,
        args: NestedNamespace,
        algo: str = "CCVR",
        unique_model=False,
        default_trainer=True,
    ):
        super().__init__(args, algo, unique_model, default_trainer)

    def test(self):
        frz_global_params_dict = deepcopy(self.global_params_dict)
        self.calibrate_classifier()
        super().test()
        self.global_params_dict = frz_global_params_dict

    def compute_classes_mean_cov(self):
        features_means, features_covs, features_count = [], [], []
        for client_id in self.train_clients:
            model_params = self.generate_client_params(client_id)
            means, covs, counts = self.get_means_covs_from_client(
                client_id, model_params
            )
            features_means.append(means)
            features_covs.append(covs)
            features_count.append(counts)

        num_classes = NUM_CLASSES[self.args.common.dataset]
        labels_count = [sum(cnts) for cnts in zip(*features_count)]
        classes_mean = [None for _ in range(num_classes)]
        classes_cov = [None for _ in range(num_classes)]
        for c, (means, counts) in enumerate(
            zip(zip(*features_means), zip(*features_count))
        ):
            if sum(counts) > 0:
                weights = torch.tensor(counts, device=self.device) / labels_count[c]
                means_ = torch.stack(means, dim=-1)
                classes_mean[c] = torch.sum(means_ * weights, dim=-1)
        for c in range(num_classes):
            if classes_mean[c] is not None and labels_count[c] > 1:
                for k in self.train_clients:
                    if classes_cov[c] is None:
                        classes_cov[c] = torch.zeros_like(features_covs[k][c])

                    classes_cov[c] += (
                        (features_count[k][c] - 1) / (labels_count[c] - 1)
                    ) * features_covs[k][c] + (
                        features_count[k][c] / (labels_count[c] - 1)
                    ) * (
                        features_means[k][c].unsqueeze(1)
                        @ features_means[k][c].unsqueeze(0)
                    )

                classes_cov[c] -= (labels_count[c] / (labels_count[c] - 1)) * (
                    classes_mean[c].unsqueeze(1) @ classes_mean[c].unsqueeze(0)
                )

        return classes_mean, classes_cov

    def generate_virtual_representation(
        self, classes_mean: List[torch.Tensor], classes_cov: List[torch.Tensor]
    ):
        data, targets = [], []
        for c, (mean, cov) in enumerate(zip(classes_mean, classes_cov)):
            if mean is not None and cov is not None:
                samples = np.random.multivariate_normal(
                    mean.cpu().numpy(),
                    cov.cpu().numpy(),
                    self.args.ccvr.sample_per_class,
                )
                data.append(
                    torch.tensor(samples, dtype=torch.float, device=self.device)
                )
                targets.append(
                    torch.ones(
                        self.args.ccvr.sample_per_class,
                        dtype=torch.long,
                        device=self.device,
                    )
                    * c
                )

        data = torch.cat(data)
        targets = torch.cat(targets)
        return data, targets

    def calibrate_classifier(self):
        classes_mean, classes_cov = self.compute_classes_mean_cov()
        data, targets = self.generate_virtual_representation(classes_mean, classes_cov)

        class RepresentationDataset(Dataset):
            def __init__(self, data, targets):
                self.data = data
                self.targets = targets

            def __getitem__(self, idx):
                return self.data[idx], self.targets[idx]

            def __len__(self):
                return len(self.targets)

        self.model.train()
        dataset = RepresentationDataset(data, targets)
        dataloader = DataLoader(
            dataset, batch_size=self.args.common.batch_size, shuffle=True
        )
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(
            trainable_params(self.model.classifier), lr=self.args.common.optimizer.lr
        )

        self.model.load_state_dict(self.global_params_dict, strict=False)
        for x, y in dataloader:
            logits = self.model.classifier(x)
            loss = criterion(logits, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        self.global_params_dict = OrderedDict(
            zip(self.trainable_params_name, trainable_params(self.model, detach=True))
        )

    def get_means_covs_from_client(
        self, client_id: int, global_params: OrderedDict[str, torch.Tensor]
    ):
        self.trainer.client_id = client_id
        self.trainer.load_dataset()
        self.trainer.set_parameters(global_params)
        features = []
        targets = []
        feature_length = None
        for x, y in self.trainer.trainloader:
            x, y = x.to(self.device), y.to(self.device)
            features.append(self.trainer.model.get_final_features(x))
            targets.append(y)

        targets = torch.cat(targets)
        features = torch.cat(features)
        feature_length = features.shape[-1]
        indices = [
            torch.where(targets == i)[0]
            for i in range(len(self.trainer.dataset.classes))
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
        return classes_means, classes_covs, [len(idxs) for idxs in indices]
