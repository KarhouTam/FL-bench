from argparse import ArgumentParser, Namespace
from collections import OrderedDict
from copy import deepcopy

import numpy as np
import torch
from omegaconf import DictConfig
from torch.utils.data import DataLoader, Dataset

from src.client.ccvr import CCVRClient
from src.server.fedavg import FedAvgServer
from src.utils.constants import NUM_CLASSES


class CCVRServer(FedAvgServer):
    @staticmethod
    def get_hyperparams(args_list=None) -> Namespace:
        parser = ArgumentParser()
        parser.add_argument("--sample_per_class", type=int, default=200)
        return parser.parse_args(args_list)

    def __init__(
        self,
        args: DictConfig,
        algorithm_name: str = "CCVR",
        unique_model=False,
        use_fedavg_client_cls=False,
        return_diff=False,
    ):
        super().__init__(
            args, algorithm_name, unique_model, use_fedavg_client_cls, return_diff
        )
        self.init_trainer(CCVRClient)

    def test(self):
        frz_global_params_dict = deepcopy(self.public_model_params)
        self.calibrate_classifier()
        super().test()
        self.public_model_params = frz_global_params_dict

    def compute_classes_mean_cov(self):
        features_means, features_covs, features_count = [], [], []
        client_packages = self.trainer.exec(
            "get_classwise_feature_means_and_covs", self.train_clients
        )
        for client_id in self.train_clients:
            features_means.append(client_packages[client_id]["means"])
            features_covs.append(client_packages[client_id]["covs"])
            features_count.append(client_packages[client_id]["counts"])

        num_classes = NUM_CLASSES[self.args.dataset.name]
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
        self, classes_mean: list[torch.Tensor], classes_cov: list[torch.Tensor]
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

        self.model.to(self.device)
        self.model.train()
        dataset = RepresentationDataset(data, targets)
        dataloader = DataLoader(
            dataset, batch_size=self.args.common.batch_size, shuffle=True
        )
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(
            self.model.classifier.parameters(), lr=self.args.optimizer.lr
        )

        self.model.load_state_dict(self.public_model_params, strict=False)
        for x, y in dataloader:
            logits = self.model.classifier(x)
            loss = criterion(logits, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        model_params = self.model.state_dict()
        self.public_model_params.update(
            (key, model_params[key]) for key in self.public_model_param_names
        )
