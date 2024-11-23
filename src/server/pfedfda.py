from argparse import ArgumentParser, Namespace
from typing import Any, Dict, List, Optional

import torch
from omegaconf import DictConfig

from src.client.pfedfda import pFedFDAClient
from src.server.fedavg import FedAvgServer
from src.utils.constants import NUM_CLASSES


class pFedFDAServer(FedAvgServer):
    @staticmethod
    def get_hyperparams(arg_list: Optional[List[str]] = None) -> Namespace:
        parser = ArgumentParser()
        parser.add_argument("--eps", type=float, default=1e-4)
        parser.add_argument("--single_beta", type=bool, default=False)
        parser.add_argument("--local_beta", type=bool, default=False)
        parser.add_argument("--num_cv_folds", type=int, default=2)
        return parser.parse_args(arg_list)

    def __init__(
        self,
        args: DictConfig,
        algorithm_name: str = "pFedFDA",
        unique_model=False,
        use_fedavg_client_cls=False,
        return_diff=False,
    ):
        super().__init__(
            args, algorithm_name, unique_model, use_fedavg_client_cls, return_diff
        )
        self.init_trainer(pFedFDAClient)

        self.global_means = torch.rand(
            [NUM_CLASSES[self.args.dataset.name], self.model.classifier.in_features]
        )
        self.global_covariances = torch.eye(self.model.classifier.in_features)

        # initialize client-specific variables
        # note that these variables should be managed by each client,
        # but for convenience and memory efficiency (may have more than one client object in parallel mode),
        # FL-bench decides to manage them at server side
        self.client_local_means: Dict[int, torch.Tensor] = {}
        self.client_local_covariances: Dict[int, torch.Tensor] = {}
        self.client_adaptive_means: Dict[int, torch.Tensor] = {}
        self.client_adaptive_covariances: Dict[int, torch.Tensor] = {}
        self.client_means_beta: Dict[int, torch.Tensor] = {}
        self.client_covariances_beta: Dict[int, torch.Tensor] = {}

        for i in range(self.client_num):
            self.client_local_means[i] = self.global_means.clone()
            self.client_local_covariances[i] = self.global_covariances.clone()
            self.client_adaptive_means[i] = self.global_means.clone()
            self.client_adaptive_covariances[i] = self.global_covariances.clone()
            self.client_means_beta[i] = torch.ones(NUM_CLASSES[self.args.dataset.name])
            self.client_covariances_beta[i] = torch.tensor(0.5)

    def package(self, client_id: int):
        package = super().package(client_id)
        package["global_means"] = self.global_means
        package["global_covariances"] = self.global_covariances
        package["local_means"] = self.client_local_means[client_id]
        package["local_covariances"] = self.client_local_covariances[client_id]
        package["adaptive_means"] = self.client_adaptive_means[client_id]
        package["adaptive_covariances"] = self.client_adaptive_covariances[client_id]
        package["means_beta"] = self.client_means_beta[client_id]
        package["covariances_beta"] = self.client_covariances_beta[client_id]
        return package

    def aggregate(self, client_packages: Dict[str, Dict[str, Any]]):
        # common aggregation
        super().aggregate(client_packages)

        # save client-specific variables
        for client_id, package in client_packages.items():
            self.client_local_means[client_id] = package["local_means"]
            self.client_local_covariances[client_id] = package["local_covariances"]
            self.client_adaptive_means[client_id] = package["adaptive_means"]
            self.client_adaptive_covariances[client_id] = package[
                "adaptive_covariances"
            ]
            self.client_means_beta[client_id] = package["means_beta"]
            self.client_covariances_beta[client_id] = package["covariances_beta"]

        # aggregate gaussian estimates
        weights = []
        client_adaptive_means = []
        client_adaptive_covariances = []
        for package in client_packages.values():
            weights.append(package["weight"])
            client_adaptive_means.append(package["adaptive_means"])
            client_adaptive_covariances.append(package["adaptive_covariances"])

        weights = torch.tensor(weights) / sum(weights)
        client_adaptive_means = (
            torch.stack(client_adaptive_means, dim=-1) * weights
        ).sum(dim=-1)
        client_adaptive_covariances = (
            torch.stack(client_adaptive_covariances, dim=-1) * weights
        ).sum(dim=-1)

        self.global_means = client_adaptive_means
        self.global_covariances = client_adaptive_covariances
