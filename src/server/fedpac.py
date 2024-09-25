from argparse import ArgumentParser
from collections import OrderedDict
from typing import Any

import cvxpy as cvx
import torch
from omegaconf import DictConfig

from src.client.fedpac import FedPACClient
from src.server.fedavg import FedAvgServer
from src.utils.constants import NUM_CLASSES


class FedPACServer(FedAvgServer):
    @staticmethod
    def get_hyperparams(args_list=None):
        parser = ArgumentParser()
        parser.add_argument("--train_classifier_round", type=int, default=1)
        parser.add_argument("--lamda", type=float, default=1.0)
        parser.add_argument("--classifier_lr", type=float, default=0.1)
        return parser.parse_args(args_list)

    def __init__(
        self,
        args: DictConfig,
        algorithm_name: str = "FedPAC",
        unique_model=False,
        use_fedavg_client_cls=False,
        return_diff=False,
    ):
        super().__init__(
            args, algorithm_name, unique_model, use_fedavg_client_cls, return_diff
        )
        self.global_prototypes = {}
        self.init_trainer(FedPACClient)

    def package(self, client_id: int):
        server_package = super().package(client_id)
        server_package["global_prototypes"] = self.global_prototypes
        return server_package

    def aggregate_prototypes(self, client_packages: list[dict[str, Any]]):
        prototypes_list = [
            package["prototypes"] for package in client_packages.values()
        ]
        weights_list = [
            package["label_distrib"] for package in client_packages.values()
        ]
        self.global_prototypes = {}
        for i, (prototypes, weights) in enumerate(
            zip(zip(*prototypes_list), zip(*weights_list))
        ):
            prototypes = list(filter(lambda p: isinstance(p, torch.Tensor), prototypes))
            weights = torch.tensor(
                list(filter(lambda w: w > 0, weights)),
                dtype=torch.float,
                device=self.device,
            )
            if len(prototypes) > 0:
                weights /= weights.sum()
                prototypes = torch.stack(prototypes, dim=-1).to(self.device)
                self.global_prototypes[i] = torch.sum(
                    prototypes * weights, dim=-1
                ).cpu()

    def aggregate(self, client_packages: OrderedDict[int, dict[str, Any]]):
        # aggregate all parameters (include classifier's)
        super().aggregate(client_packages)

        # aggregate new global prototypes
        self.aggregate_prototypes(client_packages)

        # aggregate personalized classifier and store parameters in self.client_personal_model_params
        # personal params is prior to the globals, results in personalized classifier params overlap the globals
        num_clients = len(self.selected_clients)
        client_h_ref_list = [package["h_ref"] for package in client_packages.values()]
        V = torch.tensor(
            [package["v"] for package in client_packages.values()],
            dtype=torch.float,
            device=self.device,
        )
        feature_length = client_h_ref_list[0].shape[1]
        classifier_weights = {}
        for i, client_id in enumerate(self.selected_clients):
            h_i = client_packages[client_id]["h_ref"].to(self.device)
            distances = torch.zeros((num_clients, num_clients), device=self.device)
            for (idx_j, idx_k), (j, k) in pairwise(self.selected_clients):
                h_j = client_packages[j]["h_ref"].to(self.device)
                h_k = client_packages[k]["h_ref"].to(self.device)
                H = torch.zeros((feature_length, feature_length), device=self.device)
                for c in range(NUM_CLASSES[self.args.dataset.name]):
                    H += torch.mm(
                        (h_i[c] - h_j[c]).reshape(feature_length, 1),
                        (h_i[c] - h_k[c]).reshape(1, feature_length),
                    )
                dist_jk = torch.trace(H)
                distances[idx_j][idx_k] = dist_jk
                distances[idx_k][idx_j] = dist_jk

            W = [int(idx == i) for idx in range(num_clients)]
            P = (torch.diag(V) + distances).cpu()

            if torch.all(torch.linalg.eigvalsh(P) >= 0.0):
                W = calculate_classifier_weights(num_clients, P, i)
            else:
                # QP solver
                eigenvals, eigenvecs = torch.linalg.eigh(P)
                eigenvals = eigenvals.to(self.device)
                eigenvecs = eigenvecs.to(self.device)
                # for numerical stablity
                P = torch.zeros((num_clients, num_clients), device=self.device)
                for j in range(num_clients):
                    if eigenvals[j] >= 0.01:
                        P += eigenvals[j] * torch.mm(
                            eigenvecs[:, j].reshape(num_clients, 1),
                            eigenvecs[:, j].reshape(1, num_clients),
                        )

                P = P.cpu()
                if torch.all(torch.linalg.eigvalsh(P) >= 0.0):
                    W = calculate_classifier_weights(num_clients, P, i)

            W = torch.tensor(W, dtype=torch.float, device=self.device)
            classifier_weights[client_id] = W / W.sum()

        classifier_params_name = [
            name for name in self.public_model_param_names if "classifier" in name
        ]
        for layer in classifier_params_name:
            parameters = torch.stack(
                [
                    package["regular_model_params"][layer]
                    for package in client_packages.values()
                ],
                dim=-1,
            ).to(self.device)
            for i in self.selected_clients:
                self.clients_personal_model_params[i][layer] = (
                    (parameters * classifier_weights[i]).sum(dim=-1).cpu()
                )


def calculate_classifier_weights(num_clients: int, P: torch.Tensor, idx: int):
    try:
        alpha = cvx.Variable(num_clients)
        obj = cvx.Minimize(cvx.quad_form(alpha, P.numpy()))
        prob = cvx.Problem(obj, [cvx.sum(alpha) == 1.0, alpha >= 0])
        prob.solve()
        W = [i * (i > 1e-3) for i in alpha.value]
    except cvx.error.DCPError:
        W = [int(idx == i) for i in range(num_clients)]
    return W


def pairwise(sequence):
    n = len(sequence)
    for i in range(n):
        for j in range(i, n):
            yield ((i, j), (sequence[i], sequence[j]))
