from argparse import ArgumentParser
from collections import OrderedDict
from typing import Any

import torch
import numpy as np
import cvxpy as cvx

from src.server.fedavg import FedAvgServer
from src.utils.constants import NUM_CLASSES
from src.client.fedpac import FedPACClient
from src.utils.tools import NestedNamespace


def get_fedpac_args(args_list=None):
    parser = ArgumentParser()
    parser.add_argument("--train_classifier_round", type=int, default=1)
    parser.add_argument("--lamda", type=float, default=1.0)
    return parser.parse_args(args_list)


class FedPACServer(FedAvgServer):
    def __init__(
        self,
        args: NestedNamespace,
        algo: str = "FedPAC",
        unique_model=False,
        use_fedavg_client_cls=False,
        return_diff=False,
    ):
        super().__init__(args, algo, unique_model, use_fedavg_client_cls, return_diff)
        self.global_prototypes = {}
        self.init_trainer(FedPACClient)

    def package(self, client_id: int):
        server_package = super().package(client_id)
        server_package["global_prototypes"] = self.global_prototypes
        return server_package

    def train_one_round(self):
        client_packages = self.trainer.train()
        self.aggregate_prototypes(client_packages)
        self.aggregate(client_packages)

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
            weights = torch.tensor(list(filter(lambda w: w > 0, weights)), dtype=float)
            if len(prototypes) > 0:
                # because prototypes are averaged
                weights /= weights.sum()
                prototypes = torch.stack(prototypes, dim=-1)
                self.global_prototypes[i] = torch.sum(prototypes * weights, dim=-1)

    def aggregate(self, client_packages: OrderedDict[int, dict[str, Any]]):
        # aggregate all parameters (include classifier's)
        super().aggregate(client_packages)

        # aggregate personalized classifier and store parameters in self.client_personal_model_params
        # personal params is prior to the globals, results in personalized classifier params overlap the globals
        num_clients = len(self.selected_clients)
        client_h_ref_list = [package["h_ref"] for package in client_packages.values()]
        client_v_list = [package["v"] for package in client_packages.values()]
        feature_length = client_h_ref_list[0].shape[1]
        classifier_weights = {}
        for i in range(num_clients):
            V = torch.tensor(client_v_list)
            h_ref = client_h_ref_list[i]
            distances = torch.zeros((num_clients, num_clients))
            for j, k in pairwise(tuple(range(num_clients))):
                h_j1 = client_h_ref_list[j]
                h_j2 = client_h_ref_list[k]
                H = torch.zeros((feature_length, feature_length))
                for k in range(NUM_CLASSES[self.args.common.dataset]):
                    H += torch.mm(
                        (h_ref[k] - h_j1[k]).reshape(feature_length, 1),
                        (h_ref[k] - h_j2[k]).reshape(1, feature_length),
                    )
                dist_jk = torch.trace(H)
                distances[j][k] = dist_jk
                distances[k][j] = dist_jk

            W = None
            p_matrix = torch.diag(V) + distances

            if np.all(np.linalg.eigvals(p_matrix.numpy()) >= 0.0):
                alpha = cvx.Variable(num_clients)
                obj = cvx.Minimize(cvx.quad_form(alpha, p_matrix))
                prob = cvx.Problem(obj, [cvx.sum(alpha) == 1.0, alpha >= 0])
                prob.solve()
                W = [
                    i * (i > 1e-3) for i in alpha.value
                ]  # zero-out small weights (<eps)
            else:
                # QP solver
                eigenvals, eigenvecs = torch.linalg.eig(p_matrix)
                eigenvals = eigenvals.real
                eigenvecs = eigenvecs.real

                # for numerical stablity
                p_matrix.zero_()
                for j in range(num_clients):
                    if eigenvals[j] >= 0.01:
                        p_matrix += eigenvals[j] * torch.mm(
                            eigenvecs[:, j].reshape(num_clients, 1),
                            eigenvecs[:, j].reshape(1, num_clients),
                        )

                if np.all(np.linalg.eigvals(p_matrix.numpy()) >= 0.0):
                    alpha = cvx.Variable(num_clients)
                    obj = cvx.Minimize(cvx.quad_form(alpha, p_matrix))
                    prob = cvx.Problem(obj, [cvx.sum(alpha) == 1.0, alpha >= 0])
                    prob.solve()
                    W = [
                        i * (i > 1e-3) for i in alpha.value
                    ]  # zero-out small weights (<eps)
                else:
                    W = None
            if W is None:
                W = [0.0] * num_clients
                W[i] = 1.0
            W = torch.tensor(W, dtype=torch.float)
            classifier_weights[self.selected_clients[i]] = W / W.sum()

        classifier_params_name = [
            name for name in self.trainable_params_name if "classifier" in name
        ]
        for layer in classifier_params_name:
            parameters = torch.stack(
                [
                    package["regular_model_params"][layer]
                    for package in client_packages.values()
                ],
                dim=-1,
            )
            for i in self.selected_clients:
                self.clients_personal_model_params[i][layer] = (
                    parameters * classifier_weights[i]
                ).sum(dim=-1)


def pairwise(sequence):
    n = len(sequence)
    for i in range(n):
        for j in range(i, n):
            yield (sequence[i], sequence[j])
