from argparse import ArgumentParser
from collections import OrderedDict
from typing import Any

import torch
import numpy as np
import cvxpy as cvx

from src.server.fedavg import FedAvgServer
from src.utils.constants import NUM_CLASSES
from src.client.fedpac import FedPACClient
from utils.tools import NestedNamespace


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

    def train_one_round(self):
        client_packages = self.trainer.train()
        self.aggregate_prototypes(client_packages)
        self.aggregate(client_packages)

    def aggregate_prototypes(self, client_packages: list[dict[str, Any]]):
        prototypes_list = [
            package["prototypes"] for package in client_packages.values()
        ]
        weights_list = {i: [] for i in range(NUM_CLASSES[self.args.common.dataset])}
        self.global_prototypes = {}
        for i, (prototypes, weights) in enumerate(
            zip(zip(*prototypes_list), zip(*weights_list))
        ):
            prototypes = list(filter(lambda p: p is not None, prototypes))
            weights = torch.tensor(list(filter(lambda w: w > 0, weights)), dtype=float)
            if len(prototypes) > 0:
                # because prototypes are averaged
                weights = weights * (weights / weights.sum())
                prototypes = torch.stack(prototypes, dim=-1)
                self.global_prototypes[i] = torch.sum(prototypes * weights, dim=-1)

    def aggregate(self, client_packages: OrderedDict[int, dict[str, Any]]):
        # aggregate all parameters (include classifier's)
        super().aggregate(client_packages)
        # aggregate classifier's parameters (overlap super().aggregate() result)
        num_clients = len(client_packages) 
        client_h_ref_list = [package["h_ref"] for package in client_packages]
        client_v_list = [package["v"] for package in client_packages]
        feature_length = client_h_ref_list[0].shape[1]
        classifier_weights = []
        for i in range(num_clients):
            # ---------------------------------------------------------------------------
            # variance ter
            v = torch.tensor(client_v_list)
            # ---------------------------------------------------------------------------
            # bias term
            h_ref = client_h_ref_list[i]
            dist = torch.zeros((num_clients, num_clients))
            for j1, j2 in pairwise(tuple(range(num_clients))):
                h_j1 = client_h_ref_list[j1]
                h_j2 = client_h_ref_list[j2]
                h = torch.zeros((feature_length, feature_length))
                for k in range(NUM_CLASSES[self.args.common.dataset]):
                    h += torch.mm(
                        (h_ref[k] - h_j1[k]).reshape(feature_length, 1),
                        (h_ref[k] - h_j2[k]).reshape(1, feature_length),
                    )
                dj12 = torch.trace(h)
                dist[j1][j2] = dj12
                dist[j2][j1] = dj12

            # QP solver
            p_matrix = torch.diag(v) + dist
            p_matrix = p_matrix.numpy()  # coefficient for QP problem
            evals, evecs = torch.eig(torch.tensor(p_matrix), eigenvectors=True)

            # for numerical stablity
            p_matrix_new = 0
            p_matrix_new = 0
            for ii in range(num_clients):
                if evals[ii, 0] >= 0.01:
                    p_matrix_new += evals[ii, 0] * torch.mm(
                        evecs[:, ii].reshape(num_clients, 1),
                        evecs[:, ii].reshape(1, num_clients),
                    )
            p_matrix = (
                p_matrix_new.numpy()
                if not np.all(np.linalg.eigvals(p_matrix) >= 0.0)
                else p_matrix
            )

            # solve QP
            W = 0
            if np.all(np.linalg.eigvals(p_matrix) >= 0):
                alpha = cvx.Variable(num_clients)
                obj = cvx.Minimize(cvx.quad_form(alpha, p_matrix))
                prob = cvx.Problem(obj, [cvx.sum(alpha) == 1.0, alpha >= 0])
                prob.solve()
                W = alpha.value
                W = [(i) * (i > 1e-3) for i in W]  # zero-out small weights (<eps)
            else:
                W = None

            classifier_weights.append(W)


def pairwise(sequence):
    n = len(sequence)
    for i in range(n):
        for j in range(i, n):
            yield (sequence[i], sequence[j])
