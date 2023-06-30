from argparse import ArgumentParser, Namespace
from typing import List

import torch
import numpy as np
from sklearn.cluster import AgglomerativeClustering

from fedavg import FedAvgServer, get_fedavg_argparser
from src.config.utils import vectorize


def get_cfl_argparser() -> ArgumentParser:
    parser = get_fedavg_argparser()
    parser.add_argument("--eps_1", type=float, default=0.4)
    parser.add_argument("--eps_2", type=float, default=1.6)
    parser.add_argument("--min_cluster_size", type=int, default=2)
    parser.add_argument("--start_clustering_round", type=int, default=20)
    return parser


class CFLServer(FedAvgServer):
    def __init__(
        self,
        algo: str = "CFL",
        args: Namespace = None,
        unique_model=True,
        default_trainer=True,
    ):
        if args is None:
            args = get_cfl_argparser().parse_args()
        super().__init__(algo, args, unique_model, default_trainer)
        assert (
            len(self.train_clients) == self.client_num
        ), "CFL doesn't support `User` type split."

        self.test_flag = True
        self.delta_list = [None for _ in self.train_clients]
        self.similarity_matrix = np.eye(len(self.train_clients))
        self.client_clusters = [list(range(len(self.train_clients)))]

    def train_one_round(self):
        for client_id in self.selected_clients:
            client_local_params = self.generate_client_params(client_id)

            (
                delta,
                _,
                self.client_stats[client_id][self.current_epoch],
            ) = self.trainer.train(
                client_id=client_id,
                local_epoch=self.clients_local_epoch[client_id],
                new_parameters=client_local_params,
                verbose=((self.current_epoch + 1) % self.args.verbose_gap) == 0,
            )
            self.delta_list[client_id] = [
                -diff.detach().clone() for diff in delta.values()
            ]

        self.compute_pairwise_similarity()
        client_clusters_new = []
        for indices in self.client_clusters:
            max_norm = compute_max_delta_norm([self.delta_list[i] for i in indices])
            mean_norm = compute_mean_delta_norm([self.delta_list[i] for i in indices])

            if (
                mean_norm < self.args.eps_1
                and max_norm > self.args.eps_2
                and len(indices) > self.args.min_cluster_size
                and self.current_epoch >= self.args.start_clustering_round
            ):
                cluster_1, cluster_2 = self.cluster_clients(
                    self.similarity_matrix[indices][:, indices]
                )
                client_clusters_new += [cluster_1, cluster_2]

            else:
                client_clusters_new += [indices]

        self.client_clusters = client_clusters_new
        self.aggregate_clusterwise()

    @torch.no_grad()
    def compute_pairwise_similarity(self):
        self.similarity_matrix = np.eye(len(self.train_clients))
        for i, delta_a in enumerate(self.delta_list):
            for j, delta_b in enumerate(self.delta_list[i + 1 :], i + 1):
                if delta_a is not None and delta_b is not None:
                    score = torch.cosine_similarity(
                        vectorize(delta_a), vectorize(delta_b), dim=0, eps=1e-12
                    ).item()
                    self.similarity_matrix[i, j] = score
                    self.similarity_matrix[j, i] = score

    def cluster_clients(self, similarities):
        clustering = AgglomerativeClustering(
            metric="precomputed", linkage="complete"
        ).fit(-similarities)

        cluster_1 = np.argwhere(clustering.labels_ == 0).flatten()
        cluster_2 = np.argwhere(clustering.labels_ == 1).flatten()
        return cluster_1, cluster_2

    @torch.no_grad()
    def aggregate_clusterwise(self):
        for cluster in self.client_clusters:
            delta_list = [
                self.delta_list[i] for i in cluster if self.delta_list[i] is not None
            ]
            aggregated_delta = [
                torch.stack(diff).mean(dim=0) for diff in zip(*delta_list)
            ]
            for i in cluster:
                for param, diff in zip(
                    self.client_trainable_params[i], aggregated_delta
                ):
                    param.data += diff

        self.delta_list = [None for _ in self.train_clients]
        if self.current_epoch % 5 == 0:
            print(self.client_clusters)


@torch.no_grad()
def compute_max_delta_norm(delta_list: List[List[torch.Tensor]]):
    flag = False
    for delta in delta_list:
        if delta is not None:
            flag = True
    if flag:
        return max(
            [
                vectorize(delta).norm().item()
                for delta in delta_list
                if delta is not None
            ]
        )
    return 0


@torch.no_grad()
def compute_mean_delta_norm(delta_list: List[List[torch.Tensor]]):
    flag = False
    for delta in delta_list:
        if delta is not None:
            flag = True
    if flag:
        return (
            torch.stack([vectorize(delta) for delta in delta_list if delta is not None])
            .mean(dim=0)
            .norm()
            .item()
        )
    return 0


if __name__ == "__main__":
    server = CFLServer()
    server.run()
