from argparse import Namespace
from typing import List

import torch
import numpy as np
from sklearn.cluster import AgglomerativeClustering

from fedavg import FedAvgServer
from src.config.args import get_cfl_argparser


class ClusteredFL(FedAvgServer):
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
            len(self.train_clients) == self.client_num_in_total
        ), "CFL doesn't support `User` type split."

        self.test_flag = True
        self.delta_list = [None] * len(self.train_clients)
        self.similarity_matrix = np.eye(len(self.train_clients))
        self.client_clusters = [list(range(len(self.train_clients)))]

    def train(self):
        for E in self.train_progress_bar:
            self.current_epoch = E

            if (E + 1) % self.args.verbose_gap == 0:
                self.logger.log(" " * 30, f"TRAINING EPOCH: {E + 1}", " " * 30)

            if (E + 1) % self.args.test_gap == 0:
                self.tesst_flag = True
                self.test()
                self.tesst_flag = False

            self.selected_clients = self.client_sample_stream[E]

            for client_id in self.selected_clients:

                client_local_params = self.generate_client_params(client_id)

                delta, _, self.client_stats[client_id][E] = self.trainer.train(
                    client_id=client_id,
                    new_parameters=client_local_params,
                    verbose=((E + 1) % self.args.verbose_gap) == 0,
                )
                self.delta_list[client_id] = [
                    diff.detach().to(self.device) for diff in delta.values()
                ]

            self.compute_pairwise_similarity()
            client_clusters_new = []
            for indices in self.client_clusters:
                max_norm = compute_max_delta_norm([self.delta_list[i] for i in indices])
                mean_norm = compute_mean_delta_norm(
                    [self.delta_list[i] for i in indices]
                )

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

            self.log_info()

    @torch.no_grad()
    def compute_pairwise_similarity(self):
        self.similarity_matrix = np.eye(len(self.train_clients))
        for i, delta_a in enumerate(self.delta_list):
            for j, delta_b in enumerate(self.delta_list[i + 1 :], i + 1):
                if delta_a is not None and delta_b is not None:
                    score = torch.cosine_similarity(
                        flatten_and_concat(delta_a),
                        flatten_and_concat(delta_b),
                        dim=0,
                        eps=1e-12,
                    ).item()
                    self.similarity_matrix[i, j] = score
                    self.similarity_matrix[j, i] = score

    def cluster_clients(self, similarities):
        clustering = AgglomerativeClustering(
            affinity="precomputed", linkage="complete"
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
                torch.stack(diff).mean(dim=0).to(self.device)
                for diff in zip(*delta_list)
            ]
            for i in cluster:
                for param, diff in zip(
                    self.client_trainable_params[i], aggregated_delta
                ):
                    param.data -= diff


@torch.no_grad()
def compute_max_delta_norm(delta_list: List[List[torch.Tensor]]):
    return max(
        [
            flatten_and_concat(delta).norm().item()
            for delta in delta_list
            if delta is not None
        ]
    )


@torch.no_grad()
def compute_mean_delta_norm(delta_list: List[List[torch.Tensor]]):
    return (
        torch.stack(
            [flatten_and_concat(delta) for delta in delta_list if delta is not None]
        )
        .mean(dim=0)
        .norm()
        .item()
    )


@torch.no_grad()
def flatten_and_concat(src: List[torch.Tensor]):
    return torch.cat([tensor.flatten() for tensor in src if tensor is not None])


if __name__ == "__main__":
    server = ClusteredFL()
    server.run()
