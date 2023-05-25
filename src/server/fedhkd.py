from argparse import Namespace
from copy import deepcopy
from typing import List, Tuple, Union

import torch

from fedavg import FedAvgServer
from src.config.args import get_fedhkd_argparser
from src.client.fedhkd import FedHKDClient


class FedHKDServer(FedAvgServer):
    def __init__(
        self,
        algo: str = "FedHKD",
        args: Namespace = None,
        unique_model=False,
        default_trainer=False,
    ):
        if args is None:
            args = get_fedhkd_argparser().parse_args()
        super().__init__(algo, args, unique_model, default_trainer)
        self.trainer = FedHKDClient(deepcopy(self.model), self.args, self.logger)
        self.global_hyper_knowledge = (None, None)

    def train_one_round(self):
        delta_cache = []
        weight_cache = []
        data_count_cache = []
        H_cache = []
        Q_cache = []
        for client_id in self.selected_clients:
            client_local_params = self.generate_client_params(client_id)
            (
                delta,
                hyper_knowledge,
                data_count,
                self.client_stats[client_id][self.current_epoch],
            ) = self.trainer.train(
                client_id=client_id,
                new_parameters=client_local_params,
                global_hyper_knowledge=self.global_hyper_knowledge,
                verbose=((self.current_epoch + 1) % self.args.verbose_gap) == 0,
            )
            data_count_cache.append(data_count)
            delta_cache.append(delta)
            weight_cache.append(sum(data_count))
            H_cache.append(hyper_knowledge[0])
            Q_cache.append(hyper_knowledge[1])

        self.aggregate(delta_cache, weight_cache)
        self.aggregate_hyper_knowledge(H_cache, Q_cache, data_count_cache)

    def aggregate_hyper_knowledge(
        self,
        H_cache: List[Tuple[Union[None, torch.Tensor]]],
        Q_cache: List[Tuple[Union[None, torch.Tensor]]],
        data_count_cache: List[List[int]],
    ):
        # prune data_count of each selected client
        def prune(data_count: List[int], threshold: int):
            pruned = torch.tensor(data_count)
            pruned = pruned * (pruned > threshold)
            return pruned.tolist()

        thresholds = [int(sum(cnt) * self.args.threshold) for cnt in data_count_cache]
        pruned_data_count = [
            prune(data_count, threshold)
            for data_count, threshold in zip(data_count_cache, thresholds)
        ]
        H_list = []
        Q_list = []

        for i, (count, h, q) in enumerate(
            zip(zip(*pruned_data_count), zip(*H_cache), zip(*Q_cache))
        ):
            weight = torch.tensor(count, device=self.device) / (sum(count) + 1e-12)
            H_list.append((torch.stack(h, dim=-1) * weight).sum(dim=-1))
            Q_list.append((torch.stack(q, dim=-1) * weight).sum(dim=-1))

        if self.global_hyper_knowledge == (None, None):
            self.global_hyper_knowledge = (torch.stack(H_list), torch.stack(Q_list))
        else:
            for i, (h, q) in enumerate(zip(H_list, Q_list)):
                if h.sum() != 0:
                    self.global_hyper_knowledge[0][i].copy_(h)
                if q.sum() != 0:
                    self.global_hyper_knowledge[1][i].copy_(q)


if __name__ == "__main__":
    server = FedHKDServer()
    server.run()
