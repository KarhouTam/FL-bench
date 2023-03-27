from argparse import Namespace
from copy import deepcopy
from typing import List

import torch

from fedavg import FedAvgServer
from src.config.args import get_fedmd_argparser
from src.client.fedmd import FedMDClient


class FedMDServer(FedAvgServer):
    """
    NOTE: FedMD supposes to be a pFL method with heterogeneous models, but this benchmark does not support heterogeneous model settings (for now). As a compromise, the homogeneous model version is offered.

    According to the paper, we can do experiment in 2 settings:
    1. (public: MNIST, private: FEMNIST / EMNIST);
    2. (public: CIFAR10, private: CIFAR100 but under 20 superclasses)
    """

    def __init__(
        self,
        algo: str = "FedMD",
        args: Namespace = None,
        unique_model=True,
        default_trainer=False,
    ):
        if args is None:
            args = get_fedmd_argparser().parse_args()
        super().__init__(algo, args, unique_model, default_trainer)
        self.trainer = FedMDClient(
            model=deepcopy(self.model), args=self.args, logger=self.logger
        )

    def train(self):
        for E in self.train_progress_bar:
            self.current_epoch = E

            if (E + 1) % self.args.verbose_gap == 0:
                self.logger.log(" " * 30, f"TRAINING EPOCH: {E + 1}", " " * 30)

            if (E + 1) % self.args.test_gap == 0:
                self.test()

            self.selected_clients = self.client_sample_stream[E]
            # communicate

            self.trainer.load_public_data_batches()
            scores_cache = []
            for client_id in self.selected_clients:
                client_params = self.generate_client_params(client_id)
                scores_cache.append(self.trainer.get_scores(client_id, client_params))

            # aggregate
            self.trainer.consensus = self.aggregate(scores_cache)

            # digest & revisit
            client_params_cache = []
            for client_id in self.selected_clients:
                client_params = self.generate_client_params(client_id)
                client_params, self.client_stats[client_id][E] = self.trainer.train(
                    client_id=client_id,
                    new_parameters=client_params,
                    verbose=((E + 1) % self.args.verbose_gap) == 0,
                )
                client_params_cache.append(client_params)

            self.update_client_params(client_params_cache)
            self.log_info()

    def aggregate(self, scores_cache: List[torch.Tensor]) -> List[torch.Tensor]:
        consensus = []
        for scores in zip(*scores_cache):
            consensus.append(torch.stack(scores, dim=-1).mean(dim=-1))
        return consensus


if __name__ == "__main__":
    server = FedMDServer()
    server.run()
