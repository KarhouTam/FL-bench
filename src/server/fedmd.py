from argparse import ArgumentParser, Namespace
from copy import deepcopy
from typing import List

import torch

from fedavg import FedAvgServer, get_fedavg_argparser
from src.client.fedmd import FedMDClient


def get_fedmd_argparser() -> ArgumentParser:
    parser = get_fedavg_argparser()
    parser.add_argument("--digest_epoch", type=int, default=1)
    parser.add_argument("--public_dataset", type=str, default="mnist")
    parser.add_argument("--public_batch_size", type=int, default=32)
    parser.add_argument("--public_batch_num", type=int, default=5)
    return parser


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
        if self.args.public_dataset == "mnist" and self.args.dataset not in [
            "femnist",
            "emnist",
        ]:
            raise NotImplementedError(
                "The public dataset is mnist and the --dataset should be in [femnist, emnist] (now: {})".format(
                    self.args.dataset
                )
            )
        elif self.args.public_dataset == "cifar10" and self.args.dataset != "cifar100":
            raise NotImplementedError(
                "The public dataset is cifar10 and the --dataset should be cifar100 (now: {})".format(
                    self.args.dataset
                )
            )
        self.trainer = FedMDClient(
            model=deepcopy(self.model),
            args=self.args,
            logger=self.logger,
            device=self.device,
        )

    def train_one_round(self):
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
            (
                client_params,
                self.client_metrics[client_id][self.current_epoch],
            ) = self.trainer.train(
                client_id=client_id,
                local_epoch=self.clients_local_epoch[client_id],
                new_parameters=client_params,
                verbose=((self.current_epoch + 1) % self.args.verbose_gap) == 0,
            )
            client_params_cache.append(client_params)

        self.update_client_params(client_params_cache)

    def aggregate(self, scores_cache: List[torch.Tensor]) -> List[torch.Tensor]:
        consensus = []
        for scores in zip(*scores_cache):
            consensus.append(torch.stack(scores, dim=-1).mean(dim=-1))
        return consensus


if __name__ == "__main__":
    server = FedMDServer()
    server.run()
