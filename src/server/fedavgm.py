from argparse import ArgumentParser, Namespace
from typing import Any, OrderedDict

import torch
from omegaconf import DictConfig

from src.server.fedavg import FedAvgServer


class FedAvgMServer(FedAvgServer):
    algorithm_name: str = "FedAvgM"
    all_model_params_personalized = False  # `True` indicates that clients have their own fullset of personalized model parameters.
    return_diff = True  # `True` indicates that clients return `diff = W_global - W_local` as parameter update; `False` for `W_local` only.

    @staticmethod
    def get_hyperparams(args_list=None) -> Namespace:
        parser = ArgumentParser()
        parser.add_argument("--server_momentum", type=float, default=0.9)
        return parser.parse_args(args_list)

    def __init__(self, args: DictConfig):
        super().__init__(args)

        self.global_optmizer = torch.optim.SGD(
            list(self.public_model_params.values()),
            lr=1.0,
            momentum=self.args.fedavgm.server_momentum,
            nesterov=True,
        )

    @torch.no_grad()
    def aggregate_client_updates(
        self, client_packages: OrderedDict[int, dict[str, Any]]
    ):
        self.global_optmizer.zero_grad()

        client_weights = [package["weight"] for package in client_packages.values()]
        weights = torch.tensor(client_weights) / sum(client_weights)
        for key in self.public_model_params.keys():
            if "num_batches_tracked" not in key:
                diffs = torch.stack(
                    [
                        package["model_params_diff"][key]
                        for package in client_packages.values()
                    ],
                    dim=-1,
                )
                aggregated = torch.sum(diffs * weights, dim=-1)
                self.public_model_params[key].grad = aggregated

        self.global_optmizer.step()
