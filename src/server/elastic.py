from argparse import ArgumentParser, Namespace
from collections import OrderedDict
from typing import Any

import torch

from src.client.elastic import ElasticClient
from src.server.fedavg import FedAvgServer


class ElasticServer(FedAvgServer):
    algorithm_name: str = "Elastic"
    all_model_params_personalized = False  # `True` indicates that clients have their own fullset of personalized model parameters.
    return_diff = True  # `True` indicates that clients return `diff = W_global - W_local` as parameter update; `False` for `W_local` only.
    client_cls = ElasticClient

    @staticmethod
    def get_hyperparams(args_list=None) -> Namespace:
        parser = ArgumentParser()
        parser.add_argument("--sample_ratio", type=float, default=0.3)  # opacue
        parser.add_argument("--tau", type=float, default=0.5)
        parser.add_argument("--mu", type=float, default=0.95)
        return parser.parse_args(args_list)

    def aggregate_client_updates(
        self, client_packages: OrderedDict[int, dict[str, Any]]
    ):
        sensitivities = []
        weights = []
        for package in client_packages.values():
            sensitivities.append(package["sensitivity"])
            weights.append(package["weight"])

        weights = torch.tensor(weights) / sum(weights)
        sensitivities = torch.stack(sensitivities, dim=-1)

        aggregated_sensitivity = torch.sum(sensitivities * weights, dim=-1)
        max_sensitivity = sensitivities.max(dim=-1)[0]

        zeta = 1 + self.args.elastic.tau - aggregated_sensitivity / max_sensitivity

        for (key, global_param), coef in zip(self.public_model_params.items(), zeta):
            diffs = torch.stack(
                [
                    package["model_params_diff"][key]
                    for package in client_packages.values()
                ],
                dim=-1,
            )
            aggregated = torch.sum(diffs * weights, dim=-1)
            global_param.data -= coef * aggregated
