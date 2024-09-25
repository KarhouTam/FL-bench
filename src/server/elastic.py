from argparse import ArgumentParser, Namespace
from collections import OrderedDict
from typing import Any

import torch
from omegaconf import DictConfig
from torch._tensor import Tensor

from src.client.elastic import ElasticClient
from src.server.fedavg import FedAvgServer


class ElasticServer(FedAvgServer):
    @staticmethod
    def get_hyperparams(args_list=None) -> Namespace:
        parser = ArgumentParser()
        parser.add_argument("--sample_ratio", type=float, default=0.3)  # opacue
        parser.add_argument("--tau", type=float, default=0.5)
        parser.add_argument("--mu", type=float, default=0.95)
        return parser.parse_args(args_list)

    def __init__(
        self,
        args: DictConfig,
        algorithm_name: str = "Elastic",
        unique_model=False,
        use_fedavg_client_cls=False,
        return_diff=True,
    ):
        super().__init__(
            args, algorithm_name, unique_model, use_fedavg_client_cls, return_diff
        )
        self.init_trainer(ElasticClient)

    def aggregate(self, client_packages: OrderedDict[int, dict[str, Any]]):
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
