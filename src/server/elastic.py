from argparse import ArgumentParser, Namespace
from collections import OrderedDict
from typing import Any

import torch

from torch._tensor import Tensor
from src.server.fedavg import FedAvgServer
from src.client.elastic import ElasticClient
from src.utils.tools import NestedNamespace, trainable_params


def get_elastic_args(args_list=None) -> Namespace:
    parser = ArgumentParser()
    parser.add_argument("--sample_ratio", type=float, default=0.3)  # opacue
    parser.add_argument("--tau", type=float, default=0.5)
    parser.add_argument("--mu", type=float, default=0.95)
    return parser.parse_args(args_list)


class ElasticServer(FedAvgServer):
    def __init__(
        self,
        args: NestedNamespace,
        algo: str = "Elastic",
        unique_model=False,
        use_fedavg_client_cls=False,
        return_diff=True,
    ):
        super().__init__(args, algo, unique_model, use_fedavg_client_cls, return_diff)
        self.init_trainer(ElasticClient)
        layer_num = len(self.public_model_params)
        self.clients_sensitivity = [torch.zeros(layer_num) for _ in self.train_clients]

    def package(self, client_id: int):
        server_package = super().package(client_id)
        server_package["sensitivity"] = self.clients_sensitivity[client_id]
        return server_package

    def aggregate(self, clients_package: OrderedDict[int, dict[str, Any]]):
        sensitivities = []
        weights = []
        for package in clients_package.values():
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
                    for package in clients_package.values()
                ],
                dim=-1,
            )
            aggregated = torch.sum(diffs * weights, dim=-1)

            global_param.data -= (coef * aggregated).to(global_param.dtype)
