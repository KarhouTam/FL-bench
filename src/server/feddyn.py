from argparse import ArgumentParser, Namespace
from collections import OrderedDict
from typing import Any

import torch
from omegaconf import DictConfig

from src.client.feddyn import FedDynClient
from src.server.fedavg import FedAvgServer
from src.utils.functional import vectorize


# Fixed according to FedDyn implementation in FL-Simulator (issue #133)
class FedDynServer(FedAvgServer):
    algorithm_name = "FedDyn"
    all_model_params_personalized = False
    return_diff = True  # `True` indicates that clients return `diff = W_global - W_local` as parameter update; `False` for `W_local` only.
    client_cls = FedDynClient

    @staticmethod
    def get_hyperparams(args_list=None) -> Namespace:
        parser = ArgumentParser()
        parser.add_argument("--alpha", type=float, default=0.1)
        parser.add_argument("--max_grad_norm", type=float, default=10)
        return parser.parse_args(args_list)

    def __init__(self, args: DictConfig):
        super().__init__(args)
        param_numel = vectorize(self.public_model_params).numel()
        self.nabla = torch.zeros(size=(self.client_num, param_numel))

    def package(self, client_id: int):
        server_package = super().package(client_id)
        server_package["local_dual_correction"] = self.nabla[client_id] - vectorize(
            self.public_model_params
        )
        return server_package

    def aggregate_client_updates(
        self, client_packages: OrderedDict[int, dict[str, Any]]
    ):
        super().aggregate_client_updates(client_packages)
        param_shapes = [
            (param.numel(), param.shape) for param in self.public_model_params.values()
        ]

        for client_id, package in client_packages.items():
            # model difference in FL-bench is like diff = param_old - param_new
            # so we do the negative here
            self.nabla[client_id] -= vectorize(package["model_params_diff"])

        flatten_new_params = vectorize(self.public_model_params) + self.nabla.mean(
            dim=0
        )

        # reshape
        new_params = []
        i = 0
        for numel, shape in param_shapes:
            new_params.append(flatten_new_params[i : i + numel].reshape(shape))
            i += numel
        self.public_model_params = OrderedDict(
            zip(self.public_model_params.keys(), new_params)
        )
