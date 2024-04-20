from argparse import ArgumentParser, Namespace
from collections import OrderedDict

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
        layer_num = len(trainable_params(self.model))
        self.clients_sensitivity = [torch.zeros(layer_num) for _ in self.train_clients]

    def package(self, client_id: int):
        server_package = super().package(client_id)
        server_package["sensitivity"] = self.clients_sensitivity[client_id]
        return server_package

    def train_one_round(self):
        clients_package = self.trainer.train()

        clients_params_diff = []
        clients_weight = []
        clients_sensitivity = []

        for cid in self.selected_clients:
            self.clients_sensitivity[cid] = clients_package[cid]["sensitivity"]
            clients_params_diff.append(clients_package[cid]["model_params_diff"])
            clients_weight.append(clients_package[cid]["weight"])
            clients_sensitivity.append(clients_package[cid]["sensitivity"])

        self.aggregate(clients_weight, clients_params_diff, clients_sensitivity)

    def aggregate(
        self,
        clients_weight: list[int],
        clients_params_diff: list[OrderedDict[str, Tensor]],
        clients_sensitivity: list[torch.Tensor],
    ):
        weights = torch.tensor(clients_weight) / sum(clients_weight)
        stacked_sensitivity = torch.stack(clients_sensitivity, dim=-1)
        aggregated_sensitivity = torch.sum(stacked_sensitivity * weights, dim=-1)
        max_sensitivity = stacked_sensitivity.max(dim=-1)[0]
        zeta = 1 + self.args.elastic.tau - aggregated_sensitivity / max_sensitivity
        clients_params_diff_list = [
            list(delta.values()) for delta in clients_params_diff
        ]
        aggregated_diff = [
            torch.sum(weights * torch.stack(diff, dim=-1), dim=-1)
            for diff in zip(*clients_params_diff_list)
        ]

        for param, coef, diff in zip(
            self.global_model_params.values(), zeta, aggregated_diff
        ):
            param.data -= coef * diff
        self.model.load_state_dict(self.global_model_params, strict=False)
