from argparse import ArgumentParser, Namespace
from collections import OrderedDict
from copy import deepcopy

import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import DictConfig

from src.server.fedavg import FedAvgServer


class pFedLAServer(FedAvgServer):
    @staticmethod
    def get_hyperparams(args_list=None) -> Namespace:
        parser = ArgumentParser()
        parser.add_argument("--k", type=int, default=0)
        parser.add_argument("--hn_lr", type=float, default=5e-3)
        parser.add_argument("--hn_momentum", type=float, default=0.0)
        parser.add_argument("--embedding_dim", type=int, default=100)
        parser.add_argument("--hidden_dim", type=int, default=100)
        return parser.parse_args(args_list)

    def __init__(
        self,
        args: DictConfig,
        algorithm_name: str = None,
        unique_model=True,
        use_fedavg_client_cls=True,
        return_diff=True,
    ):
        if args.mode == "parallel":
            raise NotImplementedError("pFedHN does not support paralell mode.")
        algo = "pFedLA" if args.pfedla.k == 0 else "HeurpFedLA"
        super().__init__(
            args, algorithm_name, unique_model, use_fedavg_client_cls, return_diff
        )
        self.hypernet = HyperNetwork(
            embedding_dim=self.args.pfedla.embedding_dim,
            layer_num=len(self.public_model_param_names),
            client_num=self.client_num,
            hidden_dim=self.args.pfedla.hidden_dim,
            K=self.args.pfedla.k,
        )
        self.clients_hypernet_params = {
            i: deepcopy(self.hypernet.state_dict()) for i in self.train_clients
        }
        self.hypernet_optimizer = torch.optim.SGD(
            self.hypernet.parameters(),
            lr=self.args.pfedla.hn_lr,
            momentum=self.args.pfedla.hn_momentum,
        )
        self.pfedla_aggregated_model_params: list[torch.Tensor] = None

    def train_one_round(self) -> None:
        selected_clients_this_round = self.selected_clients
        for client_id in selected_clients_this_round:
            self.hypernet.load_state_dict(self.clients_hypernet_params[client_id])
            self.selected_clients = [client_id]
            client_package = self.trainer.train()

            self.hypernet_optimizer.zero_grad()
            hn_grads = torch.autograd.grad(
                outputs=self.pfedla_aggregated_model_params,
                inputs=self.hypernet.parameters(),
                grad_outputs=[
                    -diff
                    for diff in client_package[client_id]["model_params_diff"].values()
                ],
                allow_unused=True,
            )
            for param, grad in zip(self.hypernet.parameters(), hn_grads):
                if grad is not None:
                    param.grad = grad
            self.hypernet_optimizer.step()
            self.clients_hypernet_params[client_id] = deepcopy(
                self.hypernet.state_dict()
            )

            for key in self.public_model_param_names:
                self.clients_personal_model_params[client_id][key] -= client_package[
                    client_id
                ]["model_params_diff"][key].data.cpu()

    def get_client_model_params(self, client_id: int):
        aggregated_params = OrderedDict()
        layer_params_dict = OrderedDict(
            zip(
                self.public_model_param_names,
                zip(
                    *[
                        [params_dict[key] for key in self.public_model_param_names]
                        for params_dict in self.clients_personal_model_params.values()
                    ]
                ),
            )
        )
        alpha = self.hypernet(client_id)
        default_weights = torch.zeros(
            self.client_num, dtype=torch.float, device=self.device
        )
        default_weights[client_id] = 1.0

        for i, (name, params) in enumerate(layer_params_dict.items()):
            weights = alpha[i]
            if weights.sum() == 0:
                weights = default_weights
            aggregated_params[name] = torch.sum(
                (weights / weights.sum()) * torch.stack(params, dim=-1), dim=-1
            )
        self.pfedla_aggregated_model_params = list(aggregated_params.values())

        return dict(
            regular_model_params={
                key: param.detach().clone() for key, param in aggregated_params.items()
            },
            personal_model_params=self.clients_personal_model_params[client_id],
        )


class HyperNetwork(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        layer_num: int,
        client_num: int,
        hidden_dim: int,
        K: int,
    ):
        super(HyperNetwork, self).__init__()
        self.K = K
        self.layer_num = layer_num
        self.client_num = client_num
        self.embedding = nn.Embedding(client_num, embedding_dim)

        def uniform_init_linear(in_features, out_features):
            linear = nn.Linear(in_features, out_features)
            torch.nn.init.uniform_(linear.weight)
            torch.nn.init.zeros_(linear.bias)
            return linear

        self.mlp = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.fc_layers = nn.ParameterList(
            uniform_init_linear(hidden_dim, client_num) for _ in range(self.layer_num)
        )

    def forward(self, client_id: int):
        emd = self.embedding(torch.tensor(client_id, dtype=torch.long))
        feature = self.mlp(emd)
        weights = [F.relu(fc(feature)) for fc in self.fc_layers]

        if self.K > 0:  # HeurpFedLA
            default_weight = torch.zeros(self.client_num, dtype=torch.float)
            default_weight[client_id] = 1.0

            self_weights = torch.zeros(len(weights))
            for i, weight in enumerate(weights):
                self_weights[i] = weight[client_id].data

            topk_weights_idx = torch.topk(self_weights, self.K, sorted=False)[1]

            for i in topk_weights_idx:
                weights[i] = (weights[i] * default_weight).detach().requires_grad_(True)

        return weights
