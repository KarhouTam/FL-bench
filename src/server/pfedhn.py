from argparse import ArgumentParser, Namespace
from collections import OrderedDict
from copy import deepcopy

import torch
import torch.nn as nn

from fedavg import FedAvgServer, get_fedavg_argparser
from src.client.fedper import FedPerClient
from src.config.utils import trainable_params


def get_pfedhn_argparser() -> ArgumentParser:
    parser = get_fedavg_argparser()
    parser.add_argument(
        "--version", type=str, choices=["pfedhn", "pfedhn_pc"], default="pfedhn"
    )
    parser.add_argument("--embed_dim", type=int, default=-1)
    parser.add_argument("--hn_lr", type=float, default=1e-2)
    parser.add_argument("--embed_lr", type=float, default=None)
    parser.add_argument("--hn_momentum", type=float, default=0.9)
    parser.add_argument("--hn_weight_decay", type=float, default=1e-3)
    parser.add_argument("--hidden_dim", type=int, default=100)
    parser.add_argument("--hidden_num", type=int, default=3)
    parser.add_argument("--norm_clip", type=int, default=50)
    return parser


class pFedHNServer(FedAvgServer):
    def __init__(
        self,
        algo: str = None,
        args: Namespace = None,
        unique_model=True,
        default_trainer=True,
    ):
        if args is None:
            args = get_pfedhn_argparser().parse_args()
        algo = "pFedHN" if args.version == "pfedhn" else "pFedHN-PC"
        default_trainer = True if args.version == "pfedhn" else False
        super().__init__(algo, args, unique_model, default_trainer)
        if args.version == "pfedhn_pc":
            self.trainer = FedPerClient(
                deepcopy(self.model), self.args, self.logger, self.device
            )

        self.hn = HyperNetwork(self.model, self.args, self.client_num).to(self.device)
        embed_lr = (
            self.args.embed_lr if self.args.embed_lr is not None else self.args.hn_lr
        )
        self.hn_optimizer = torch.optim.SGD(
            [
                {
                    "params": [
                        param
                        for name, param in self.hn.named_parameters()
                        if "embed" not in name
                    ]
                },
                {
                    "params": [
                        param
                        for name, param in self.hn.named_parameters()
                        if "embed" in name
                    ],
                    "lr": embed_lr,
                },
            ],
            lr=self.args.hn_lr,
            momentum=self.args.hn_momentum,
            weight_decay=self.args.hn_weight_decay,
        )

    def train_one_round(self):
        hn_grads_cache = []
        weight_cache = []
        for client_id in self.selected_clients:
            client_local_params = self.generate_client_params(client_id)
            (
                delta,
                weight,
                self.client_stats[client_id][self.current_epoch],
            ) = self.trainer.train(
                client_id=client_id,
                local_epoch=self.clients_local_epoch[client_id],
                new_parameters=client_local_params,
                verbose=((self.current_epoch + 1) % self.args.verbose_gap) == 0,
            )
            if self.args.version == "pfedhn_pc":
                for name, diff in delta.items():
                    # set diff to all-zero can stop HN's nn.Linear modules that responsible for the classifier from updating.
                    if "classifier" in name:
                        diff.zero_()
            weight_cache.append(weight)
            hn_grads_cache.append(
                torch.autograd.grad(
                    outputs=list(client_local_params.values()),
                    inputs=self.hn.parameters(),
                    grad_outputs=list(delta.values()),
                    allow_unused=True,
                )
            )

        self.update_hn(weight_cache, hn_grads_cache)

    def generate_client_params(self, client_id) -> OrderedDict[str, torch.Tensor]:
        if not self.test_flag:
            self.client_trainable_params[client_id] = self.hn(
                torch.tensor(client_id, device=self.device)
            )
        return OrderedDict(
            zip(self.trainable_params_name, self.client_trainable_params[client_id])
        )

    def update_hn(self, weight_cache, hn_grads_cache):
        weights = torch.tensor(weight_cache, device=self.device) / sum(weight_cache)
        hn_grads = []
        for grads in zip(*hn_grads_cache):
            hn_grads.append(torch.sum(torch.stack(grads, dim=-1) * weights, dim=-1))

        self.hn_optimizer.zero_grad()
        for param, grad in zip(self.hn.parameters(), hn_grads):
            if grad is not None:
                param.grad = grad
        torch.nn.utils.clip_grad_norm_(self.hn.parameters(), self.args.norm_clip)
        self.hn_optimizer.step()


class HyperNetwork(nn.Module):
    def __init__(self, backbone: nn.Module, args: Namespace, client_num: int):
        super().__init__()
        embed_dim = args.embed_dim if args.embed_dim > 0 else int((1 + client_num) / 4)
        self.embedding = nn.Embedding(
            num_embeddings=client_num, embedding_dim=embed_dim
        )

        mlp_layers = [nn.Linear(embed_dim, args.hidden_dim)]
        for _ in range(args.hidden_num):
            mlp_layers.append(nn.ReLU(True))
            mlp_layers.append(nn.Linear(args.hidden_dim, args.hidden_dim))
        self.mlp = nn.Sequential(*mlp_layers)

        parameters, self.params_name = trainable_params(backbone, requires_name=True)
        self.params_shape = {
            name: backbone.state_dict()[name].shape for name in self.params_name
        }
        self.params_generator = nn.ParameterDict()
        for name, param in zip(self.params_name, parameters):
            self.params_generator[name] = nn.Linear(
                args.hidden_dim, len(param.flatten())
            )

    def forward(self, client_id):
        emd = self.embedding(client_id)
        features = self.mlp(emd)

        parameters = [
            self.params_generator[name](features).reshape(self.params_shape[name])
            for name in self.params_name
        ]

        return parameters


if __name__ == "__main__":
    server = pFedHNServer()
    server.run()
