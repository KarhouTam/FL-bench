from argparse import ArgumentParser, Namespace
from collections import OrderedDict

import torch
import torch.nn as nn

from src.server.fedavg import FedAvgServer
from src.client.fedper import FedPerClient
from src.utils.tools import trainable_params, NestedNamespace


def get_pfedhn_args(args_list=None) -> Namespace:
    parser = ArgumentParser()
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
    return parser.parse_args(args_list)


class pFedHNServer(FedAvgServer):
    def __init__(
        self,
        args: NestedNamespace,
        algo: str = None,
        unique_model=False,
        use_fedavg_client_cls=True,
        return_diff=True,
    ):
        if args.mode == "parallel":
            print("pFedHN does not support paralell mode and is fallback to serial.")
            args.mode = "serial"
        algo = "pFedHN" if args.pfedhn.version == "pfedhn" else "pFedHN-PC"
        use_fedavg_client_cls = True if args.pfedhn.version == "pfedhn" else False
        super().__init__(args, algo, unique_model, use_fedavg_client_cls, return_diff)
        if self.args.pfedhn.version == "pfedhn_pc":
            self.init_trainer(FedPerClient)

        self.hypernet = HyperNetwork(self.model, self.args.pfedhn, self.client_num)
        embed_lr = (
            self.args.pfedhn.embed_lr
            if self.args.pfedhn.embed_lr is not None
            else self.args.pfedhn.hn_lr
        )
        self.hn_optimizer = torch.optim.SGD(
            [
                {
                    "params": [
                        param
                        for name, param in self.hypernet.named_parameters()
                        if "embed" not in name
                    ]
                },
                {
                    "params": [
                        param
                        for name, param in self.hypernet.named_parameters()
                        if "embed" in name
                    ],
                    "lr": embed_lr,
                },
            ],
            lr=self.args.pfedhn.hn_lr,
            momentum=self.args.pfedhn.hn_momentum,
            weight_decay=self.args.pfedhn.hn_weight_decay,
        )

    def train_one_round(self):
        selected_clients_this_round = self.selected_clients
        for client_id in selected_clients_this_round:
            self.selected_clients = [client_id]
            clients_package = self.trainer.train()

            if self.args.pfedhn.version == "pfedhn_pc":
                for name, diff in clients_package[client_id][
                    "model_params_diff"
                ].items():
                    # set diff to all-zero can stop HN's nn.Linear modules that responsible for the classifier from updating.
                    if "classifier" in name:
                        diff.zero_()

            hn_grads = torch.autograd.grad(
                outputs=self.hypernet.outputs,
                inputs=self.hypernet.parameters(),
                grad_outputs=list(
                    clients_package[client_id]["model_params_diff"].values()
                ),
                allow_unused=True,
            )

            self.hn_optimizer.zero_grad()
            for param, grad in zip(self.hypernet.parameters(), hn_grads):
                if grad is not None:
                    param.grad = grad
            torch.nn.utils.clip_grad_norm_(
                self.hypernet.parameters(), self.args.pfedhn.norm_clip
            )
            self.hn_optimizer.step()

    def get_client_model_params(self, client_id) -> OrderedDict[str, torch.Tensor]:
        return dict(
            regular_model_params=OrderedDict(
                zip(self.trainable_params_name, self.hypernet(torch.tensor(client_id)))
            ),
            personal_model_params=self.clients_personal_model_params[client_id],
        )


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
        self.outputs: list[torch.Tensor] = []

    def forward(self, client_id):
        emd = self.embedding(client_id)
        features = self.mlp(emd)

        parameters = [
            self.params_generator[name](features).reshape(self.params_shape[name])
            for name in self.params_name
        ]
        self.outputs = parameters
        return [params.detach().clone().cpu() for params in parameters]
