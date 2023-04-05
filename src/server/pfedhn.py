from argparse import Namespace
from collections import OrderedDict
from copy import deepcopy

import torch
import torch.nn as nn

from fedavg import FedAvgServer
from src.client.fedper import FedPerClient
from src.config.args import get_pfedhn_argparser
from src.config.utils import trainable_params


args = get_pfedhn_argparser().parse_args()


class pFedHNServer(FedAvgServer):
    def __init__(
        self,
        algo: str = None,
        args: Namespace = None,
        unique_model=False,
        default_trainer=True,
    ):
        if args is None:
            args = get_pfedhn_argparser().parse_args()
        algo = "pFedHN" if args.version == "pfedhn" else "pFedHN-PC"
        default_trainer = True if args.version == "pfedhn" else False
        super().__init__(algo, args, unique_model, default_trainer)
        if args.version == "pfedhn_pc":
            self.trainer = FedPerClient(deepcopy(self.model), self.args, self.logger)

        self.hn = HyperNetwork(self.model, self.args, self.client_num_in_total).to(
            self.device
        )
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

    def train(self):
        for E in self.train_progress_bar:
            self.current_epoch = E

            if (E + 1) % self.args.verbose_gap == 0:
                self.logger.log(" " * 30, f"TRAINING EPOCH: {E + 1}", " " * 30)

            if (E + 1) % self.args.test_gap == 0:
                self.test()

            self.selected_clients = self.client_sample_stream[E]
            hn_grads_cache = []
            weight_cache = []
            for client_id in self.selected_clients:
                client_local_params = self.generate_client_params(client_id)

                delta, weight, self.client_stats[client_id][E] = self.trainer.train(
                    client_id=client_id,
                    new_parameters=client_local_params,
                    verbose=((E + 1) % self.args.verbose_gap) == 0,
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
            self.log_info()

    def generate_client_params(self, client_id) -> OrderedDict[str, torch.Tensor]:
        return self.hn(torch.tensor(client_id, device=self.device))

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

        self.params_name, parameters = trainable_params(backbone, requires_name=True)
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

        parameters = OrderedDict(
            [
                [
                    name,
                    self.params_generator[name](features).reshape(
                        self.params_shape[name]
                    ),
                ]
                for name in self.params_name
            ]
        )
        return parameters


if __name__ == "__main__":
    server = pFedHNServer()
    server.run()
