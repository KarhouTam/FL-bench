from argparse import ArgumentParser, Namespace
from collections import OrderedDict
from typing import Any

import torch
from omegaconf import DictConfig

from src.client.pefll import EmbedNetwork, HyperNetwork, PeFLLClient
from src.server.fedavg import FedAvgServer


# about other hyperparameter settings, most you can find in the common section.
class PeFLLServer(FedAvgServer):
    algorithm_name: str = "PeFLL"
    all_model_params_personalized = True  # `True` indicates that clients have their own fullset of personalized model parameters.
    return_diff = False  # `True` indicates that clients return `diff = W_global - W_local` as parameter update; `False` for `W_local` only.
    client_cls = PeFLLClient

    @staticmethod
    def get_hyperparams(args_list=None) -> Namespace:
        parser = ArgumentParser()
        parser.add_argument("--embed_dim", type=int, default=-1)
        parser.add_argument("--embed_y", type=int, default=1)
        parser.add_argument("--embed_num_kernels", type=int, default=16)
        parser.add_argument("--embed_num_batches", type=int, default=1)
        parser.add_argument("--hyper_embed_lr", type=float, default=2e-4)
        parser.add_argument("--hyper_hidden_dim", type=int, default=100)
        parser.add_argument("--hyper_num_hidden_layers", type=int, default=3)
        parser.add_argument("--clip_norm", type=float, default=50.0)
        return parser.parse_args(args_list)

    def __init__(self, args: DictConfig):
        if args.common.buffers == "global":
            raise NotImplementedError("PeFLL doesn't support global buffers.")
        super().__init__(args, False)
        if self.args.dataset.split == "user":
            raise NotImplementedError(
                "PeFLL is not available with user-based data partition"
                "(i.e., users are divided into train users and test users, the latter has no data for training at all.)."
            )
        if self.args.pefll.embed_dim <= 0:
            self.args.pefll.embed_dim = int(1 + self.client_num / 4)
        self.embed_net = EmbedNetwork(self.args)
        self.hyper_net = HyperNetwork(self.model, self.args)
        self.embed_hyper_optimizer = torch.optim.Adam(
            list(self.embed_net.parameters()) + list(self.hyper_net.parameters()),
            lr=self.args.pefll.hyper_embed_lr,
        )
        self.init_trainer(embed_net=self.embed_net, hyper_net=self.hyper_net)

    def package(self, client_id: int):
        server_package = super().package(client_id)
        # in FL-bench, I choose to send both embed net and hyper net to clients for easier reproduction
        # this won't do anything out of the algorithm
        # but make PeFLL available for client parallel training
        server_package["embed_net_params"] = self.embed_net.state_dict()
        server_package["hyper_net_params"] = self.hyper_net.state_dict()
        return server_package

    def aggregate_client_updates(
        self, client_packages: OrderedDict[int, dict[str, Any]]
    ):
        all_embed_net_grads = [
            package["embed_net_grads"] for package in client_packages.values()
        ]
        all_hyper_net_grads = [
            package["hyper_net_grads"] for package in client_packages.values()
        ]
        self.embed_hyper_optimizer.zero_grad()
        for param, grads in zip(self.embed_net.parameters(), zip(*all_embed_net_grads)):
            param.grad = torch.stack(grads, dim=0).mean(dim=0)
        for param, grads in zip(self.hyper_net.parameters(), zip(*all_hyper_net_grads)):
            param.grad = torch.stack(grads, dim=0).mean(dim=0)
        self.embed_hyper_optimizer.step()

    # in FL-bench's workflow, is better to let clients generate model parameters by themselves
    def get_client_model_params(self, client_id: int) -> OrderedDict[str, torch.Tensor]:
        return dict(
            regular_model_params={},
            personal_model_params=self.clients_personal_model_params[client_id],
        )
