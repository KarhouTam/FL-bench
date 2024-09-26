from argparse import ArgumentParser, Namespace
from collections import OrderedDict
from typing import Any

import torch
from omegaconf import DictConfig

from src.client.pefll import EmbedNetwork, HyperNetwork, PeFLLClient
from src.server.fedavg import FedAvgServer


# about other hyperparameter settings, most you can find in the common section.
class PeFLLServer(FedAvgServer):
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

    def __init__(
        self,
        args: DictConfig,
        algorithm_name: str = "PeFLL",
        unique_model=True,
        use_fedavg_client_cls=False,
        return_diff=False,
    ):
        if args.common.buffers == "global":
            raise NotImplementedError("PeFLL doesn't support global buffers.")
        super().__init__(
            args, algorithm_name, unique_model, use_fedavg_client_cls, return_diff
        )
        if self.args.pefll.embed_dim <= 0:
            self.args.pefll.embed_dim = int(1 + self.client_num / 4)
        self.embed_net = EmbedNetwork(self.args)
        self.hyper_net = HyperNetwork(self.model, self.args)
        self.embed_hyper_optimizer = torch.optim.Adam(
            list(self.embed_net.parameters()) + list(self.hyper_net.parameters()),
            lr=self.args.pefll.hyper_embed_lr,
        )
        self.init_trainer(
            PeFLLClient, embed_net=self.embed_net, hyper_net=self.hyper_net
        )

    def package(self, client_id: int):
        server_package = super().package(client_id)
        # in FL-bench, I choose to send both embed net and hyper net to clients for easier reproduction
        # this won't do anything out of the algorithm
        # but make PeFLL available for client parallel training
        server_package["embed_net_params"] = self.embed_net.state_dict()
        server_package["hyper_net_params"] = self.hyper_net.state_dict()
        return server_package

    def aggregate(self, client_packages: OrderedDict[int, dict[str, Any]]):
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
