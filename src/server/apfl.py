from argparse import ArgumentParser, Namespace
from copy import deepcopy

import torch
from omegaconf import DictConfig

from src.client.apfl import APFLClient
from src.server.fedavg import FedAvgServer


class APFLServer(FedAvgServer):
    algorithm_name: str = "APFL"
    all_model_params_personalized = False  # `True` indicates that clients have their own fullset of personalized model parameters.
    return_diff = False  # `True` indicates that clients return `diff = W_global - W_local` as parameter update; `False` for `W_local` only.
    client_cls = APFLClient

    @staticmethod
    def get_hyperparams(args_list=None) -> Namespace:
        parser = ArgumentParser()
        parser.add_argument("--alpha", type=float, default=0.5)
        parser.add_argument("--adaptive_alpha", type=int, default=1)
        return parser.parse_args(args_list)

    def __init__(self, args: DictConfig):
        super().__init__(args)
        self.client_local_model_params = {
            i: deepcopy(self.model.state_dict()) for i in self.train_clients
        }
        self.client_alphas = {
            i: torch.tensor(self.args.apfl.alpha) for i in self.train_clients
        }

    def package(self, client_id: int):
        server_package = super().package(client_id)
        server_package["alpha"] = self.client_alphas[client_id]
        server_package["local_model_params"] = self.client_local_model_params[client_id]
        return server_package

    def train_one_round(self):
        client_packages = self.trainer.train()

        for client_id in self.selected_clients:
            self.client_local_model_params[client_id] = client_packages[client_id][
                "local_model_params"
            ]
            self.client_alphas[client_id] = client_packages[client_id]["alpha"]
        self.aggregate_client_updates(client_packages)
