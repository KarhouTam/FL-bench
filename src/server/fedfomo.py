from argparse import ArgumentParser, Namespace

import torch
from omegaconf import DictConfig

from src.client.fedfomo import FedFomoClient
from src.server.fedavg import FedAvgServer


class FedFomoServer(FedAvgServer):
    algorithm_name: str = "FedFomo"
    all_model_params_personalized = True  # `True` indicates that clients have their own fullset of personalized model parameters.
    return_diff = False  # `True` indicates that clients return `diff = W_global - W_local` as parameter update; `False` for `W_local` only.
    client_cls = FedFomoClient

    @staticmethod
    def get_hyperparams(args_list=None) -> Namespace:
        parser = ArgumentParser()
        parser.add_argument("--M", type=int, default=5)
        parser.add_argument("--valset_ratio", type=float, default=0.2)
        return parser.parse_args(args_list)

    def __init__(self, args: DictConfig):
        super().__init__(args)
        self.P = torch.eye(self.client_num, device=self.device)

    def train_one_round(self):
        client_packages = self.trainer.train()
        for client_id, package in client_packages.items():
            for i, val in package["client_weights"].items():
                self.P[client_id][i] += val

    @torch.no_grad()
    def get_client_model_params(self, client_id):
        prev_round_clients = self.client_sample_stream[self.current_epoch - 1]
        selected_params = {}
        if not self.testing and self.current_epoch > 0:
            similar_clients = [
                prev_round_clients[i]
                for i in torch.topk(
                    self.P[client_id][prev_round_clients], self.args.fedfomo.M
                ).indices.tolist()
            ]
            for i in similar_clients:
                selected_params[i] = {
                    key: self.clients_personal_model_params[i][key]
                    for key in self.public_model_param_names
                }
        selected_params[client_id] = self.clients_personal_model_params[client_id]
        return dict(model_params_from_selected_clients=selected_params)
