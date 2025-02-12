from argparse import ArgumentParser, Namespace
from copy import deepcopy

from omegaconf import DictConfig

from src.client.ditto import DittoClient
from src.server.fedavg import FedAvgServer


class DittoServer(FedAvgServer):
    algorithm_name: str = "Ditto"
    all_model_params_personalized = False  # `True` indicates that clients have their own fullset of personalized model parameters.
    return_diff = False  # `True` indicates that clients return `diff = W_global - W_local` as parameter update; `False` for `W_local` only.
    client_cls = DittoClient

    @staticmethod
    def get_hyperparams(args_list=None) -> Namespace:
        parser = ArgumentParser()
        parser.add_argument("--pers_epoch", type=int, default=1)
        parser.add_argument("--lamda", type=float, default=1)
        return parser.parse_args(args_list)

    def __init__(self, args: DictConfig):
        super().__init__(args)
        self.clients_personalized_model_params = {
            i: deepcopy(self.model.state_dict()) for i in self.train_clients
        }

    def train_one_round(self):
        client_packages = self.trainer.train()
        for client_id in self.selected_clients:
            self.clients_personalized_model_params[client_id] = client_packages[
                client_id
            ]["personalized_model_params"]
        self.aggregate_client_updates(client_packages)

    def package(self, client_id: int):
        server_package = super().package(client_id)
        server_package["personalized_model_params"] = (
            self.clients_personalized_model_params[client_id]
        )
        return server_package
