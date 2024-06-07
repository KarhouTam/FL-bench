from argparse import ArgumentParser, Namespace
from copy import deepcopy

import torch

from src.server.fedavg import FedAvgServer
from src.client.apfl import APFLClient
from src.utils.tools import NestedNamespace


class APFLServer(FedAvgServer):

    @staticmethod
    def get_hyperparams(args_list=None) -> Namespace:
        parser = ArgumentParser()
        parser.add_argument("--alpha", type=float, default=0.5)
        parser.add_argument("--adaptive_alpha", type=int, default=1)
        return parser.parse_args(args_list)

    def __init__(
        self,
        args: NestedNamespace,
        algo: str = "APFL",
        unique_model=False,
        use_fedavg_client_cls=False,
        return_diff=False,
    ):
        super().__init__(args, algo, unique_model, use_fedavg_client_cls, return_diff)
        self.init_trainer(APFLClient)
        self.client_local_model_params = {
            i: deepcopy(self.model.state_dict()) for i in self.train_clients
        }
        self.clients_alpha = {
            i: torch.tensor(self.args.apfl.alpha) for i in self.train_clients
        }

    def package(self, client_id: int):
        server_package = super().package(client_id)
        server_package["alpha"] = self.clients_alpha[client_id]
        server_package["local_model_params"] = self.client_local_model_params[client_id]
        return server_package

    def train_one_round(self):
        clients_package = self.trainer.train()

        for client_id in self.selected_clients:
            self.client_local_model_params[client_id] = clients_package[client_id][
                "local_model_params"
            ]
            self.clients_alpha[client_id] = clients_package[client_id]["alpha"]
        self.aggregate(clients_package)
