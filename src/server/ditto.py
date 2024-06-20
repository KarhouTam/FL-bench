from argparse import ArgumentParser, Namespace
from copy import deepcopy

from src.client.ditto import DittoClient
from src.server.fedavg import FedAvgServer
from src.utils.tools import NestedNamespace


class DittoServer(FedAvgServer):

    @staticmethod
    def get_hyperparams(args_list=None) -> Namespace:
        parser = ArgumentParser()
        parser.add_argument("--pers_epoch", type=int, default=1)
        parser.add_argument("--lamda", type=float, default=1)
        return parser.parse_args(args_list)

    def __init__(
        self,
        args: NestedNamespace,
        algo: str = "Ditto",
        unique_model=False,
        use_fedavg_client_cls=False,
        return_diff=False,
    ):
        super().__init__(args, algo, unique_model, use_fedavg_client_cls, return_diff)
        self.init_trainer(DittoClient)
        self.clients_personalized_model_params = {
            i: deepcopy(self.model.state_dict()) for i in self.train_clients
        }

    def train_one_round(self):
        clients_package = self.trainer.train()
        for client_id in self.selected_clients:
            self.clients_personalized_model_params[client_id] = clients_package[
                client_id
            ]["personalized_model_params"]
        self.aggregate(clients_package)

    def package(self, client_id: int):
        server_package = super().package(client_id)
        server_package["personalized_model_params"] = (
            self.clients_personalized_model_params[client_id]
        )
        return server_package
