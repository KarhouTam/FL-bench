from argparse import ArgumentParser, Namespace

from omegaconf import DictConfig

from src.client.moon import MOONClient
from src.server.fedavg import FedAvgServer


class MOONServer(FedAvgServer):
    @staticmethod
    def get_hyperparams(args_list=None) -> Namespace:
        parser = ArgumentParser()
        parser.add_argument("--tau", type=float, default=0.5)
        parser.add_argument("--mu", type=float, default=5)
        return parser.parse_args(args_list)

    def __init__(
        self,
        args: DictConfig,
        algorithm_name: str = "MOON",
        unique_model=False,
        use_fedavg_client_cls=False,
        return_diff=False,
    ):
        super().__init__(
            args, algorithm_name, unique_model, use_fedavg_client_cls, return_diff
        )
        self.init_trainer(MOONClient)
        self.clients_prev_model_params = {i: {} for i in self.train_clients}

    def package(self, client_id: int):
        server_package = super().package(client_id)
        server_package["prev_model_params"] = self.clients_prev_model_params[client_id]
        return server_package

    def train_one_round(self):
        client_packages = self.trainer.train()
        for client_id, package in zip(self.selected_clients, client_packages.values()):
            self.clients_prev_model_params[client_id].update(
                package["regular_model_params"]
            )
            self.clients_prev_model_params[client_id].update(
                package["personal_model_params"]
            )
        self.aggregate(client_packages)
