from argparse import ArgumentParser, Namespace
from typing import Any, Dict

from omegaconf import DictConfig

from src.client.fedas import FedASClient
from src.server.fedavg import FedAvgServer


class FedASServer(FedAvgServer):
    @staticmethod
    def get_hyperparams(args_list=None) -> Namespace:
        parser = ArgumentParser()
        parser.add_argument("--alignment_lr", type=float, default=0.01)
        parser.add_argument("--alignment_epoch", type=int, default=1)
        return parser.parse_args(args_list)

    def __init__(
        self,
        args: DictConfig,
        algorithm_name: str = "FedAS",
        unique_model=False,
        use_fedavg_client_cls=False,
        return_diff=False,
    ):
        super().__init__(
            args, algorithm_name, unique_model, use_fedavg_client_cls, return_diff
        )
        self.client_prev_model_states: Dict[int, Dict[str, Any]] = {}
        self.init_trainer(FedASClient)

    def train_one_round(self):
        """The function of indicating specific things FL method need to do (at
        server side) in each communication round."""

        client_packages = self.trainer.train()
        for client_id, package in client_packages.items():
            self.client_prev_model_states[client_id] = package["prev_model_state"]
        self.aggregate(client_packages)

    def package(self, client_id: int):
        server_package = super().package(client_id)
        if client_id in self.client_prev_model_states:
            server_package["prev_model_state"] = self.client_prev_model_states[
                client_id
            ]
        else:
            server_package["prev_model_state"] = None
        return server_package
