from omegaconf import DictConfig

from src.server.fedavg import FedAvgServer


class LocalServer(FedAvgServer):
    def __init__(
        self,
        args: DictConfig,
        algorithm_name: str = "Local-only",
        unique_model=True,
        use_fedavg_client_cls=True,
        return_diff=False,
    ):
        super().__init__(
            args, algorithm_name, unique_model, use_fedavg_client_cls, return_diff
        )

    def train_one_round(self):
        client_packages = self.trainer.train()
        for client_id, package in client_packages.items():
            self.clients_personal_model_params[client_id].update(
                package["regular_model_params"]
            )
            self.clients_personal_model_params[client_id].update(
                package["personal_model_params"]
            )
