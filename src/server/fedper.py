from omegaconf import DictConfig

from src.client.fedper import FedPerClient
from src.server.fedavg import FedAvgServer


class FedPerServer(FedAvgServer):
    def __init__(
        self,
        args: DictConfig,
        algorithm_name: str = "FedPer",
        unique_model=False,
        use_fedavg_client_cls=False,
        return_diff=False,
    ):
        super().__init__(
            args, algorithm_name, unique_model, use_fedavg_client_cls, return_diff
        )
        self.init_trainer(FedPerClient)
