from argparse import ArgumentParser, Namespace

from omegaconf import DictConfig

from src.client.fedprox import FedProxClient
from src.server.fedavg import FedAvgServer


class FedProxServer(FedAvgServer):
    @staticmethod
    def get_hyperparams(args_list=None) -> Namespace:
        parser = ArgumentParser()
        parser.add_argument("--mu", type=float, default=1.0)
        return parser.parse_args(args_list)

    def __init__(
        self,
        args: DictConfig,
        algorithm_name: str = "FedProx",
        unique_model=False,
        use_fedavg_client_cls=False,
        return_diff=False,
    ):
        super().__init__(
            args, algorithm_name, unique_model, use_fedavg_client_cls, return_diff
        )
        self.init_trainer(FedProxClient)
