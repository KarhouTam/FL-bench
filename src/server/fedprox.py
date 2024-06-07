from argparse import ArgumentParser, Namespace

from src.server.fedavg import FedAvgServer
from src.client.fedprox import FedProxClient
from src.utils.tools import NestedNamespace


class FedProxServer(FedAvgServer):

    @staticmethod
    def get_hyperparams(args_list=None) -> Namespace:
        parser = ArgumentParser()
        parser.add_argument("--mu", type=float, default=1.0)
        return parser.parse_args(args_list)

    def __init__(
        self,
        args: NestedNamespace,
        algo: str = "FedProx",
        unique_model=False,
        use_fedavg_client_cls=False,
        return_diff=False,
    ):
        super().__init__(args, algo, unique_model, use_fedavg_client_cls, return_diff)
        self.init_trainer(FedProxClient)
