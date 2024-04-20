from argparse import ArgumentParser, Namespace

from src.server.fedavg import FedAvgServer
from src.client.knnper import kNNPerClient
from src.utils.tools import NestedNamespace


def get_knnper_args(args_list=None) -> Namespace:
    parser = ArgumentParser()
    parser.add_argument("--capacity", type=int, default=500)
    parser.add_argument("--weight", type=float, default=0.5)
    parser.add_argument("--scale", type=float, default=1)
    parser.add_argument("--k", type=int, default=5)
    return parser.parse_args(args_list)


class kNNPerServer(FedAvgServer):
    def __init__(
        self,
        args: NestedNamespace,
        algo: str = "kNN-Per",
        unique_model=False,
        use_fedavg_client_cls=False,
        return_diff=False,
    ):
        super().__init__(args, algo, unique_model, use_fedavg_client_cls, return_diff)
        self.init_trainer(kNNPerClient)
