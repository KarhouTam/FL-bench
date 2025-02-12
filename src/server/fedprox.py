from argparse import ArgumentParser, Namespace

from src.client.fedprox import FedProxClient
from src.server.fedavg import FedAvgServer


class FedProxServer(FedAvgServer):
    algorithm_name: str = "FedProx"
    all_model_params_personalized = False  # `True` indicates that clients have their own fullset of personalized model parameters.
    return_diff = False  # `True` indicates that clients return `diff = W_global - W_local` as parameter update; `False` for `W_local` only.
    client_cls = FedProxClient

    @staticmethod
    def get_hyperparams(args_list=None) -> Namespace:
        parser = ArgumentParser()
        parser.add_argument("--mu", type=float, default=1.0)
        return parser.parse_args(args_list)
