from argparse import ArgumentParser, Namespace

from src.client.knnper import kNNPerClient
from src.server.fedavg import FedAvgServer


class kNNPerServer(FedAvgServer):
    algorithm_name: str = "kNN-Per"
    all_model_params_personalized = False  # `True` indicates that clients have their own fullset of personalized model parameters.
    return_diff = False  # `True` indicates that clients return `diff = W_global - W_local` as parameter update; `False` for `W_local` only.
    client_cls = kNNPerClient

    @staticmethod
    def get_hyperparams(args_list=None) -> Namespace:
        parser = ArgumentParser()
        parser.add_argument("--capacity", type=int, default=500)
        parser.add_argument("--weight", type=float, default=0.5)
        parser.add_argument("--scale", type=float, default=1)
        parser.add_argument("--k", type=int, default=5)
        return parser.parse_args(args_list)
