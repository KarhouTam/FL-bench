from argparse import ArgumentParser, Namespace

from src.client.lgfedavg import LGFedAvgClient
from src.server.fedavg import FedAvgServer


class LGFedAvgServer(FedAvgServer):
    algorithm_name: str = "LG-FedAvg"
    all_model_params_personalized = False  # `True` indicates that clients have their own fullset of personalized model parameters.
    return_diff = False  # `True` indicates that clients return `diff = W_global - W_local` as parameter update; `False` for `W_local` only.
    client_cls = LGFedAvgClient

    @staticmethod
    def get_hyperparams(args_list=None) -> Namespace:
        parser = ArgumentParser()
        parser.add_argument("--num_global_layers", type=int, default=1)
        return parser.parse_args(args_list)
