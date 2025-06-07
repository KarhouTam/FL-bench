from argparse import ArgumentParser, Namespace

from src.client.fedah import FedAHClient
from src.server.fedavg import FedAvgServer


class FedAHServer(FedAvgServer):
    algorithm_name = "FedAH"
    all_model_params_personalized = False  # `True` indicates that clients have their own fullset of personalized model parameters.
    return_diff = False  # `True` indicates that clients return `diff = W_global - W_local` as parameter update; `False` for `W_local` only.
    client_cls = FedAHClient

    @staticmethod
    def get_hyperparams(args_list=None) -> Namespace:
        parser = ArgumentParser()
        parser.add_argument(
            "--eta", type=float, default=1.0, help="Weight learning rate."
        )
        parser.add_argument("--plocal_epochs", type=int, default=1)
        return parser.parse_args(args_list)
