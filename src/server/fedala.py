from argparse import ArgumentParser, Namespace

from src.client.fedala import FedALAClient
from src.server.fedavg import FedAvgServer


class FedALAServer(FedAvgServer):
    algorithm_name = "FedALA"
    all_model_params_personalized = False  # `True` indicates that clients have their own fullset of personalized model parameters.
    return_diff = False  # `True` indicates that clients return `diff = W_global - W_local` as parameter update; `False` for `W_local` only.
    client_cls = FedALAClient

    @staticmethod
    def get_hyperparams(args_list=None) -> Namespace:
        parser = ArgumentParser()
        parser.add_argument(
            "--layer_idx",
            type=int,
            default=2,
            help="Control the weight range. By default, all the layers are selected.",
        )
        parser.add_argument(
            "--num_pre_loss",
            type=int,
            default=10,
            help="The number of the recorded losses to be considered to calculate the standard deviation.",
        )
        parser.add_argument(
            "--eta", type=float, default=1.0, help="Weight learning rate."
        )
        parser.add_argument(
            "--threshold",
            type=float,
            default=0.1,
            help="Train the weight until the standard deviation of the recorded losses is less than a given threshold.",
        )
        parser.add_argument(
            "--rand_percent",
            type=float,
            default=0.8,
            help="The percent of the local training data to sample.",
        )
        return parser.parse_args(args_list)
