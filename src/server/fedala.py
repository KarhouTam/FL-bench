from argparse import ArgumentParser, Namespace

from omegaconf import DictConfig

from src.client.fedala import FedALAClient
from src.server.fedavg import FedAvgServer


class FedALAServer(FedAvgServer):
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

    def __init__(
        self,
        args: DictConfig,
        algorithm_name: str = "FedALA",
        unique_model=False,
        use_fedavg_client_cls=False,
        return_diff=False,
    ):
        super().__init__(
            args, algorithm_name, unique_model, use_fedavg_client_cls, return_diff
        )
        self.init_trainer(FedALAClient)
