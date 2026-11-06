from argparse import ArgumentParser, Namespace
from src.client.fedpaq import FedPAQClient
from src.server.fedavg import FedAvgServer


class FedPAQServer(FedAvgServer):
    """Server implementation for FedPAQ (AISTATS'20).

    Reference:
        Amirhossein Reisizadeh, Aryan Mokhtari, Hamed Hassani, Ali Jadbabaie, and Ramtin Pedarsani,
        "FedPAQ: A Communication-Efficient Federated Learning Method with Periodic Averaging and Quantization,"
        Proceedings of the 23rd International Conference on Artificial Intelligence and Statistics (AISTATS), 2020.
        http://proceedings.mlr.press/v108/reisizadeh20a.html
    """

    algorithm_name: str = "FedPAQ"
    all_model_params_personalized: bool = False
    return_diff: bool = True
    client_cls = FedPAQClient

    @staticmethod
    def get_hyperparams(args_list=None) -> Namespace:
        parser = ArgumentParser()
        parser.add_argument(
            "--num_bits",
            type=int,
            default=4,
            help="Number of quantization bits for client uplink updates",
        )
        parser.add_argument(
            "--stochastic",
            type=int,
            default=0,
            choices=[0, 1],
            help="1 for stochastic rounding, 0 for deterministic rounding",
        )
        return parser.parse_args(args_list)
