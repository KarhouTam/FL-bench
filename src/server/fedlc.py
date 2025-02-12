from argparse import ArgumentParser, Namespace

from src.client.fedlc import FedLCClient
from src.server.fedavg import FedAvgServer

"""
NOTE: The difference between the loss function in this benchmark and the one in the paper.
In the paper, the logit of right class is removed from the sum (the denominator).
However, I had tried to use the same one in the paper, but the training collapsed.
So the reproduction of FedLC is arguable and you should not fully trust it.
If you figure out the loss funciton implementation, please open an issue and let me know.
More discussions about FedLC: https://github.com/KarhouTam/FL-bench/issues/5
"""


class FedLCServer(FedAvgServer):
    algorithm_name: str = "FedLC"
    all_model_params_personalized = False  # `True` indicates that clients have their own fullset of personalized model parameters.
    return_diff = False  # `True` indicates that clients return `diff = W_global - W_local` as parameter update; `False` for `W_local` only.
    client_cls = FedLCClient

    @staticmethod
    def get_hyperparams(args_list=None) -> Namespace:
        parser = ArgumentParser()
        parser.add_argument("--tau", type=float, default=1.0)
        return parser.parse_args(args_list)
