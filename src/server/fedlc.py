from argparse import ArgumentParser, Namespace
from copy import deepcopy

from fedavg import FedAvgServer
from src.client.fedlc import FedLCClient
from src.utils.tools import NestedNamespace


def get_fedlc_args(args_list=None) -> Namespace:
    parser = ArgumentParser()
    parser.add_argument("--tau", type=float, default=1.0)
    return parser.parse_args(args_list)


# NOTE: The difference between the loss function in this benchmark and the one in the paper.
# In the paper, the logit of right class is removed from the sum (the denominator).
# However, I had tried to use the same one in the paper, but the training collapsed.
# So the reproduction of FedLC is arguable and you should not fully trust it.
# If you figure out the loss funciton implementation, please open an issue and let me know.
# More discussions about FedLC: https://github.com/KarhouTam/FL-bench/issues/5
class FedLCServer(FedAvgServer):
    def __init__(
        self,
        args: NestedNamespace,
        algo: str = "FedLC",
        unique_model=False,
        default_trainer=False,
    ):
        super().__init__(args, algo, unique_model, default_trainer)
        self.trainer = FedLCClient(
            deepcopy(self.model), self.args, self.logger, self.device
        )
