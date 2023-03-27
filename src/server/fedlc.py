from argparse import Namespace
from copy import deepcopy

from fedavg import FedAvgServer
from src.config.args import get_fedlc_argparser
from src.client.fedlc import FedLCClient

# NOTE: The difference between the loss function in this benchmark and the one in the paper.
# In the paper, the logit of right class is removed from the sum (the denominator).
# However, I had tried to use the same one in the paper, but the training collapsed.
# So the reproduction of FedLC is arguable and you should not fully trust it.
# If you figure out the loss funciton implementation, please open an issue and let me know.
class FedLCServer(FedAvgServer):
    def __init__(
        self,
        algo: str = "FedLC",
        args: Namespace = None,
        unique_model=False,
        default_trainer=False,
    ):
        if args is None:
            args = get_fedlc_argparser().parse_args()
        super().__init__(algo, args, unique_model, default_trainer)
        self.trainer = FedLCClient(deepcopy(self.model), self.args, self.logger)


if __name__ == "__main__":
    server = FedLCServer()
    server.run()
