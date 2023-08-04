from argparse import ArgumentParser, Namespace
from copy import deepcopy

from fedavg import FedAvgServer, get_fedavg_argparser
from src.client.apfl import APFLClient


def get_apfl_argparser() -> ArgumentParser:
    parser = get_fedavg_argparser()
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--adaptive_alpha", type=int, default=1)
    return parser


class APFLServer(FedAvgServer):
    def __init__(
        self,
        algo: str = "APFL",
        args: Namespace = None,
        unique_model=False,
        default_trainer=False,
    ):
        if args is None:
            args = get_apfl_argparser().parse_args()
        super().__init__(algo, args, unique_model, default_trainer)
        self.trainer = APFLClient(
            deepcopy(self.model), self.args, self.logger, self.device, self.client_num
        )


if __name__ == "__main__":
    server = APFLServer()
    server.run()
