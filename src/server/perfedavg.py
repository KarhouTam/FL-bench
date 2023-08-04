from argparse import ArgumentParser, Namespace
from copy import deepcopy

from fedavg import FedAvgServer, get_fedavg_argparser
from src.client.perfedavg import PerFedAvgClient


def get_perfedavg_argparser() -> ArgumentParser:
    parser = get_fedavg_argparser()
    parser.add_argument("--version", choices=["fo", "hf"], default="fo")
    parser.add_argument("--beta", type=float, default=1e-3)
    parser.add_argument("--delta", type=float, default=1e-3)
    return parser


class PerFedAvgServer(FedAvgServer):
    def __init__(
        self,
        algo: str = None,
        args: Namespace = None,
        unique_model=False,
        default_trainer=False,
    ):
        if args is None:
            args = get_perfedavg_argparser().parse_args()
        algo = "Per-FedAvg(FO)" if args.version == "fo" else "Per-FedAvg(HF)"
        args.finetune_epoch = max(1, args.finetune_epoch)
        super().__init__(algo, args, unique_model, default_trainer)
        self.trainer = PerFedAvgClient(
            deepcopy(self.model), self.args, self.logger, self.device
        )


if __name__ == "__main__":
    server = PerFedAvgServer()
    server.run()
