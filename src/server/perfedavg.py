from argparse import ArgumentParser, Namespace
from copy import deepcopy

from fedavg import FedAvgServer
from src.client.perfedavg import PerFedAvgClient
from src.utils.tools import NestedNamespace


def get_perfedavg_args(args_list=None) -> Namespace:
    parser = ArgumentParser()
    parser.add_argument("--version", choices=["fo", "hf"], default="fo")
    parser.add_argument("--beta", type=float, default=1e-3)
    parser.add_argument("--delta", type=float, default=1e-3)
    return parser.parse_args(args_list)


class PerFedAvgServer(FedAvgServer):
    def __init__(
        self,
        args: NestedNamespace,
        algo: str = "Per-FedAvg(FO)",
        unique_model=False,
        default_trainer=False,
    ):
        algo = "Per-FedAvg(FO)" if args.perfedavg.version == "fo" else "Per-FedAvg(HF)"
        args.common.finetune_epoch = max(1, args.common.finetune_epoch)
        super().__init__(args, algo, unique_model, default_trainer)
        self.trainer = PerFedAvgClient(
            deepcopy(self.model), self.args, self.logger, self.device
        )
