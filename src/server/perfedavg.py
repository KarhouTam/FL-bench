from argparse import ArgumentParser, Namespace

from src.server.fedavg import FedAvgServer
from src.client.perfedavg import PerFedAvgClient
from src.utils.tools import NestedNamespace


class PerFedAvgServer(FedAvgServer):

    @staticmethod
    def get_hyperparams(args_list=None) -> Namespace:
        parser = ArgumentParser()
        parser.add_argument("--version", choices=["fo", "hf"], default="fo")
        parser.add_argument("--beta", type=float, default=1e-3)
        parser.add_argument("--delta", type=float, default=1e-3)
        return parser.parse_args(args_list)

    def __init__(
        self,
        args: NestedNamespace,
        algo: str = "Per-FedAvg(FO)",
        unique_model=False,
        use_fedavg_client_cls=False,
        return_diff=False,
    ):
        algo = "Per-FedAvg(FO)" if args.perfedavg.version == "fo" else "Per-FedAvg(HF)"
        args.common.finetune_epoch = max(1, args.common.finetune_epoch)
        super().__init__(args, algo, unique_model, use_fedavg_client_cls, return_diff)
        self.init_trainer(PerFedAvgClient)
