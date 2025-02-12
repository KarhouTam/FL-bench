from argparse import ArgumentParser, Namespace

from omegaconf import DictConfig

from src.client.perfedavg import PerFedAvgClient
from src.server.fedavg import FedAvgServer


class PerFedAvgServer(FedAvgServer):
    algorithm_name: str = "Per-FedAvg"
    all_model_params_personalized = False  # `True` indicates that clients have their own fullset of personalized model parameters.
    return_diff = False  # `True` indicates that clients return `diff = W_global - W_local` as parameter update; `False` for `W_local` only.
    client_cls = PerFedAvgClient

    @staticmethod
    def get_hyperparams(args_list=None) -> Namespace:
        parser = ArgumentParser()
        parser.add_argument("--version", choices=["fo", "hf"], default="fo")
        parser.add_argument("--beta", type=float, default=1e-3)
        parser.add_argument("--delta", type=float, default=1e-3)
        return parser.parse_args(args_list)

    def __init__(self, args: DictConfig):
        self.algorithm_name += "(FO)" if args.perfedavg.version == "fo" else "(HF)"
        args.common.test.client.finetune_epoch = max(
            1, args.common.test.client.finetune_epoch
        )
        super().__init__(args)
