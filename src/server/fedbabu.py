from omegaconf import DictConfig

from src.client.fedbabu import FedBabuClient
from src.server.fedavg import FedAvgServer


class FedBabuServer(FedAvgServer):
    algorithm_name = "FedBabu"
    all_model_params_personalized = False  # `True` indicates that clients have their own fullset of personalized model parameters.
    return_diff = False  # `True` indicates that clients return `diff = W_global - W_local` as parameter update; `False` for `W_local` only.
    client_cls = FedBabuClient

    def __init__(self, args: DictConfig):
        # Fine-tuning is indispensable to FedBabu.
        assert (
            args.common.test.client.finetune_epoch > 0
        ), f"FedBABU needs finetuning. Now finetune_epoch = {args.common.test.client.finetune_epoch}"
        super().__init__(args)
