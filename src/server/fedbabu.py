from omegaconf import DictConfig

from src.client.fedbabu import FedBabuClient
from src.server.fedavg import FedAvgServer


class FedBabuServer(FedAvgServer):
    def __init__(
        self,
        args: DictConfig,
        algorithm_name: str = "FedBabu",
        unique_model=False,
        use_fedavg_client_cls=False,
        return_diff=False,
    ):
        # Fine-tuning is indispensable to FedBabu.
        assert (
            args.common.finetune_epoch > 0
        ), f"FedBABU needs finetuning. Now finetune_epoch = {args.common.finetune_epoch}"
        super().__init__(
            args, algorithm_name, unique_model, use_fedavg_client_cls, return_diff
        )
        self.init_trainer(FedBabuClient)
