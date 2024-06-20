from src.client.fedper import FedPerClient
from src.server.fedavg import FedAvgServer
from src.utils.tools import NestedNamespace


class FedPerServer(FedAvgServer):
    def __init__(
        self,
        args: NestedNamespace,
        algo: str = "FedPer",
        unique_model=False,
        use_fedavg_client_cls=False,
        return_diff=False,
    ):
        super().__init__(args, algo, unique_model, use_fedavg_client_cls, return_diff)
        self.init_trainer(FedPerClient)
