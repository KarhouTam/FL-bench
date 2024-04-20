from src.server.fedavg import FedAvgServer
from src.client.fedbn import FedBNClient
from src.utils.tools import NestedNamespace


class FedBNServer(FedAvgServer):
    def __init__(
        self,
        args: NestedNamespace,
        algo: str = "FedBN",
        unique_model=False,
        use_fedavg_client_cls=False,
        return_diff=False,
    ):
        super().__init__(args, algo, unique_model, use_fedavg_client_cls, return_diff)
        self.init_trainer(FedBNClient)
