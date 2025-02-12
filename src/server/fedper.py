from src.client.fedper import FedPerClient
from src.server.fedavg import FedAvgServer


class FedPerServer(FedAvgServer):
    algorithm_name: str = "FedPer"
    all_model_params_personalized = False  # `True` indicates that clients have their own fullset of personalized model parameters.
    return_diff = False  # `True` indicates that clients return `diff = W_global - W_local` as parameter update; `False` for `W_local` only.
    client_cls = FedPerClient
