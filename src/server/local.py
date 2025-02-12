from src.server.fedavg import FedAvgServer


class LocalServer(FedAvgServer):
    algorithm_name: str = "Local-only"
    all_model_params_personalized = True  # `True` indicates that clients have their own fullset of personalized model parameters.
    return_diff = False  # `True` indicates that clients return `diff = W_global - W_local` as parameter update; `False` for `W_local` only.

    def train_one_round(self):
        client_packages = self.trainer.train()
        for client_id, package in client_packages.items():
            self.clients_personal_model_params[client_id].update(
                package["regular_model_params"]
            )
            self.clients_personal_model_params[client_id].update(
                package["personal_model_params"]
            )
