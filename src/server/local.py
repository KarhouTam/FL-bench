from argparse import Namespace

from fedavg import FedAvgServer


class LocalServer(FedAvgServer):
    def __init__(
        self,
        algo: str = "Local-only",
        args: Namespace = None,
        unique_model=True,
        default_trainer=True,
    ):
        super().__init__(algo, args, unique_model, default_trainer)

    def train_one_round(self):
        client_params_cache = []
        for client_id in self.selected_clients:
            client_local_params = self.generate_client_params(client_id)

            (
                client_params,
                _,
                self.client_stats[client_id][self.current_epoch],
            ) = self.trainer.train(
                client_id=client_id,
                local_epoch=self.clients_local_epoch[client_id],
                new_parameters=client_local_params,
                return_diff=False,
                verbose=((self.current_epoch + 1) % self.args.verbose_gap) == 0,
            )
            client_params_cache.append(client_params)

        self.update_client_params(client_params_cache)


if __name__ == "__main__":
    server = LocalServer()
    server.run()
