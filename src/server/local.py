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

    def train(self):
        for E in self.train_progress_bar:
            self.current_epoch = E

            if (E + 1) % self.args.verbose_gap == 0:
                self.logger.log(" " * 30, f"TRAINING EPOCH: {E + 1}", " " * 30)

            if (E + 1) % self.args.test_gap == 0:
                self.test()

            self.selected_clients = self.client_sample_stream[E]

            client_params_cache = []
            for client_id in self.selected_clients:

                client_local_params = self.generate_client_params(client_id)

                (
                    client_params,
                    _,
                    self.client_stats[client_id][E],
                ) = self.trainer.train(
                    client_id=client_id,
                    new_parameters=client_local_params,
                    return_diff=False,
                    verbose=((E + 1) % self.args.verbose_gap) == 0,
                )

                client_params_cache.append(client_params)

            self.update_client_params(client_params_cache)
            self.log_info()


if __name__ == "__main__":
    server = LocalServer()
    server.run()
