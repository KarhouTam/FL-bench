from argparse import Namespace
from copy import deepcopy

import torch

from fedavg import FedAvgServer
from src.config.args import get_fedfomo_argparser
from src.client.fedfomo import FedFomoClient


class FedFomoServer(FedAvgServer):
    def __init__(
        self,
        algo: str = "FedFomo",
        args: Namespace = None,
        unique_model=True,
        default_trainer=False,
    ):
        if args is None:
            args = get_fedfomo_argparser().parse_args()
        super().__init__(algo, args, unique_model, default_trainer)
        self.trainer = FedFomoClient(
            deepcopy(self.model), self.args, self.logger, self.client_num_in_total
        )
        self.P = torch.eye(self.client_num_in_total, device=self.device)
        self.test_flag = False

    def train(self):
        for E in self.train_progress_bar:
            self.current_epoch = E

            if (E + 1) % self.args.verbose_gap == 0:
                self.logger.log(" " * 30, f"TRAINING EPOCH: {E + 1}", " " * 30)

            if (E + 1) % self.args.test_gap == 0:
                self.test_flag = True
                self.test()
                self.test_flag = False

            self.selected_clients = self.client_sample_stream[E]
            client_params_cache = []
            for client_id in self.selected_clients:

                selected_params = self.generate_client_params(client_id)

                (
                    client_params,
                    weight_vector,
                    self.client_stats[client_id][E],
                ) = self.trainer.train(
                    client_id=client_id,
                    received_params=selected_params,
                    verbose=((E + 1) % self.args.verbose_gap) == 0,
                )

                client_params_cache.append(client_params)
                self.P[client_id] += weight_vector.to(self.device)

            self.update_client_params(client_params_cache)
            self.log_info()

    @torch.no_grad()
    def generate_client_params(self, client_id):
        prev_round_clients = self.client_sample_stream[self.current_epoch - 1]
        selected_params = {}
        if not self.test_flag and self.current_epoch > 0:
            selected_params = {
                prev_round_clients[i]: self.client_trainable_params[
                    prev_round_clients[i]
                ]
                for i in torch.topk(
                    self.P[client_id][prev_round_clients], self.args.M
                ).indices.tolist()
            }
        selected_params[client_id] = self.client_trainable_params[client_id]
        return selected_params


if __name__ == "__main__":
    server = FedFomoServer()
    server.run()
