from argparse import ArgumentParser, Namespace
from copy import deepcopy

import torch

from fedavg import FedAvgServer, get_fedavg_argparser
from src.client.fedfomo import FedFomoClient


def get_fedfomo_argparser() -> ArgumentParser:
    parser = get_fedavg_argparser()
    parser.add_argument("--M", type=int, default=5)
    parser.add_argument("--valset_ratio", type=float, default=0.2)
    return parser


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
            deepcopy(self.model), self.args, self.logger, self.device, self.client_num
        )
        self.P = torch.eye(self.client_num, device=self.device)

    def train_one_round(self):
        client_params_cache = []
        for client_id in self.selected_clients:
            selected_params = self.generate_client_params(client_id)

            (
                client_params,
                weight_vector,
                self.client_stats[client_id][self.current_epoch],
            ) = self.trainer.train(
                client_id=client_id,
                local_epoch=self.clients_local_epoch[client_id],
                received_params=selected_params,
                verbose=((self.current_epoch + 1) % self.args.verbose_gap) == 0,
            )

            client_params_cache.append(client_params)
            self.P[client_id] += weight_vector

        self.update_client_params(client_params_cache)

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
