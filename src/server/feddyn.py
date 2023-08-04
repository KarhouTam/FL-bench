from argparse import ArgumentParser, Namespace
from copy import deepcopy
from collections import OrderedDict

import torch

from fedavg import FedAvgServer, get_fedavg_argparser
from src.client.feddyn import FedDynClient
from src.config.utils import trainable_params, vectorize


def get_feddyn_argparser() -> ArgumentParser:
    parser = get_fedavg_argparser()
    parser.add_argument("--alpha", type=float, default=0.01)
    parser.add_argument("--max_grad_norm", type=float, default=10)
    return parser


class FedDynServer(FedAvgServer):
    def __init__(
        self,
        algo: str = "FedDyn",
        args: Namespace = None,
        unique_model=False,
        default_trainer=False,
    ):
        if args is None:
            args = get_feddyn_argparser().parse_args()
        super().__init__(algo, args, unique_model, default_trainer)
        self.trainer = FedDynClient(deepcopy(self.model), self.args, self.logger, self.device)
        param_numel = vectorize(trainable_params(self.model)).numel()
        self.nabla = [
            torch.zeros(param_numel, device=self.device) for _ in range(self.client_num)
        ]

    def train_one_round(self):
        delta_cache = []
        for client_id in self.selected_clients:
            client_local_params = self.generate_client_params(client_id)
            (
                delta,
                _,
                self.client_stats[client_id][self.current_epoch],
            ) = self.trainer.train(
                client_id=client_id,
                local_epoch=self.clients_local_epoch[client_id],
                new_parameters=client_local_params,
                nabla=self.nabla[client_id],
                return_diff=False,
                verbose=((self.current_epoch + 1) % self.args.verbose_gap) == 0,
            )

            delta_cache.append(delta)

        self.aggregate(delta_cache)

    @torch.no_grad()
    def aggregate(self, client_params_cache):
        avg_parameters = [
            torch.stack(params).mean(dim=0) for params in zip(*client_params_cache)
        ]
        params_shape = [(param.numel(), param.shape) for param in avg_parameters]
        flatten_avg_parameters = vectorize(avg_parameters)

        for i, client_params in enumerate(client_params_cache):
            self.nabla[i] += vectorize(client_params) - flatten_avg_parameters

        flatten_new_parameters = flatten_avg_parameters + torch.stack(self.nabla).mean(
            dim=0
        )

        # reshape
        new_parameters = []
        i = 0
        for numel, shape in params_shape:
            new_parameters.append(flatten_new_parameters[i : i + numel].reshape(shape))
            i += numel
        self.global_params_dict = OrderedDict(
            zip(self.trainable_params_name, new_parameters)
        )


if __name__ == '__main__':
    server = FedDynServer()
    server.run()
