from argparse import ArgumentParser, Namespace
from copy import deepcopy
from collections import OrderedDict

import torch

from fedavg import FedAvgServer
from src.client.feddyn import FedDynClient
from src.utils.tools import trainable_params, NestedNamespace, vectorize
from src.utils.tools import NestedNamespace


def get_feddyn_args(args_list=None) -> Namespace:
    parser = ArgumentParser()
    parser.add_argument("--alpha", type=float, default=0.01)
    parser.add_argument("--max_grad_norm", type=float, default=10)
    return parser.parse_args(args_list)


class FedDynServer(FedAvgServer):
    def __init__(
        self,
        args: NestedNamespace,
        algo: str = "FedDyn",
        unique_model=False,
        default_trainer=False,
    ):
        super().__init__(args, algo, unique_model, default_trainer)
        self.trainer = FedDynClient(
            deepcopy(self.model), self.args, self.logger, self.device
        )
        param_numel = vectorize(trainable_params(self.model)).numel()
        self.nabla = [
            torch.zeros(param_numel, device=self.device) for _ in range(self.client_num)
        ]
        self.weight_list = torch.tensor(
            [len(self.trainer.data_indices[i]["train"]) for i in self.train_clients],
            device=self.device,
        )
        self.weight_list = (
            self.weight_list / self.weight_list.sum() * len(self.train_clients)
        )

    def train_one_round(self):
        delta_cache = []
        for client_id in self.selected_clients:
            client_local_params = self.generate_client_params(client_id)
            (delta, _, self.client_metrics[client_id][self.current_epoch]) = (
                self.trainer.train(
                    client_id=client_id,
                    local_epoch=self.clients_local_epoch[client_id],
                    new_parameters=client_local_params,
                    nabla=self.nabla[client_id],
                    alpha=self.args.feddyn.alpha / self.weight_list[client_id],
                    return_diff=False,
                    verbose=((self.current_epoch + 1) % self.args.common.verbose_gap)
                    == 0,
                )
            )

            delta_cache.append(delta)

        self.aggregate(delta_cache)

    @torch.no_grad()
    def aggregate(self, client_params_cache):
        avg_parameters = [
            torch.stack(params).mean(dim=0) for params in zip(*client_params_cache)
        ]
        params_shape = [(param.numel(), param.shape) for param in avg_parameters]
        flatten_global_params = vectorize(self.global_params_dict)

        for i, client_params in enumerate(client_params_cache):
            self.nabla[i] += vectorize(client_params) - flatten_global_params

        flatten_new_params = vectorize(avg_parameters) + torch.stack(self.nabla).mean(
            dim=0
        )

        # reshape
        new_parameters = []
        i = 0
        for numel, shape in params_shape:
            new_parameters.append(flatten_new_params[i : i + numel].reshape(shape))
            i += numel
        self.global_params_dict = OrderedDict(
            zip(self.trainable_params_name, new_parameters)
        )
