from argparse import ArgumentParser, Namespace
from collections import OrderedDict
from copy import deepcopy

import torch
from rich.progress import track

from fedavg import FedAvgServer
from src.utils.tools import trainable_params, NestedNamespace


def get_pfedsim_args(args_list=None) -> Namespace:
    parser = ArgumentParser()
    parser.add_argument("-wr", "--warmup_round", type=float, default=0.5)
    return parser.parse_args(args_list)


class pFedSimServer(FedAvgServer):
    def __init__(self, args: NestedNamespace, algo: str = "pFedSim"):
        super().__init__(args, algo)
        self.weight_matrix = torch.eye(self.client_num, device=self.device)

        self.warmup_round = 0
        if 0 <= self.args.pfedsim.warmup_round <= 1:
            self.warmup_round = int(
                self.args.common.global_epoch * self.args.pfedsim.warmup_round
            )
        elif 1 < self.args.pfedsim.warmup_round < self.args.common.global_epoch:
            self.warmup_round = int(self.args.pfedsim.warmup_round)
        else:
            raise RuntimeError(
                "warmup_round need to be set in the range of [0, 1) or [1, global_epoch)."
            )

    def train(self):
        # Warm-up Phase
        self.train_progress_bar = track(
            range(self.warmup_round),
            f"[bold cyan]Warming-up...",
            console=self.logger.stdout,
        )
        super().train()

        # Personalization Phase
        self.unique_model = True
        pfedsim_progress_bar = track(
            range(self.warmup_round, self.args.common.global_epoch),
            "[bold green]Personalizing...",
            console=self.logger.stdout,
        )
        self.trainer.personal_params_name.extend(
            [name for name in self.model.state_dict().keys() if "classifier" in name]
        )
        self.client_trainable_params = [
            [
                self.global_params_dict[key]
                for key in trainable_params(self.model, requires_name=True)[1]
            ]
            for _ in self.train_clients
        ]

        for E in pfedsim_progress_bar:
            self.current_epoch = E

            if (E + 1) % self.args.common.verbose_gap == 0:
                self.logger.log(" " * 30, f"TRAINING EPOCH: {E + 1}", " " * 30)

            if (E + 1) % self.args.common.test_interval == 0:
                self.test()

            self.selected_clients = self.client_sample_stream[E]
            client_params_cache = []
            for client_id in self.selected_clients:
                client_pers_params = self.generate_client_params(client_id)
                (client_params, _, self.client_metrics[client_id][E]) = (
                    self.trainer.train(
                        client_id=client_id,
                        local_epoch=self.clients_local_epoch[client_id],
                        new_parameters=client_pers_params,
                        return_diff=False,
                        verbose=((E + 1) % self.args.common.verbose_gap) == 0,
                    )
                )
                client_params_cache.append(client_params)

            self.update_client_params(client_params_cache)
            self.update_weight_matrix()
            self.log_info()

    @torch.no_grad()
    def generate_client_params(self, client_id):
        if self.current_epoch < self.warmup_round:
            return self.global_params_dict

        new_parameters = OrderedDict(
            zip(
                self.trainable_params_name,
                deepcopy(self.client_trainable_params[client_id]),
            )
        )
        if not self.testing:
            if sum(self.weight_matrix[client_id]) > 1:
                weights = self.weight_matrix[client_id].clone()
                weights[client_id] = 0.9999
                weights = -torch.log(1 - weights)
                weights /= weights.sum()
                for name, layer_params in zip(
                    self.trainable_params_name, zip(*self.client_trainable_params)
                ):
                    new_parameters[name] = torch.sum(
                        torch.stack(layer_params, dim=-1) * weights, dim=-1
                    )
        return new_parameters

    @torch.no_grad()
    def update_weight_matrix(self):
        for idx_i, i in enumerate(self.selected_clients):
            client_params_i = OrderedDict(
                zip(self.trainable_params_name, self.client_trainable_params[i])
            )
            for j in self.selected_clients[idx_i + 1 :]:
                client_params_j = OrderedDict(
                    zip(self.trainable_params_name, self.client_trainable_params[j])
                )
                sim_ij = max(
                    0,
                    torch.cosine_similarity(
                        client_params_i["classifier.weight"],
                        client_params_j["classifier.weight"],
                        dim=-1,
                    ).mean(),
                )

                self.weight_matrix[i, j] = sim_ij
                self.weight_matrix[j, i] = sim_ij
