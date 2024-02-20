from argparse import ArgumentParser, Namespace
from collections import OrderedDict
from copy import deepcopy

import torch
from rich.progress import track

from fedavg import FedAvgServer, get_fedavg_argparser
from src.utils.tools import trainable_params


def get_pfedsim_argparser() -> ArgumentParser:
    parser = get_fedavg_argparser()
    parser.add_argument("-wr", "--warmup_round", type=float, default=0.5)
    return parser


class pFedSimServer(FedAvgServer):
    def __init__(
        self,
        algo: str = "pFedSim",
        args: Namespace = None,
        unique_model=False,
        default_trainer=True,
    ):
        if args is None:
            args = get_pfedsim_argparser().parse_args()
        super().__init__(algo, args, unique_model, default_trainer)
        self.test_flag = False
        self.weight_matrix = torch.eye(self.client_num, device=self.device)

        self.warmup_round = 0
        if 0 <= self.args.warmup_round <= 1:
            self.warmup_round = int(self.args.global_epoch * self.args.warmup_round)
        elif 1 < self.args.warmup_round < self.args.global_epoch:
            self.warmup_round = int(self.args.warmup_round)
        else:
            raise RuntimeError(
                "args.warmup_round need to be set in the range of [0, 1) or [1, args.global_epoch)."
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
            range(self.warmup_round, self.args.global_epoch),
            "[bold green]Personalizing...",
            console=self.logger.stdout,
        )
        self.trainer.personal_params_name.extend(
            [name for name in self.model.state_dict() if "classifier" in name]
        )
        self.client_trainable_params = [
            trainable_params(self.global_params_dict, detach=True)
            for _ in self.train_clients
        ]

        for E in pfedsim_progress_bar:
            self.current_epoch = E

            if (E + 1) % self.args.verbose_gap == 0:
                self.logger.log(" " * 30, f"TRAINING EPOCH: {E + 1}", " " * 30)

            if (E + 1) % self.args.test_gap == 0:
                self.test()

            self.selected_clients = self.client_sample_stream[E]
            client_params_cache = []
            for client_id in self.selected_clients:
                client_pers_params = self.generate_client_params(client_id)
                (
                    client_params,
                    _,
                    self.client_metrics[client_id][E],
                ) = self.trainer.train(
                    client_id=client_id,
                    local_epoch=self.clients_local_epoch[client_id],
                    new_parameters=client_pers_params,
                    return_diff=False,
                    verbose=((E + 1) % self.args.verbose_gap) == 0,
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
        if not self.test_flag:
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


if __name__ == "__main__":
    server = pFedSimServer()
    server.run()
