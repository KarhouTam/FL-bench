import time
from argparse import ArgumentParser, Namespace
from copy import deepcopy

import torch
from omegaconf import DictConfig
from rich.progress import track

from src.server.fedavg import FedAvgServer


class pFedSimServer(FedAvgServer):
    @staticmethod
    def get_hyperparams(args_list=None) -> Namespace:
        parser = ArgumentParser()
        parser.add_argument("-wr", "--warmup_round", type=float, default=0.5)
        return parser.parse_args(args_list)

    def __init__(self, args: DictConfig, algorithm_name: str = "pFedSim"):
        # layers join aggregation in personalization phase
        self.params_name_join_aggregation = None
        self.warmup_round = 0
        if 0 <= args.pfedsim.warmup_round <= 1:
            self.warmup_round = int(
                args.common.global_epoch * args.pfedsim.warmup_round
            )
        elif 1 < args.pfedsim.warmup_round < args.common.global_epoch:
            self.warmup_round = int(args.pfedsim.warmup_round)
        else:
            raise ValueError(
                "warmup_round need to be set in the range of [0, 1) or [1, global_epoch)."
            )
        super().__init__(args, algorithm_name)
        self.weight_matrix = torch.eye(self.client_num)

    def train(self):
        # Warm-up Phase
        self.train_progress_bar = track(
            range(self.warmup_round),
            f"[bold cyan]Warming-up...",
            console=self.logger.stdout,
        )
        super().train()

        # Personalization Phase
        avg_round_time = 0
        self.train_progress_bar = track(
            range(self.warmup_round, self.args.common.global_epoch),
            "[bold green]Personalizing...",
            console=self.logger.stdout,
        )

        for params_dict in self.clients_personal_model_params.values():
            params_dict.update(self.public_model_params)

        self.params_name_join_aggregation = [
            key for key in self.public_model_params.keys() if "classifier" not in key
        ]

        for E in self.train_progress_bar:
            self.current_epoch = E
            self.selected_clients = self.client_sample_stream[E]
            self.verbose = (E + 1) % self.args.common.verbose_gap == 0
            if self.verbose:
                self.logger.log("-" * 28, f"TRAINING EPOCH: {E + 1}", "-" * 28)

            begin = time.time()
            client_packages = self.trainer.train()
            end = time.time()
            for client_id in self.selected_clients:
                if not self.return_diff:
                    self.clients_personal_model_params[client_id].update(
                        client_packages[client_id]["regular_model_params"]
                    )
                else:
                    self.clients_personal_model_params[client_id].update(
                        {
                            key: self.clients_personal_model_params[client_id][key]
                            - client_packages[client_id]["model_params_diff"][key]
                            for key in self.public_model_param_names
                        }
                    )
            self.update_weight_matrix()
            self.log_info()
            avg_round_time = (avg_round_time * self.current_epoch + (end - begin)) / (
                self.current_epoch + 1
            )

            if (E + 1) % self.args.common.test_interval == 0:
                self.test()

        self.logger.log(
            f"{self.algorithm_name}'s average time taken by each global epoch: "
            f"{int(avg_round_time // 60)} min {(avg_round_time % 60):.2f} sec."
        )

    @torch.no_grad()
    def get_client_model_params(self, client_id):
        if self.current_epoch < self.warmup_round:
            return super().get_client_model_params(client_id)
        if self.testing:
            return dict(
                regular_model_params={},
                personal_model_params=self.clients_personal_model_params[client_id],
            )
        pfedsim_aggregated_params = deepcopy(
            self.clients_personal_model_params[client_id]
        )
        clients_model_params_list = [
            [params_dict[key] for key in self.params_name_join_aggregation]
            for params_dict in self.clients_personal_model_params.values()
        ]
        if sum(self.weight_matrix[client_id]) > 1:
            weights = self.weight_matrix[client_id].clone()
            weights[client_id] = 0.9999
            weights = -torch.log(1 - weights)
            weights /= weights.sum()
            for name, layer_params in zip(
                self.params_name_join_aggregation, zip(*clients_model_params_list)
            ):
                pfedsim_aggregated_params[name] = torch.sum(
                    torch.stack(layer_params, dim=-1) * weights, dim=-1
                )
        return dict(
            regular_model_params={},
            personal_model_params=pfedsim_aggregated_params,  # includes all params
        )

    def update_weight_matrix(self):
        for idx_i, i in enumerate(self.selected_clients):
            for j in self.selected_clients[idx_i + 1 :]:
                sim_ij = max(
                    0,
                    torch.cosine_similarity(
                        self.clients_personal_model_params[i]["classifier.weight"],
                        self.clients_personal_model_params[j]["classifier.weight"],
                        dim=-1,
                    ).mean(),
                )

                self.weight_matrix[i, j] = sim_ij
                self.weight_matrix[j, i] = sim_ij
