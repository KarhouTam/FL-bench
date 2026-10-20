from argparse import ArgumentParser, Namespace
from collections import OrderedDict
from typing import Any, Dict

import numpy as np
import torch
from omegaconf import DictConfig

from src.client.dtfl import DTFLClient
from src.server.fedavg import FedAvgServer


class DTFLServer(FedAvgServer):
    """Server implementation for Dynamic Tiering Federated Learning (DTFL).

    Reference:
        Seyed Mahmoud Sajjadi Mohammadabadi, Sydneyuzorert Huffman, Lei Yang, and Feng Yan,
        "Speed Up Federated Learning in Heterogeneous Environments: A Dynamic Tiering Approach,"
        IEEE Internet of Things Journal (IoT-J), 2024.
        DOI: https://doi.org/10.1109/JIOT.2024.3468511
    """

    algorithm_name: str = "DTFL"
    all_model_params_personalized = False
    return_diff = False
    client_cls = DTFLClient

    @staticmethod
    def get_hyperparams(args_list=None) -> Namespace:
        parser = ArgumentParser()
        parser.add_argument(
            "--num_tiers",
            type=int,
            default=3,
            help="Number of dynamic tiers for heterogeneous clients.",
        )
        parser.add_argument(
            "--tier_update_interval",
            type=int,
            default=1,
            help="Communication round interval for dynamic re-tiering.",
        )
        parser.add_argument(
            "--min_local_epoch",
            type=int,
            default=1,
            help="Minimum local training epochs assigned to the slowest tier.",
        )
        parser.add_argument(
            "--max_local_epoch",
            type=int,
            default=5,
            help="Maximum local training epochs assigned to the fastest tier.",
        )
        parser.add_argument(
            "--ema_alpha",
            type=float,
            default=0.6,
            help="Exponential moving average factor for client latency and utility profiling.",
        )
        parser.add_argument(
            "--tier_balance_beta",
            type=float,
            default=0.5,
            help="Strength of tier-aware staleness and work-normalization weighting during aggregation.",
        )
        return parser.parse_args(args_list)

    def __init__(
        self,
        args: DictConfig,
        init_trainer: bool = True,
        init_model: bool = True,
    ):
        super().__init__(args, init_trainer=init_trainer, init_model=init_model)

        self.num_tiers = max(1, int(self.args.dtfl.num_tiers))
        self.min_local_epoch = max(1, int(self.args.dtfl.min_local_epoch))
        self.max_local_epoch = max(
            self.min_local_epoch,
            int(getattr(self.args.dtfl, "max_local_epoch", self.args.common.local_epoch)),
        )

        # Dynamic profiling state across all clients
        self.client_tiers = {i: 0 for i in range(self.client_num)}
        self.client_step_times = {i: None for i in range(self.client_num)}
        self.client_utilities = {i: 1.0 for i in range(self.client_num)}
        self.client_last_round = {i: 0 for i in range(self.client_num)}

        # Precompute local epoch schedule per tier: Tier 0 (fastest) -> Tier M-1 (slowest)
        if self.num_tiers == 1:
            self.tier_epochs = [self.args.common.local_epoch]
        else:
            self.tier_epochs = [
                int(
                    round(
                        self.max_local_epoch
                        - t
                        * (self.max_local_epoch - self.min_local_epoch)
                        / (self.num_tiers - 1)
                    )
                )
                for t in range(self.num_tiers)
            ]

    def package(self, client_id: int) -> dict[str, Any]:
        server_package = super().package(client_id)
        tier = self.client_tiers.get(client_id, 0)
        server_package["tier"] = tier
        if not self.testing:
            server_package["local_epoch"] = self.tier_epochs[tier]
        return server_package

    def train_one_round(self):
        client_packages = self.trainer.train()
        self.update_client_profiles_and_tiers(client_packages)
        self.aggregate_client_updates(client_packages)

    def update_client_profiles_and_tiers(
        self, client_packages: OrderedDict[int, Dict[str, Any]]
    ):
        alpha = float(self.args.dtfl.ema_alpha)
        for client_id, pkg in client_packages.items():
            step_time = float(pkg.get("step_time", 1e-4))
            avg_loss = float(pkg.get("avg_loss", 1.0))

            if self.client_step_times[client_id] is None:
                self.client_step_times[client_id] = step_time
            else:
                self.client_step_times[client_id] = (
                    alpha * step_time + (1.0 - alpha) * self.client_step_times[client_id]
                )

            self.client_utilities[client_id] = (
                alpha * max(avg_loss, 1e-6)
                + (1.0 - alpha) * self.client_utilities[client_id]
            )

        if (self.current_epoch + 1) % max(1, int(self.args.dtfl.tier_update_interval)) == 0:
            profiled_clients = [
                cid
                for cid, t_val in self.client_step_times.items()
                if t_val is not None
            ]
            if len(profiled_clients) >= self.num_tiers and self.num_tiers > 1:
                times = np.array([self.client_step_times[cid] for cid in profiled_clients])
                quantiles = np.linspace(0.0, 1.0, self.num_tiers + 1)[1:-1]
                thresholds = np.quantile(times, quantiles)
                for cid in profiled_clients:
                    tier_idx = int(np.searchsorted(thresholds, self.client_step_times[cid]))
                    tier_idx = min(max(tier_idx, 0), self.num_tiers - 1)
                    self.client_tiers[cid] = tier_idx
                    self.client_local_epoches[cid] = self.tier_epochs[tier_idx]

    @torch.no_grad()
    def aggregate_client_updates(
        self, client_packages: OrderedDict[int, Dict[str, Any]]
    ):
        beta = float(self.args.dtfl.tier_balance_beta)
        effective_weights = []

        for client_id, pkg in client_packages.items():
            data_weight = float(pkg["weight"])
            tier = self.client_tiers.get(client_id, 0)
            assigned_epochs = max(1, self.tier_epochs[tier])

            # Normalize for heterogeneous local epochs across tiers and account for staleness
            staleness = max(1, (self.current_epoch + 1) - self.client_last_round[client_id])
            staleness_factor = 1.0 / (1.0 + 0.1 * (staleness - 1))
            epoch_norm = (self.max_local_epoch / assigned_epochs) ** beta

            effective_weights.append(data_weight * epoch_norm * staleness_factor)
            self.client_last_round[client_id] = self.current_epoch + 1

        weights = torch.tensor(effective_weights, dtype=torch.float32)
        weights = weights / torch.sum(weights)

        if self.return_diff:
            for name, global_param in self.public_model_params.items():
                diffs = torch.stack(
                    [
                        package["model_params_diff"][name]
                        for package in client_packages.values()
                    ],
                    dim=-1,
                )
                aggregated = torch.sum(diffs * weights, dim=-1)
                self.public_model_params[name].data -= aggregated
        else:
            for name, global_param in self.public_model_params.items():
                client_params = torch.stack(
                    [
                        package["regular_model_params"][name]
                        for package in client_packages.values()
                    ],
                    dim=-1,
                )
                aggregated = torch.sum(client_params * weights, dim=-1)
                global_param.data = aggregated

        self.model.load_state_dict(self.public_model_params, strict=False)