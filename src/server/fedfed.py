import math
import random
from argparse import ArgumentParser, Namespace
from collections import OrderedDict
from copy import deepcopy
from functools import partial

import numpy as np
import torch
import torch.nn as nn
from omegaconf import DictConfig
from rich.progress import track

from src.client.fedfed import FedFedClient
from src.server.fedavg import FedAvgServer
from src.utils.constants import DATA_SHAPE


class FedFedServer(FedAvgServer):
    @staticmethod
    def get_hyperparams(arg_list=None) -> Namespace:
        parser = ArgumentParser()
        parser.add_argument("--VAE_train_global_epoch", type=int, default=15)
        parser.add_argument("--VAE_train_local_epoch", type=int, default=1)
        parser.add_argument("--VAE_lr", type=float, default=1e-3)
        parser.add_argument("--VAE_weight_decay", type=float, default=1e-6)
        parser.add_argument("--VAE_alpha", type=float, default=2.0)
        parser.add_argument("--VAE_noise_mean", type=float, default=0)
        parser.add_argument("--VAE_noise_std1", type=float, default=0.15)
        parser.add_argument("--VAE_noise_std2", type=float, default=0.25)
        parser.add_argument("--VAE_re", type=float, default=5.0)
        parser.add_argument("--VAE_x_ce", type=float, default=0.4)
        parser.add_argument("--VAE_kl", type=float, default=0.005)
        parser.add_argument("--VAE_ce", type=float, default=2.0)
        parser.add_argument("--VAE_batch_size", type=int, default=64)
        parser.add_argument("--VAE_block_depth", type=int, default=32)
        parser.add_argument(
            "--VAE_noise_type",
            type=str,
            choices=["laplace", "gaussian"],
            default="gaussian",
        )
        return parser.parse_args(arg_list)

    def __init__(
        self,
        args: DictConfig,
        algorithm_name: str = "FedFed",
        unique_model=False,
        use_fedavg_client_cls=False,
        return_diff=False,
    ):
        super().__init__(
            args, algorithm_name, unique_model, use_fedavg_client_cls, return_diff
        )
        dummy_VAE_model = VAE(self.args)
        VAE_optimizer_cls = partial(
            torch.optim.AdamW,
            lr=self.args.fedfed.VAE_lr,
            weight_decay=self.args.fedfed.VAE_weight_decay,
        )
        dummy_VAE_optimizer = VAE_optimizer_cls(params=dummy_VAE_model.parameters())
        self.init_trainer(
            FedFedClient, VAE_cls=VAE, VAE_optimizer_cls=VAE_optimizer_cls
        )
        self.global_VAE_params = OrderedDict()
        for key, param in dummy_VAE_model.named_parameters():
            self.global_VAE_params[key] = param.data.clone()
        self.client_VAE_personal_params = {i: {} for i in self.train_clients}
        self.client_VAE_optimizer_states = {
            i: deepcopy(dummy_VAE_optimizer.state_dict()) for i in self.train_clients
        }
        del dummy_VAE_model, dummy_VAE_optimizer

    def train(self):
        # do the feature distillation before regular FL training
        self.feature_distill()
        super().train()

    def feature_distill(self):
        """Train VAE, generate shared data, distribute shared data."""

        def _package_VAE(client_id: int):
            server_package = self.package(client_id)
            server_package["distilling"] = True
            server_package["VAE_regular_params"] = self.global_VAE_params
            server_package["VAE_personal_params"] = self.client_VAE_personal_params.get(
                client_id
            )
            server_package["VAE_optimizer_state"] = (
                self.client_VAE_optimizer_states.get(client_id)
            )
            return server_package

        num_join = max(1, int(self.args.common.join_ratio * len(self.train_clients)))
        self.train_progress_bar = track(
            range(self.args.fedfed.VAE_train_global_epoch),
            description="[magenta bold]Training VAE...",
            console=self.logger.stdout,
        )
        for _ in self.train_progress_bar:
            selected_clients = random.sample(self.train_clients, num_join)
            client_packages = self.trainer.exec(
                func_name="train_VAE",
                clients=selected_clients,
                package_func=_package_VAE,
            )
            for client_id, package in client_packages.items():
                self.clients_personal_model_params[client_id] = package[
                    "personal_model_params"
                ]
                self.client_optimizer_states[client_id] = package["optimizer_state"]

                self.client_VAE_personal_params[client_id] = package[
                    "VAE_personal_params"
                ]
                self.client_VAE_optimizer_states[client_id] = package[
                    "VAE_optimizer_state"
                ]
            super().aggregate(client_packages)

            # aggregate client VAEs
            weights = torch.tensor(
                [package["weight"] for package in client_packages.values()],
                dtype=torch.float,
            )
            weights /= weights.sum()
            for key, global_param in self.global_VAE_params.items():
                client_VAE_regular_params = torch.stack(
                    [
                        package["VAE_regular_params"][key]
                        for package in client_packages.values()
                    ],
                    dim=-1,
                )
                global_param.data = torch.sum(
                    client_VAE_regular_params * weights, dim=-1
                )

        # gather client performance-sensitive data
        client_packages = self.trainer.exec(
            func_name="generate_shared_data",
            clients=self.train_clients,
            package_func=_package_VAE,
        )
        data1, data2, targets = [], [], []
        for package in client_packages.values():
            data1.append(package["data1"])
            data2.append(package["data2"])
            targets.append(package["targets"])

        global_shared_data1 = torch.cat(data1)
        global_shared_data2 = torch.cat(data2)
        global_shared_targets = torch.cat(targets)

        # distribute global shared
        def _package_distribute_data(client_id: int):
            nonlocal global_shared_data1, global_shared_data2, global_shared_targets
            return dict(
                client_id=client_id,
                data1=global_shared_data1,
                data2=global_shared_data2,
                targets=global_shared_targets,
            )

        self.trainer.exec(
            func_name="accept_global_shared_data",
            clients=self.train_clients,
            package_func=_package_distribute_data,
        )

        # restore the progress bar for regular FL training
        self.train_progress_bar = track(
            range(self.args.common.global_epoch),
            "[bold green]Training...",
            console=self.logger.stdout,
        )


# Modified from the official codes
class VAE(nn.Module):
    def __init__(self, args):
        super(VAE, self).__init__()

        class ResidualBlock(nn.Module):
            def __init__(self, in_channels, out_channels=None):
                super(ResidualBlock, self).__init__()
                if out_channels is None:
                    out_channels = in_channels
                layers = [
                    nn.LeakyReLU(),
                    nn.Conv2d(
                        in_channels, out_channels, kernel_size=3, stride=1, padding=1
                    ),
                    nn.BatchNorm2d(out_channels),
                    nn.LeakyReLU(),
                    nn.Conv2d(
                        out_channels, out_channels, kernel_size=1, stride=1, padding=0
                    ),
                ]
                self.block = nn.Sequential(*layers)

            def forward(self, x):
                return x + self.block(x)

        self.args = deepcopy(args)
        img_depth = DATA_SHAPE[self.args.dataset.name][0]
        img_shape = DATA_SHAPE[self.args.dataset.name][:-1]

        dummy_input = torch.randn(2, *DATA_SHAPE[self.args.dataset.name])
        self.encoder = nn.Sequential(
            nn.Conv2d(
                img_depth,
                self.args.fedfed.VAE_block_depth // 2,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(self.args.fedfed.VAE_block_depth // 2),
            nn.ReLU(),
            nn.Conv2d(
                self.args.fedfed.VAE_block_depth // 2,
                self.args.fedfed.VAE_block_depth,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(self.args.fedfed.VAE_block_depth),
            nn.ReLU(),
            ResidualBlock(self.args.fedfed.VAE_block_depth),
            nn.BatchNorm2d(self.args.fedfed.VAE_block_depth),
            ResidualBlock(self.args.fedfed.VAE_block_depth),
        )
        with torch.no_grad():
            dummy_feature = self.encoder(dummy_input)
        self.feature_length = dummy_feature.flatten(start_dim=1).shape[-1]
        self.feature_side = int(
            math.sqrt(self.feature_length // self.args.fedfed.VAE_block_depth)
        )

        self.decoder = nn.Sequential(
            ResidualBlock(self.args.fedfed.VAE_block_depth),
            nn.BatchNorm2d(self.args.fedfed.VAE_block_depth),
            ResidualBlock(self.args.fedfed.VAE_block_depth),
            nn.BatchNorm2d(self.args.fedfed.VAE_block_depth),
            nn.ConvTranspose2d(
                self.args.fedfed.VAE_block_depth,
                self.args.fedfed.VAE_block_depth // 2,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(self.args.fedfed.VAE_block_depth // 2),
            nn.LeakyReLU(),  # really confused me here
            # in the offcial codes, they use Tanh() right after LeakyReLU() what???
            nn.Tanh(),
            # BTW, FedFed's codes of beta VAE is hugely different from other reproductions,
            # such as https://github.com/AntixK/PyTorch-VAE/blob/master/models/beta_vae.py
            nn.ConvTranspose2d(
                self.args.fedfed.VAE_block_depth // 2,
                img_depth,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(img_depth),
            nn.Sigmoid(),
        )

        self.fc_mu = nn.Linear(self.feature_length, self.feature_length)
        self.fc_logvar = nn.Linear(self.feature_length, self.feature_length)
        self.decoder_input = nn.Linear(self.feature_length, self.feature_length)

        if args.common.buffers == "global":
            for module in self.modules():
                if isinstance(module, torch.nn.BatchNorm2d):
                    buffers_list = list(module.named_buffers())
                    for name_buffer, buffer in buffers_list:
                        # transform buffer to parameter
                        # for showing out in parameters()
                        delattr(module, name_buffer)
                        module.register_parameter(
                            name_buffer,
                            torch.nn.Parameter(buffer.float(), requires_grad=False),
                        )

    def add_noise(self, data: torch.Tensor, mean, std):
        if self.args.fedfed.VAE_noise_type == "gaussian":
            noise = torch.normal(
                mean=mean, std=std, size=data.shape, device=data.device
            )
        if self.args.fedfed.VAE_noise_type == "laplace":
            noise = torch.tensor(
                np.random.laplace(loc=mean, scale=std, size=data.shape),
                device=data.device,
            )
        return data + noise

    def encode(self, x):
        x = self.encoder(x).flatten(start_dim=1, end_dim=-1)
        return self.fc_mu(x), self.fc_logvar(x)

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor):
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std, device=std.device)
            return eps * std + mu
        else:
            return mu

    def decode(self, z):
        result = self.decoder_input(z)
        result = result.view(
            -1, self.args.fedfed.VAE_block_depth, self.feature_side, self.feature_side
        )
        return self.decoder(result)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        robust = self.decode(z)
        return robust, mu, logvar
