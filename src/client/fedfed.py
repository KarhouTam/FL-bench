import gc
from copy import deepcopy
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from src.client.fedavg import FedAvgClient


class FedFedClient(FedAvgClient):
    def __init__(self, VAE_cls, VAE_optimizer_cls, **commons):
        super().__init__(**commons)
        self.VAE: torch.nn.Module = VAE_cls(self.args).to(self.device)
        self.VAE_optimizer: torch.optim.Optimizer = VAE_optimizer_cls(
            params=self.VAE.parameters()
        )
        self.offset_ori_dataset = len(self.dataset)
        self.distilling = True

    def set_parameters(self, package: dict[str, Any]):
        self.distilling = package.get("distilling", False)
        super().set_parameters(package)
        if self.distilling:
            self.VAE.load_state_dict(package["VAE_regular_params"], strict=False)
            self.VAE.load_state_dict(package["VAE_personal_params"], strict=False)
            self.VAE_optimizer.load_state_dict(package["VAE_optimizer_state"])

    def load_data_indices(self):
        if self.distilling:
            self.trainset.indices = self.data_indices[self.client_id]["train"]
        else:
            idxs_shared = np.random.choice(
                len(self.dataset) - self.offset_ori_dataset,
                len(self.data_indices[self.client_id]["train"]),
                replace=False,
            )
            self.trainset.indices = np.concatenate(
                [self.data_indices[self.client_id]["train"] + idxs_shared]
            )
        self.valset.indices = self.data_indices[self.client_id]["val"]
        self.testset.indices = self.data_indices[self.client_id]["test"]

    def train_VAE(self, package: dict[str, Any]):
        self.set_parameters(package)
        self.model.train()
        self.dataset.train()
        for _ in range(self.args.fedfed.VAE_train_local_epoch):
            self.VAE.eval()
            for x, y in self.trainloader:
                if len(y) <= 1:
                    continue
                x, y = x.to(self.device), y.to(self.device)
                x_mixed, y_ori, y_rand, lamda = self.mixup_data(x, y)

                logits = self.model(x_mixed)
                loss_classifier = lamda * F.cross_entropy(logits, y_ori) + (
                    1 - lamda
                ) * F.cross_entropy(logits, y_rand)
                self.optimizer.zero_grad()
                loss_classifier.backward()
                self.optimizer.step()

            self.VAE.train()
            for x, y in self.trainloader:
                if len(y) <= 1:
                    continue
                x, y = x.to(self.device), y.to(self.device)

                robust, mu, logvar = self.VAE(x)

                loss_VAE = (self.args.fedfed.VAE_re * F.mse_loss(robust, x)) + (
                    self.args.fedfed.VAE_kl
                    * (-0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()))
                    / (self.args.fedfed.VAE_batch_size * 3 * self.VAE.feature_length)
                )
                self.VAE_optimizer.zero_grad()
                loss_VAE.backward()
                self.VAE_optimizer.step()

            for x, y in self.trainloader:
                if len(y) <= 1:
                    continue
                x, y = x.to(self.device), y.to(self.device)
                batch_size = x.shape[0]
                robust, mu, logvar = self.VAE(x)
                sensitive = x - robust
                sensitive_protected1 = self.VAE.add_noise(
                    sensitive,
                    self.args.fedfed.VAE_noise_mean,
                    self.args.fedfed.VAE_noise_std1,
                )
                sensitive_protected2 = self.VAE.add_noise(
                    sensitive,
                    self.args.fedfed.VAE_noise_mean,
                    self.args.fedfed.VAE_noise_std2,
                )
                data = torch.cat([sensitive_protected1, sensitive_protected2, x])
                logits = self.model(data)

                loss_features_sensitive_protected = F.cross_entropy(
                    logits[: batch_size * 2], y.repeat(2)
                )
                loss_x = F.cross_entropy(logits[batch_size * 2 :], y)
                loss_mse = F.mse_loss(robust, x)
                loss_kl = (
                    -0.5
                    * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
                    / (self.args.fedfed.VAE_batch_size * 3 * self.VAE.feature_length)
                )

                loss = (
                    self.args.fedfed.VAE_re * loss_mse
                    + self.args.fedfed.VAE_kl * loss_kl
                    + self.args.fedfed.VAE_ce * loss_features_sensitive_protected
                    + self.args.fedfed.VAE_x_ce * loss_x
                )

                self.VAE_optimizer.zero_grad()
                self.optimizer.zero_grad()
                loss.backward()
                self.VAE_optimizer.step()
                self.optimizer.step()

        VAE_regular_params, VAE_personal_params = {}, {}
        VAE_regular_param_names = list(key for key, _ in self.VAE.named_parameters())
        VAE_model_params = self.VAE.state_dict()
        for key, param in VAE_model_params.items():
            if key in VAE_regular_param_names:
                VAE_regular_params[key] = param.clone().cpu()
            else:
                VAE_personal_params[key] = param.clone().cpu()
        client_package = super().package()
        client_package["VAE_regular_params"] = VAE_regular_params
        client_package["VAE_personal_params"] = VAE_personal_params
        client_package["VAE_optimizer_state"] = deepcopy(
            self.VAE_optimizer.state_dict()
        )
        return client_package

    @torch.no_grad()
    def generate_shared_data(self, package: dict[str, Any]):
        self.set_parameters(package)
        self.dataset.eval()
        self.VAE.eval()

        data1 = []
        data2 = []
        targets = []
        for x, y in self.trainloader:
            x, y = x.to(self.device), y.to(self.device)

            robust, _, _ = self.VAE(x)
            sensitive = x - robust
            data1.append(
                self.VAE.add_noise(
                    sensitive,
                    self.args.fedfed.VAE_noise_mean,
                    self.args.fedfed.VAE_noise_std1,
                )
            )
            data2.append(
                self.VAE.add_noise(
                    sensitive,
                    self.args.fedfed.VAE_noise_mean,
                    self.args.fedfed.VAE_noise_std2,
                )
            )
            targets.append(y)

        data1 = torch.cat(data1).float().cpu()
        data2 = torch.cat(data2).float().cpu()
        targets = torch.cat(targets).long().cpu()

        return dict(data1=data1, data2=data2, targets=targets)

    def accept_global_shared_data(self, package: dict[str, Any]):
        # avoid loading multiple times
        # only trigger once per worker (worker != client)
        # serial training mode (1 worker)
        # parallel training mode (args.parallel.num_workers workers (>= 2))
        if self.distilling:
            self.distilling = False

            # regular training doesn't need VAE
            del self.VAE, self.VAE_optimizer
            gc.collect()
            torch.cuda.empty_cache()

            self.dataset.data = torch.cat(
                [self.dataset.data, package["data1"], package["data2"]]
            )
            self.dataset.targets = torch.cat(
                [self.dataset.targets, package["targets"], package["targets"]]
            )

    def mixup_data(self, x: torch.Tensor, y: torch.Tensor):
        if self.args.fedfed.VAE_alpha > 0:
            lamda = np.random.beta(
                self.args.fedfed.VAE_alpha, self.args.fedfed.VAE_alpha
            )
        else:
            lamda = 1.0

        shfl_idxs = np.random.permutation(x.shape[0])
        x_mixed = lamda * x + (1 - lamda) * x[shfl_idxs, :]
        return x_mixed, y, y[shfl_idxs], lamda
