import math
import time
from argparse import ArgumentParser, Namespace
from collections import OrderedDict

import numpy as np
import torch
from omegaconf import DictConfig
from rich.progress import track

from src.client.fedap import FedAPClient
from src.server.fedavg import FedAvgServer


# Codes below are modified from FedAP's official repo: https://github.com/microsoft/PersonalizedFL
class FedAPServer(FedAvgServer):
    @staticmethod
    def get_hyperparams(args_list=None) -> Namespace:
        parser = ArgumentParser()
        parser.add_argument(
            "--version", type=str, choices=["original", "f", "d"], default="original"
        )
        parser.add_argument("--pretrain_ratio", type=float, default=0.3)
        parser.add_argument("--warmup_round", type=float, default=0.5)
        parser.add_argument("--model_momentum", type=float, default=0.5)
        return parser.parse_args(args_list)

    def __init__(
        self,
        args: DictConfig,
        algorithm_name: str = None,
        unique_model=False,
        use_fedavg_client_cls=False,
        return_diff=False,
    ):
        algo_name = {"original": "FedAP", "f": "f-FedAP", "d": "d-FedAP"}
        algo = algo_name[args.fedap.version]
        super().__init__(
            args, algorithm_name, unique_model, use_fedavg_client_cls, return_diff
        )

        self.init_trainer(FedAPClient)
        self.weight_matrix = torch.eye(self.client_num)
        self.pretrain_round = 0
        if 0 < self.args.fedap.warmup_round < 1:
            self.pretrain_round = int(
                self.args.common.global_epoch * self.args.fedap.warmup_round
            )
        elif 1 <= self.args.fedap.warmup_round < self.args.common.global_epoch:
            self.pretrain_round = int(self.args.fedap.warmup_round)

        self.pretrain = True
        if self.args.fedap.version in ["original", "d"]:
            if not 0 < self.args.fedap.pretrain_ratio < 1:
                raise RuntimeError(
                    "FedAP or d-FedAP need `pretrain_ratio` in the range of [0, 1]."
                )

    def package(self, client_id: int):
        server_package = super().package(client_id)
        if self.pretrain:
            server_package["local_epoch"] = 1
        server_package["pretrain"] = self.pretrain
        return server_package

    def train(self):
        # Pre-training phase
        self.train_progress_bar = track(
            range(self.pretrain_round),
            "[bold green]Warming-up...",
            console=self.logger.stdout,
        )
        for E in self.train_progress_bar:
            self.current_epoch = E

            if self.args.fedap.version == "f":
                self.selected_clients = self.client_sample_stream[E]
            else:
                self.selected_clients = self.train_clients

            if self.args.fedap.version == "f":
                client_packages = self.trainer.train()
                self.aggregate(client_packages)
            else:
                # FedAP and d-FedAP needs one-by-one training this phase
                selected_clients_this_round = self.selected_clients
                for client_id in selected_clients_this_round:
                    self.selected_clients = [client_id]
                    client_package = self.trainer.train()
                    self.public_model_params = client_package[client_id][
                        "regular_model_params"
                    ]

        # update clients params to pretrain params
        for client_id in self.train_clients:
            self.clients_personal_model_params[client_id].update(
                self.public_model_params
            )

        # generate weight matrix
        bn_mean_list, bn_var_list = [], []
        client_packages = self.trainer.exec("get_all_features", self.train_clients)
        for client_id in self.train_clients:
            avgmeta = metacount(self.get_form()[0])
            with torch.no_grad():
                for features, batchsize in zip(
                    client_packages[client_id]["features_list"],
                    client_packages[client_id]["batch_size_list"],
                ):
                    tm, tv = [], []
                    for item in features:
                        if len(item.shape) == 4:
                            tm.append(torch.mean(item, dim=[0, 2, 3]).numpy())
                            tv.append(torch.var(item, dim=[0, 2, 3]).numpy())
                    avgmeta.update(batchsize, tm, tv)
            bn_mean_list.append(avgmeta.getmean())
            bn_var_list.append(avgmeta.getvar())
        self.generate_weight_matrix(bn_mean_list, bn_var_list)

        # regular training
        self.pretrain = False
        self.train_progress_bar = track(
            range(self.pretrain_round, self.args.common.global_epoch),
            "[bold green]Training...",
            console=self.logger.stdout,
        )
        avg_round_time = 0
        for E in self.train_progress_bar:
            self.current_epoch = E
            self.verbose = (self.current_epoch + 1) % self.args.common.verbose_gap == 0
            if self.verbose:
                self.logger.log(" " * 30, f"TRAINING EPOCH: {E + 1}", " " * 30)

            self.selected_clients = self.client_sample_stream[E]

            begin = time.time()
            self.trainer.train()
            end = time.time()
            self.log_info()
            avg_round_time = (avg_round_time * (self.current_epoch) + (end - begin)) / (
                self.current_epoch + 1
            )

            if (E + 1) % self.args.common.test_interval == 0:
                self.test()

        self.logger.log(
            f"{self.algorithm_name}'s average time taken by each global epoch: {int(avg_round_time // 60)} m {(avg_round_time % 60):.2f} s."
        )

    def get_form(self):
        tmp_mean = []
        tmp_var = []
        for name in self.model.state_dict().keys():
            if "running_mean" in name:
                tmp_mean.append(self.model.state_dict()[name].cpu().numpy())
            if "running_var" in name:
                tmp_var.append(self.model.state_dict()[name].cpu().numpy())

        if self.args.fedap.version == "d":
            tmp_mean = [tmp_mean[-1]]
            tmp_var = [tmp_var[-1]]
        return tmp_mean, tmp_var

    def generate_weight_matrix(
        self, bnmlist: list[torch.Tensor], bnvlist: list[torch.Tensor]
    ):
        client_num = len(bnmlist)
        weight_m = np.zeros((client_num, client_num))
        for i in range(client_num):
            for j in range(client_num):
                if i == j:
                    weight_m[i, j] = 0
                else:
                    tmp = wasserstein(bnmlist[i], bnvlist[i], bnmlist[j], bnvlist[j])
                    if tmp == 0:
                        weight_m[i, j] = 100000000000000
                    else:
                        weight_m[i, j] = 1 / tmp
        weight_s = np.sum(weight_m, axis=1)
        weight_s = np.repeat(weight_s, client_num).reshape((client_num, client_num))
        weight_m = (weight_m / weight_s) * (1 - self.args.fedap.model_momentum)
        for i in range(client_num):
            weight_m[i, i] = self.args.fedap.model_momentum
        self.weight_matrix = torch.from_numpy(weight_m)

    def get_client_model_params(self, client_id) -> OrderedDict[str, torch.Tensor]:
        if self.pretrain:
            return super().get_client_model_params(client_id)

        personal_params = self.clients_personal_model_params[client_id]
        if not self.testing:
            for key in self.public_model_param_names:
                layer_params = [
                    model_params[key]
                    for model_params in self.clients_personal_model_params.values()
                ]
                personal_params[key] = torch.sum(
                    torch.stack(layer_params, dim=-1) * self.weight_matrix[client_id],
                    dim=-1,
                )

        return dict(regular_model_params={}, personal_model_params=personal_params)


def wasserstein(m1, v1, m2, v2, mode="nosquare"):
    W = 0
    bn_layer_num = len(m1)
    for i in range(bn_layer_num):
        tw = 0
        tw += np.sum(np.square(m1[i] - m2[i]))
        tw += np.sum(np.square(np.sqrt(v1[i]) - np.sqrt(v2[i])))
        if mode == "square":
            W += tw
        else:
            W += math.sqrt(tw)
    return W


class metacount(object):
    def __init__(self, numpyform):
        super(metacount, self).__init__()
        self.count = 0
        self.mean = []
        self.var = []
        self.bl = len(numpyform)
        for i in range(self.bl):
            self.mean.append(np.zeros(len(numpyform[i])))
            self.var.append(np.zeros(len(numpyform[i])))

    def update(self, m, tm, tv):
        tmpcount = self.count + m
        for i in range(self.bl):
            tmpm = (self.mean[i] * self.count + tm[i] * m) / tmpcount
            self.var[i] = (
                self.count * (self.var[i] + np.square(tmpm - self.mean[i]))
                + m * (tv[i] + np.square(tmpm - tm[i]))
            ) / tmpcount
            self.mean[i] = tmpm
        self.count = tmpcount

    def getmean(self):
        return self.mean

    def getvar(self):
        return self.var
