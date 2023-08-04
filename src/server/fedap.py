import math
from argparse import ArgumentParser, Namespace
from collections import OrderedDict
from copy import deepcopy
from typing import List

import torch
import numpy as np
from rich.progress import track

from fedavg import FedAvgServer, get_fedavg_argparser
from src.config.utils import trainable_params
from src.client.fedap import FedAPClient


def get_fedap_argparser() -> ArgumentParser:
    parser = get_fedavg_argparser()
    parser.add_argument(
        "--version", type=str, choices=["original", "f", "d"], default="original"
    )
    parser.add_argument("--pretrain_ratio", type=float, default=0.3)
    parser.add_argument("--warmup_round", type=float, default=0.5)
    parser.add_argument("--model_momentum", type=float, default=0.5)
    return parser


# Codes below are modified from FedAP's official repo: https://github.com/microsoft/PersonalizedFL
class FedAPServer(FedAvgServer):
    def __init__(
        self,
        algo: str = None,
        args: Namespace = None,
        unique_model=True,
        default_trainer=False,
    ):
        if args is None:
            args = get_fedap_argparser().parse_args()
        algo_name = {"original": "FedAP", "f": "f-FedAP", "d": "d-FedAP"}
        algo = algo_name[args.version]
        super().__init__(algo, args, unique_model, default_trainer)

        self.trainer = FedAPClient(
            deepcopy(self.model), self.args, self.logger, self.device
        )
        self.weight_matrix = torch.eye(self.client_num, device=self.device)
        self.warmup_round = 0
        if 0 < self.args.warmup_round < 1:
            self.warmup_round = int(self.args.global_epoch * self.args.warmup_round)
        elif 1 <= self.args.warmup_round < self.args.global_epoch:
            self.warmup_round = int(self.args.warmup_round)

    def train(self):
        # Pre-training phase
        if self.args.version in ["original", "d"]:
            if 0 < self.args.pretrain_ratio < 1:
                self.trainer.pretrain = True
            else:
                raise RuntimeError(
                    "FedAP or d-FedAP need `pretrain_ratio` in the range of [0, 1]."
                )

        warmup_progress_bar = track(
            range(self.warmup_round),
            "[bold green]Warming-up...",
            console=self.logger.stdout,
        )
        for E in warmup_progress_bar:
            self.current_epoch = E

            if self.args.version == "f":
                self.selected_clients = self.client_sample_stream[E]
            else:
                self.selected_clients = list(range(self.client_num))

            delta_cache = []
            weight_cache = []
            for client_id in self.selected_clients:
                (delta, weight, self.client_stats[client_id][E]) = self.trainer.train(
                    client_id=client_id,
                    local_epoch=self.clients_local_epoch[client_id],
                    new_parameters=self.global_params_dict,
                    verbose=((E + 1) % self.args.verbose_gap) == 0,
                )
                if self.args.version == "f":
                    delta_cache.append(delta)
                    weight_cache.append(weight)
                else:
                    for old_param, diff in zip(
                        self.global_params_dict.values(), delta.values()
                    ):
                        old_param.data -= diff.data

            if self.args.version == "f":
                self.aggregate(delta_cache, weight_cache)

            self.log_info()

        # update clients params to pretrain params
        self.client_trainable_params = [
            trainable_params(self.global_params_dict, detach=True)
            for _ in self.train_clients
        ]

        # generate weight matrix
        bn_mean_list, bn_var_list = [], []
        for client_id in track(
            self.train_clients,
            "[bold cyan]Generating weight matrix...",
            console=self.logger.stdout,
        ):
            avgmeta = metacount(self.get_form()[0])
            client_local_params = self.generate_client_params(client_id)
            features_list, batch_size_list = self.trainer.get_all_features(
                client_id, client_local_params
            )
            with torch.no_grad():
                for features, batchsize in zip(features_list, batch_size_list):
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
        self.train_progress_bar = track(
            range(self.warmup_round, self.args.global_epoch),
            "[bold green]Training...",
            console=self.logger.stdout,
        )
        self.trainer.pretrain = False
        for E in self.train_progress_bar:
            self.current_epoch = E

            if (E + 1) % self.args.verbose_gap == 0:
                self.logger.log(" " * 30, f"TRAINING EPOCH: {E + 1}", " " * 30)

            if (E + 1) % self.args.test_gap == 0:
                self.test()

            self.selected_clients = self.client_sample_stream[E]

            delta_cache = []
            for client_id in self.selected_clients:
                client_local_params = self.generate_client_params(client_id)
                delta, _, self.client_stats[client_id][E] = self.trainer.train(
                    client_id=client_id,
                    local_epoch=self.clients_local_epoch[client_id],
                    new_parameters=client_local_params,
                    return_diff=False,
                    verbose=((E + 1) % self.args.verbose_gap) == 0,
                )

                delta_cache.append(delta)

            self.update_client_params(delta_cache)
            self.log_info()

    def get_form(self):
        tmp_mean = []
        tmp_var = []
        for name in self.model.state_dict().keys():
            if "running_mean" in name:
                tmp_mean.append(self.model.state_dict()[name].cpu().numpy())
            if "running_var" in name:
                tmp_var.append(self.model.state_dict()[name].cpu().numpy())

        if self.args.version == "d":
            tmp_mean = [tmp_mean[-1]]
            tmp_var = [tmp_var[-1]]
        return tmp_mean, tmp_var

    def generate_weight_matrix(
        self, bnmlist: List[torch.Tensor], bnvlist: List[torch.Tensor]
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
        weight_m = (weight_m / weight_s) * (1 - self.args.model_momentum)
        for i in range(client_num):
            weight_m[i, i] = self.args.model_momentum
        self.weight_matrix = torch.from_numpy(weight_m).to(self.device)

    def generate_client_params(self, client_id) -> OrderedDict[str, torch.Tensor]:
        new_parameters = OrderedDict()
        for name, layer_params in zip(
            self.trainable_params_name, zip(*self.client_trainable_params)
        ):
            new_parameters[name] = torch.sum(
                torch.stack(layer_params, dim=-1) * self.weight_matrix[client_id],
                dim=-1,
            )
        return new_parameters


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


if __name__ == "__main__":
    server = FedAPServer()
    server.run()
