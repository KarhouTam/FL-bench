import random
from collections import OrderedDict
from copy import deepcopy
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from src.client.fedavg import FedAvgClient


class FedALAClient(FedAvgClient):
    def __init__(self, **commons):
        super().__init__(**commons)
        self.weights = None  # Learnable local aggregation weights.
        self.start_phase = True
        self.sampled_trainset = Subset(self.dataset, indices=[])
        self.sampled_trainloader = DataLoader(
            self.sampled_trainset, self.args.common.batch_size
        )

    def load_data_indices(self):
        train_data_indices = deepcopy(self.data_indices[self.client_id]["train"])
        random.shuffle(train_data_indices)
        sampled_size = int(len(train_data_indices) * self.args.fedala.rand_percent)
        self.sampled_trainset.indices = train_data_indices[:sampled_size]

        self.trainset.indices = self.data_indices[self.client_id]["train"]
        self.valset.indices = self.data_indices[self.client_id]["val"]
        self.testset.indices = self.data_indices[self.client_id]["test"]

    def set_parameters(self, package: dict[str, Any]) -> None:
        self.client_id = package["client_id"]
        self.local_epoch = package["local_epoch"]
        self.load_data_indices()

        if package["optimizer_state"]:
            self.optimizer.load_state_dict(package["optimizer_state"])
        else:
            self.optimizer.load_state_dict(self.init_optimizer_state)

        if self.lr_scheduler is not None:
            if package["lr_scheduler_state"]:
                self.lr_scheduler.load_state_dict(package["lr_scheduler_state"])
            else:
                self.lr_scheduler.load_state_dict(self.init_lr_scheduler_state)

        # obtain the references of the parameters
        global_model = deepcopy(self.model)
        global_model.load_state_dict(package["regular_model_params"], strict=False)
        params_g = list(global_model.parameters())
        params = list(self.model.parameters())

        # deactivate ALA at the 1st communication iteration
        if torch.sum(params_g[0] - params[0]) == 0:
            self.model.load_state_dict(package["regular_model_params"], strict=False)
            self.model.load_state_dict(package["personal_model_params"], strict=False)
            if self.args.common.buffers == "drop":
                self.model.load_state_dict(self.init_buffers, strict=False)

            if self.return_diff:
                model_params = self.model.state_dict()
                self.regular_model_params = OrderedDict(
                    (key, model_params[key].clone().cpu())
                    for key in self.regular_params_name
                )

        # preserve all the updates in the lower layers
        for param, param_g in zip(
            params[: -self.args.fedala.layer_idx],
            params_g[: -self.args.fedala.layer_idx],
        ):
            param.data = param_g.data.clone()

        # temp local model only for weight learning
        model_t = deepcopy(self.model)
        params_t = list(model_t.parameters())

        # only consider higher layers
        params_p = params[-self.args.fedala.layer_idx :]
        params_gp = params_g[-self.args.fedala.layer_idx :]
        params_tp = params_t[-self.args.fedala.layer_idx :]

        # frozen the lower layers to reduce computational cost in Pytorch
        for param in params_t[: -self.args.fedala.layer_idx]:
            param.requires_grad = False

        # initialize the weight to all ones in the beginning
        if self.weights == None:
            self.weights = [
                torch.ones_like(param.data).to(self.device) for param in params_p
            ]

        # initialize the higher layers in the temp local model
        for param_t, param, param_g, weight in zip(
            params_tp, params_p, params_gp, self.weights
        ):
            param_t.data = param + (param_g - param) * weight

        # weight learning
        losses = []  # record losses
        while True:
            for x, y in self.sampled_trainloader:
                x, y = x.to(self.device), y.to(self.device)
                model_t.zero_grad()
                logits = model_t(x)
                loss_value = self.criterion(logits, y)
                loss_value.backward()

                # update weight in this batch
                for param_t, param, param_g, weight in zip(
                    params_tp, params_p, params_gp, self.weights
                ):
                    weight.data = torch.clamp(
                        weight
                        - self.args.fedala.eta * (param_t.grad * (param_g - param)),
                        0,
                        1,
                    )

                # update temp local model in this batch
                for param_t, param, param_g, weight in zip(
                    params_tp, params_p, params_gp, self.weights
                ):
                    param_t.data = param + (param_g - param) * weight

            losses.append(loss_value.item())

            # only train one epoch in the subsequent iterations
            if not self.start_phase:
                break

            # train the weight until convergence
            if (
                len(losses) > self.args.fedala.num_pre_loss
                and np.std(losses[-self.args.fedala.num_pre_loss :])
                < self.args.fedala.threshold
            ):
                break

        self.start_phase = False

        # obtain initialized local model
        for param, param_t in zip(params_p, params_tp):
            param.data = param_t.data.clone()

        if self.args.common.buffers == "drop":
            self.model.load_state_dict(self.init_buffers, strict=False)

        if self.return_diff:
            model_params = self.model.state_dict()
            self.regular_model_params = OrderedDict(
                (key, model_params[key].clone().cpu())
                for key in self.regular_params_name
            )
