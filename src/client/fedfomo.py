import random
from copy import deepcopy
from typing import Any

import torch

from src.client.fedavg import FedAvgClient
from src.utils.tools import trainable_params, evalutate_model, vectorize


class FedFomoClient(FedAvgClient):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.eval_model = deepcopy(self.model).to(self.device)
        self.personal_params_name = list(self.model.state_dict().keys())
        self.clients_weight = {}

    def package(self):
        client_package = super().package()
        client_package.pop("regular_model_params")
        client_package["clients_weight"] = self.clients_weight
        return client_package

    def load_data_indices(self):
        super().load_data_indices()
        train_data_indices = deepcopy(self.trainset.indices)
        num_val_samples = int(len(train_data_indices) * self.args.fedfomo.valset_ratio)
        random.shuffle(train_data_indices)
        self.valset.indices = train_data_indices[:num_val_samples]
        self.trainset.indices = train_data_indices[num_val_samples:]

    @torch.no_grad
    def set_parameters(self, package: dict[str, Any]):
        self.client_id = package["client_id"]
        self.local_epoch = package["local_epoch"]
        self.load_data_indices()

        if package["optimizer_state"]:
            self.optimizer.load_state_dict(package["optimizer_state"])
        else:
            self.optimizer = self.optimizer_cls(params=trainable_params(self.model))

        if package["lr_scheduler_state"]:
            self.lr_scheduler.load_state_dict(package["lr_scheduler_state"])
        elif self.lr_scheduler_cls is not None:
            self.lr_scheduler = self.lr_scheduler_cls(optimizer=self.optimizer)

        self.model.load_state_dict(
            package["model_params_from_selected_clients"][self.client_id]
        )
        self.eval_model.load_state_dict(
            package["model_params_from_selected_clients"][self.client_id]
        )
        vectorized_self_params = vectorize(self.eval_model)

        LOSS = evalutate_model(
            model=self.eval_model,
            dataloader=self.valloader,
            criterion=self.criterion,
            device=self.device,
        ).loss
        W = torch.zeros(
            len(package["model_params_from_selected_clients"]), device=self.device
        )
        self.clients_weight = {}
        for i, (client_id, params_i) in enumerate(
            package["model_params_from_selected_clients"].items()
        ):
            self.eval_model.load_state_dict(params_i, strict=False)
            loss = evalutate_model(
                model=self.eval_model,
                dataloader=self.valloader,
                criterion=self.criterion,
                device=self.device,
            ).loss
            params_diff = vectorize(self.eval_model) - vectorized_self_params
            w = (LOSS - loss) / (torch.norm(params_diff) + 1e-5)
            W[i] = w
            self.clients_weight[client_id] = w

        # compute the weight for params aggregation
        W.clip_(min=0)
        weight_sum = W.sum()
        if weight_sum > 0:
            _, trainable_params_name = trainable_params(
                self.eval_model, requires_name=True
            )
            aggregated_params = deepcopy(self.model.state_dict())
            W /= weight_sum
            clients_model_params = []
            for client_id, model_params in package[
                "model_params_from_selected_clients"
            ].items():
                clients_model_params.append(
                    [model_params[key] for key in trainable_params_name]
                )
            for key, params in zip(trainable_params_name, zip(*clients_model_params)):
                aggregated_params[key] = torch.sum(
                    torch.stack(params, dim=-1).to(self.device) * W, dim=-1
                )

            self.model.load_state_dict(aggregated_params)
