import random
from copy import deepcopy
from typing import Any

import torch

from src.client.fedavg import FedAvgClient
from src.utils.tools import evalutate_model, vectorize


class FedFomoClient(FedAvgClient):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.eval_model = deepcopy(self.model).to(self.device)
        self.personal_params_name = list(self.model.state_dict().keys())
        self.client_weights = {}

    def package(self):
        client_package = super().package()
        client_package.pop("regular_model_params")
        client_package["client_weights"] = self.client_weights
        return client_package

    def load_data_indices(self):
        super().load_data_indices()
        train_data_indices = deepcopy(self.trainset.indices)
        num_val_samples = int(len(train_data_indices) * self.args.fedfomo.valset_ratio)
        random.shuffle(train_data_indices)
        self.valset.indices = train_data_indices[:num_val_samples]
        self.trainset.indices = train_data_indices[num_val_samples:]

    @torch.no_grad()
    def set_parameters(self, package: dict[str, Any]):
        self.client_id = package["client_id"]
        self.local_epoch = package["local_epoch"]
        self.load_data_indices()

        if package["optimizer_state"]:
            self.optimizer.load_state_dict(package["optimizer_state"])
        else:
            self.optimizer.load_state_dict(self.init_optimizer_state)

        if package["optimizer_state"]:
            self.optimizer.load_state_dict(package["optimizer_state"])
        else:
            self.optimizer.load_state_dict(self.init_optimizer_state)

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
        self.client_weights = {}
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
            self.client_weights[client_id] = w

        # compute the weight for params aggregation
        W.clip_(min=0)
        weight_sum = W.sum()
        if weight_sum > 0:
            W /= weight_sum
            for key, param in self.model.state_dict(keep_vars=True).items():
                clients_model_params = []
                for model_params in package[
                    "model_params_from_selected_clients"
                ].values():
                    if key in model_params.keys():
                        clients_model_params.append(model_params[key])

                if len(clients_model_params) > 0:
                    aggregated = torch.sum(
                        torch.stack(clients_model_params, dim=-1).to(self.device) * W,
                        dim=-1,
                    )
                    param.data = aggregated
