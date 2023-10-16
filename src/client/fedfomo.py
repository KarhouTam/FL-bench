from collections import OrderedDict
from copy import deepcopy
from typing import Dict, List

import torch
from torch.utils.data import DataLoader, Subset

from fedavg import FedAvgClient
from src.config.utils import trainable_params, evalutate_model, vectorize


class FedFomoClient(FedAvgClient):
    def __init__(self, model, args, logger, device, client_num):
        super().__init__(model, args, logger, device)
        self.received_params = {}
        self.eval_model = deepcopy(self.model)
        self.weight_vector = torch.zeros(client_num, device=self.device)
        self.trainable_params_name = trainable_params(self.model, requires_name=True)[1]
        self.valset = Subset(self.dataset, indices=[])
        self.valloader: DataLoader = None

    def train(
        self,
        client_id: int,
        local_epoch: int,
        received_params: Dict[int, List[torch.Tensor]],
        verbose=False,
    ):
        self.client_id = client_id
        self.local_epoch = local_epoch
        self.load_dataset()
        self.set_parameters(received_params)
        stats = self.train_and_log(verbose=verbose)
        return (
            trainable_params(self.model, detach=True),
            self.weight_vector.clone(),
            stats,
        )

    def load_dataset(self):
        super().load_dataset()
        num_val_samples = int(len(self.trainset) * self.args.valset_ratio)
        self.valset.indices = self.trainset.indices[:num_val_samples]
        self.trainset.indices = self.trainset.indices[num_val_samples:]
        self.valloader = DataLoader(self.valset, 32, shuffle=True)

    def set_parameters(self, received_params: Dict[int, List[torch.Tensor]]):
        local_params_dict = OrderedDict(
            zip(self.trainable_params_name, received_params[self.client_id])
        )
        personal_params_dict = (
            self.init_personal_params_dict
            if self.client_id not in self.personal_params_dict.keys()
            else self.personal_params_dict[self.client_id]
        )
        self.eval_model.load_state_dict(local_params_dict, strict=False)
        self.eval_model.load_state_dict(personal_params_dict, strict=False)
        LOSS = evalutate_model(
            model=self.eval_model,
            dataloader=self.valloader,
            criterion=self.criterion,
            device=self.device,
        )[0]
        LOSS /= len(self.valset)
        W = torch.zeros(len(received_params), device=self.device)
        self.weight_vector.zero_()
        with torch.no_grad():
            vectorized_self_params = vectorize(received_params[self.client_id])
            for i, (client_id, params_i) in enumerate(received_params.items()):
                self.eval_model.load_state_dict(
                    OrderedDict(zip(self.trainable_params_name, params_i)), strict=False
                )
                loss = evalutate_model(
                    model=self.eval_model,
                    dataloader=self.valloader,
                    criterion=self.criterion,
                    device=self.device,
                )[0]
                loss /= len(self.valset)
                params_diff = vectorize(params_i) - vectorized_self_params
                w = (LOSS - loss) / (torch.norm(params_diff) + 1e-5)
                W[i] = w
                self.weight_vector[client_id] = w

        # compute the weight for params aggregation
        W.clip_(min=0)
        weight_sum = W.sum()
        if weight_sum > 0:
            W /= weight_sum
            for params, key in zip(
                zip(*list(received_params.values())), self.trainable_params_name
            ):
                local_params_dict[key] = torch.sum(
                    torch.stack(params, dim=-1) * W, dim=-1
                )
        super().set_parameters(local_params_dict)
