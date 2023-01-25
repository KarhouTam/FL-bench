from collections import OrderedDict
from copy import deepcopy
from typing import Dict, List, OrderedDict

import torch
from torch.utils.data import DataLoader, Subset

from .fedavg import FedAvgClient
from config.utils import trainable_params


class FedFomoClient(FedAvgClient):
    def __init__(self, model, args, client_num_in_total, logger):
        super().__init__(model, args, logger)
        self.received_params = {}
        self.eval_model = deepcopy(self.model)
        self.weight_vector = torch.zeros(client_num_in_total, device=self.device)
        self.trainable_params_name = trainable_params(self.model, requires_name=True)[0]
        self.valset: Subset = None
        self.valloader: DataLoader = None

    def train(
        self,
        client_id: int,
        received_params: Dict[int, List[torch.Tensor]],
        evaluate=True,
        verbose=False,
    ):
        self.client_id = client_id
        self.get_client_local_dataset()
        self.set_parameters(received_params)
        stats = self.log_while_training(evaluate, verbose)
        return (
            deepcopy(trainable_params(self.model)),
            deepcopy(self.weight_vector),
            stats,
        )

    def get_client_local_dataset(self):
        super().get_client_local_dataset()
        self.valset = deepcopy(self.trainset)
        num_val_samples = int(len(self.trainset) * self.args.valset_ratio)
        self.valset.indices = self.trainset.indices[:num_val_samples]
        self.trainset.indices = self.trainset.indices[num_val_samples:]
        self.valloader = DataLoader(self.valset, 32, shuffle=True)

    def set_parameters(self, received_params: Dict[int, List[torch.Tensor]]):
        received_params = {
            i: [p.to(self.device) for p in params]
            for i, params in received_params.items()
        }
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
        LOSS, _, num_val_samples = self.evaluate(self.eval_model, self.valloader)
        LOSS /= num_val_samples
        W = []
        self.weight_vector.zero_()
        with torch.no_grad():
            for i, params_i in received_params.items():
                self.eval_model.load_state_dict(
                    OrderedDict(zip(self.trainable_params_name, params_i)), strict=False
                )
                loss = self.evaluate(self.eval_model, self.valloader)[0]
                loss /= num_val_samples
                params_diff = []
                for p_new, p_old in zip(params_i, received_params[self.client_id]):
                    params_diff.append((p_new - p_old).flatten())
                params_diff = torch.cat(params_diff)
                w = (LOSS - loss) / (torch.norm(params_diff) + 1e-5)
                W.append(w)
                self.weight_vector[i] = w

        # compute the weight for params aggregation
        W = torch.maximum(torch.tensor(W), torch.tensor(0)).to(self.device)
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

    @torch.no_grad()
    def evaluate(self, specified_model=None, specified_datalaoder=None):
        model = self.model if specified_model is None else specified_model
        dataloader = (
            self.testloader if specified_datalaoder is None else specified_datalaoder
        )
        criterion = torch.nn.CrossEntropyLoss(reduction="sum")
        model.eval()
        loss = 0
        correct = 0
        for x, y in dataloader:
            x, y = x.to(self.device), y.to(self.device)
            logits = self.model(x)
            loss += criterion(logits, y)
            pred = torch.argmax(logits, -1)
            correct += (pred == y).int().sum()
        return (loss.item(), correct.item(), len(dataloader.dataset))
