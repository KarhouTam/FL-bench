from collections import OrderedDict
from typing import OrderedDict

import torch

from .fedbn import FedBNClient


class FedAPClient(FedBNClient):
    def __init__(self, model, args, logger):
        super(FedAPClient, self).__init__(model, args, logger)
        self.model.need_all_features()
        self.pretrain = False

    def get_client_local_dataset(self):
        super().get_client_local_dataset()
        num_pretrain_samples = int(self.args.pretrain_ratio * len(self.trainset))
        if self.pretrain:
            self.trainset.indices = self.trainset.indices[:num_pretrain_samples]
            self.visited_time[self.client_id] = 0
        else:
            self.trainset.indices = self.trainset.indices[num_pretrain_samples:]

    @torch.no_grad()
    def get_all_features(
        self, client_id: int, new_parameters: OrderedDict[str, torch.nn.Parameter]
    ):
        self.client_id = client_id
        self.get_client_local_dataset()
        self.set_parameters(new_parameters)
        features_list = []
        batch_size_list = []
        for x, _ in self.trainloader:
            features_list.append(self.model.get_all_features(x.to(self.device)))
            batch_size_list.append(len(x))

        self.save_state()
        return features_list, batch_size_list
