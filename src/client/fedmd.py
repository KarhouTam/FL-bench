from copy import deepcopy
from typing import List

import torch
from torchvision.transforms import Compose, Normalize
from torch.utils.data import DataLoader

from fedavg import FedAvgClient, _PROJECT_DIR
from data.utils.datasets import DATASETS
from data.utils.constants import MEAN, STD
from src.config.utils import trainable_params


class FedMDClient(FedAvgClient):
    def __init__(self, model, args, logger):
        super(FedMDClient, self).__init__(model, args, logger)

        transform = Compose(
            [Normalize(MEAN[self.args.public_dataset], STD[self.args.public_dataset])]
        )
        # transform = None
        target_transform = None

        self.public_dataset = DATASETS[self.args.public_dataset](
            root=_PROJECT_DIR / "data" / args.public_dataset,
            args=None,
            transform=transform,
            target_transform=target_transform,
        )
        self.public_dataset_loader = DataLoader(
            self.public_dataset, self.args.public_batch_size, shuffle=True
        )
        self.iter_public_loader = iter(self.public_dataset_loader)
        self.public_data: List[torch.Tensor] = []
        self.public_targets: List[torch.Tensor] = []
        self.consensus: List[torch.Tensor] = []
        self.mse_criterion = torch.nn.MSELoss()

    def load_public_data_batches(self):
        for _ in range(self.args.public_batch_num):
            try:
                x, y = next(self.iter_public_loader)
                if len(x) <= 1:
                    x, y = next(self.iter_public_loader)

            except StopIteration:
                self.iter_public_loader = iter(self.public_dataset_loader)
                x, y = next(self.iter_public_loader)
            self.public_data.append(x.to(self.device))
            self.public_targets.append(y.to(self.device))

    def train(self, client_id, new_parameters, verbose):
        self.client_id = client_id
        self.load_dataset()
        self.set_parameters(new_parameters)
        self.digest()
        res = self.train_and_log(verbose=verbose)
        return deepcopy(trainable_params(self.model)), res

    @torch.no_grad()
    def get_scores(self, client_id, new_parameters):
        self.client_id = client_id
        self.set_parameters(new_parameters)
        self.model.eval()
        return [self.model(x).clone() for x in self.public_data]

    def digest(self):
        self.model.train()
        for _ in range(self.args.digest_epoch):
            for i in range(self.args.public_batch_num):
                logit = self.model(self.public_data[i])
                loss = self.mse_criterion(logit, self.consensus[i])
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
