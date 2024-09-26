from argparse import ArgumentParser, Namespace
from copy import deepcopy

import torch
import torch.nn as nn
from omegaconf import DictConfig
from torch.utils.data import DataLoader, Dataset

from src.client.adcol import ADCOLClient
from src.server.fedavg import FedAvgServer


class Discriminator(nn.Module):
    # discriminator for adversarial training in ADCOL
    def __init__(self, base_model, client_num):
        super(Discriminator, self).__init__()
        try:
            in_features = base_model.classifier.in_features
        except:
            raise ValueError("base model has no classifier")
        self.discriminator = nn.Sequential(
            nn.Linear(in_features, 512, bias=False),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, 512, bias=False),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, client_num, bias=False),
        )

    def forward(self, x):
        x = self.discriminator(x)
        return x


class DiscriminateDataset(Dataset):
    def __init__(self, feature, index):
        # initiate this class
        self.feature = feature
        self.index = index

    def __getitem__(self, idx):
        single_feature = self.feature[idx]
        single_index = self.index[idx]
        return single_feature, single_index

    def __len__(self):
        return len(self.index)


class ADCOLServer(FedAvgServer):
    @staticmethod
    def get_hyperparams(args_list=None) -> Namespace:
        parser = ArgumentParser()
        parser.add_argument("--mu", type=float, default=0.5)
        parser.add_argument(
            "--dis_lr", type=float, default=0.01, help="learning rate for discriminator"
        )
        parser.add_argument(
            "--dis_epoch",
            type=int,
            default=3,
            help="epochs for trainig discriminator. larger dis_epoch is recommende when mu is large",
        )
        return parser.parse_args(args_list)

    def __init__(
        self,
        args: DictConfig,
        algorithm_name: str = "ADCOL",
        unique_model=False,
        use_fedavg_client_cls=False,
        return_diff=False,
    ):
        super().__init__(
            args, algorithm_name, unique_model, use_fedavg_client_cls, return_diff
        )
        self.train_client_num = len(self.train_clients)
        self.discriminator = Discriminator(
            base_model=self.model, client_num=len(self.train_clients)
        )
        self.init_trainer(
            ADCOLClient,
            discriminator=deepcopy(self.discriminator),
            client_num=self.client_num,
        )
        self.feature_dataloader: DataLoader = None
        self.features = {}

    def train_one_round(self):
        client_packages = self.trainer.train()

        self.features = {}
        self.feature_dataloader = None
        for cid in self.selected_clients:
            self.features[cid] = client_packages[cid]["features_list"]
        self.aggregate(client_packages)
        self.train_and_test_discriminator()

    def package(self, client_id: int):
        server_package = super().package(client_id)
        server_package["new_discriminator_params"] = deepcopy(
            self.discriminator.state_dict()
        )
        return server_package

    def train_and_test_discriminator(self):
        self.generate_client_index()
        if (self.current_epoch + 1) % self.args.common.test_interval == 0:
            acc_before = self.test_discriminator()

        self.train_discriminator()

        if (self.current_epoch + 1) % self.args.common.test_interval == 0:
            acc_after = self.test_discriminator()
            if self.verbose:
                self.logger.log(
                    f"The accuracy of discriminator: {acc_before*100 :.2f}% -> {acc_after*100 :.2f}%"
                )

        self.discriminator.cpu()

    def train_discriminator(self):
        self.discriminator.to(self.device)
        self.discriminator.train()
        self.discriminator_optimizer = torch.optim.SGD(
            self.discriminator.parameters(), lr=self.args.adcol.dis_lr
        )
        loss_func = nn.CrossEntropyLoss().to(self.device)
        # train discriminator
        for _ in range(self.args.adcol.dis_epoch):
            for x, y in self.feature_dataloader:
                x, y = x.to(self.device), y.to(self.device)
                y = y.type(torch.float32)
                y_pred = self.discriminator(x)
                loss = loss_func(y_pred, y).mean()
                self.discriminator_optimizer.zero_grad()
                loss.backward()
                self.discriminator_optimizer.step()

    def test_discriminator(self):
        # test discriminator
        self.discriminator.to(self.device)
        self.discriminator.eval()
        if self.feature_dataloader:
            self.accuracy_list = []
            for x, y in self.feature_dataloader:
                x, y = x.to(self.device), y.to(self.device)
                y_pred = self.discriminator(x)
                y_pred = torch.argmax(y_pred, dim=1)
                y = torch.argmax(y, dim=1)
                correct = torch.sum(y_pred == y).item()
                self.accuracy_list.append(correct / self.args.common.batch_size)
            accuracy = sum(self.accuracy_list) / len(self.accuracy_list)
            return accuracy

    def generate_client_index(self):
        # generate client_index_list by self.features
        client_index_list = []
        feature_list = []
        for client, feature in self.features.items():
            feature = torch.cat(feature, 0)
            index_tensor = torch.full(
                (feature.shape[0],), fill_value=client, dtype=torch.int64
            )
            client_index_list.append(index_tensor)
            feature_list.append(feature)
        orgnized_features = torch.cat(feature_list, 0)
        orgnized_client_index = torch.cat(client_index_list).type(torch.int64)
        targets = torch.zeros(
            (orgnized_client_index.shape[0], len(self.train_clients)), dtype=torch.int64
        )
        targets = targets.scatter(
            dim=1,
            index=orgnized_client_index.unsqueeze(-1),
            src=torch.ones((orgnized_client_index.shape[0], 1), dtype=torch.int64),
        ).type(torch.float32)
        discriminator_training_dataset = DiscriminateDataset(orgnized_features, targets)
        self.feature_dataloader = DataLoader(
            discriminator_training_dataset,
            batch_size=self.args.common.batch_size,
            shuffle=True,
        )
