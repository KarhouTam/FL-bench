from argparse import ArgumentParser
from copy import deepcopy

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from fedavg import FedAvgServer, get_fedavg_argparser
from src.client.adcol import ADCOLClient


def get_adcol_argparser() -> ArgumentParser:
    parser = get_fedavg_argparser()
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
    return parser


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
    def __init__(
        self, algo: str = "ADCOL", args=None, unique_model=False, default_trainer=False
    ):
        if args is None:
            args = get_adcol_argparser().parse_args()
        super().__init__(algo, args, unique_model, default_trainer)
        self.train_client_num = len(self.train_clients)
        self.discriminator = Discriminator(
            base_model=self.model, client_num=len(self.train_clients)
        ).to(self.device)
        self.trainer = ADCOLClient(
            deepcopy(self.model),
            deepcopy(self.discriminator),
            self.args,
            self.logger,
            self.device,
            self.client_num,
        )
        self.feature_dataloader = None

    def train_one_round(self):
        delta_cache = []
        weight_cache = []
        self.features = {}
        self.feature_dataloader: DataLoader = None
        for client_id in self.selected_clients:
            client_local_params = self.generate_client_params(client_id)
            (
                delta,
                weight,
                self.client_metrics[client_id][self.current_epoch],
                self.features[client_id],
            ) = self.trainer.train(
                client_id=client_id,
                local_epoch=self.clients_local_epoch[client_id],
                new_parameters=client_local_params,
                new_discriminator_parameters=self.discriminator.state_dict(),
                verbose=((self.current_epoch + 1) % self.args.verbose_gap) == 0,
            )

            delta_cache.append(delta)
            weight_cache.append(weight)

        self.aggregate(delta_cache, weight_cache)
        self.train_and_test_discriminator()

    def train_and_test_discriminator(self):
        self.generate_client_index()
        if (self.current_epoch + 1) % self.args.test_gap == 0:
            acc_before = self.test_discriminator()

        self.train_discriminator()

        if (self.current_epoch + 1) % self.args.test_gap == 0:
            acc_after = self.test_discriminator()
            if (self.current_epoch + 1) % self.args.verbose_gap == 0:
                self.logger.log(
                    f"The accuracy of discriminator: {acc_before*100 :.2f}% -> {acc_after*100 :.2f}%"
                )

    def train_discriminator(self):
        self.discriminator.train()
        self.discriminator_optimizer = torch.optim.SGD(
            self.discriminator.parameters(), lr=self.args.dis_lr
        )
        loss_func = nn.CrossEntropyLoss().to(self.device)
        # train discriminator
        for _ in range(self.args.dis_epoch):
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
        if self.feature_dataloader:
            self.discriminator.eval()
            self.accuracy_list = []
            for x, y in self.feature_dataloader:
                x, y = x.to(self.device), y.to(self.device)
                y_pred = self.discriminator(x)
                y_pred = torch.argmax(y_pred, dim=1)
                y = torch.argmax(y, dim=1)
                correct = torch.sum(y_pred == y).item()
                self.accuracy_list.append(correct / self.args.batch_size)
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
            batch_size=self.args.batch_size,
            shuffle=True,
        )


if __name__ == "__main__":
    server = ADCOLServer()
    server.run()
