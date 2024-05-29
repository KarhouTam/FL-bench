from argparse import ArgumentParser, Namespace

import torch
from torchvision import transforms
from torch.utils.data import DataLoader

from src.server.fedavg import FedAvgServer
from src.client.fedmd import FedMDClient
from src.utils.tools import NestedNamespace
from src.utils.constants import DATA_MEAN, DATA_STD, FLBENCH_ROOT
from data.utils.datasets import DATASETS


def get_fedmd_args(args_list=None) -> Namespace:
    parser = ArgumentParser()
    parser.add_argument("--digest_epoch", type=int, default=1)
    parser.add_argument("--public_dataset", type=str, default="mnist")
    parser.add_argument("--public_batch_size", type=int, default=32)
    parser.add_argument("--public_batch_num", type=int, default=5)
    return parser.parse_args(args_list)


class FedMDServer(FedAvgServer):
    """
    NOTE:
    FedMD supposes to be a pFL method with heterogeneous models, but this benchmark does not support heterogeneous model settings (for now).
    As a compromise, the homogeneous model version is offered.

    According to the paper, we can run experiment in 2 settings:
    1. (public: MNIST, private: FEMNIST / EMNIST);
    2. (public: CIFAR10, private: CIFAR100 but under 20 superclasses)
    """

    def __init__(
        self,
        args: NestedNamespace,
        algo: str = "FedMD",
        unique_model=True,
        use_fedavg_client_cls=False,
        return_diff=False,
    ):
        if args.fedmd.public_dataset == "mnist" and args.common.dataset not in [
            "femnist",
            "emnist",
        ]:
            raise NotImplementedError(
                "The public dataset is mnist and the --dataset should be in [femnist, emnist] (now: {})".format(
                    args.common.dataset
                )
            )
        elif (
            args.fedmd.public_dataset == "cifar10" and args.common.dataset != "cifar100"
        ):
            raise NotImplementedError(
                "The public dataset is cifar10 and the dataset should be cifar100 (now: {})".format(
                    args.common.dataset
                )
            )
        super().__init__(args, algo, unique_model, use_fedavg_client_cls, return_diff)
        test_data_transform = transforms.Compose(
            [
                transforms.Normalize(
                    DATA_MEAN[self.args.common.dataset],
                    DATA_STD[self.args.common.dataset],
                )
            ]
            if self.args.common.dataset in DATA_MEAN
            and self.args.common.dataset in DATA_STD
            else []
        )
        test_target_transform = transforms.Compose([])
        train_data_transform = transforms.Compose(
            [
                transforms.Normalize(
                    DATA_MEAN[self.args.common.dataset],
                    DATA_STD[self.args.common.dataset],
                )
            ]
            if self.args.common.dataset in DATA_MEAN
            and self.args.common.dataset in DATA_STD
            else []
        )
        train_target_transform = transforms.Compose([])

        self.public_dataset = DATASETS[self.args.fedmd.public_dataset](
            root=FLBENCH_ROOT / "data" / self.args.fedmd.public_dataset,
            args=None,
            train_data_transform=train_data_transform,
            train_target_transform=train_target_transform,
            test_data_transform=test_data_transform,
            test_target_transform=test_target_transform,
        )
        self.public_dataset_loader = DataLoader(
            self.public_dataset, self.args.fedmd.public_batch_size, shuffle=True
        )
        self.iter_public_loader = iter(self.public_dataset_loader)
        self.public_data: list[torch.Tensor] = []
        self.consensus: list[torch.Tensor] = []

        self.init_trainer(FedMDClient)

    def load_public_data_batches(self):
        for _ in range(self.args.fedmd.public_batch_num):
            try:
                x, _ = next(self.iter_public_loader)
                if len(x) <= 1:
                    x, _ = next(self.iter_public_loader)
            except StopIteration:
                self.iter_public_loader = iter(self.public_dataset_loader)
                x, _ = next(self.iter_public_loader)
            self.public_data.append(x)

    @torch.no_grad()
    def get_scores(self, client_id):
        self.client_id = client_id
        self.model.load_state_dict(self.clients_personal_model_params[client_id])
        self.model.eval()
        return [self.model(x).clone() for x in self.public_data]

    def package(self, client_id: int):
        server_package = super().package(client_id)
        server_package["consensus"] = self.consensus
        server_package["public_data"] = self.public_data
        return server_package

    def train_one_round(self):
        self.load_public_data_batches()
        batches_scores = []
        for client_id in self.selected_clients:
            batches_scores.append(self.get_scores(client_id))
        self.compute_consensus(batches_scores)
        self.trainer.train()

    def compute_consensus(
        self, batches_scores: list[torch.Tensor]
    ) -> list[torch.Tensor]:
        self.consensus = []
        for scores in zip(*batches_scores):
            self.consensus.append(torch.stack(scores, dim=-1).mean(dim=-1).cpu())
