from argparse import ArgumentParser, Namespace

import torch
from omegaconf import DictConfig

from src.client.fedproto import FedProtoClient
from src.server.fedavg import FedAvgServer
from src.utils.constants import NUM_CLASSES


class FedProtoServer(FedAvgServer):
    @staticmethod
    def get_hyperparams(args_list=None) -> Namespace:
        parser = ArgumentParser()
        parser.add_argument("--lamda", type=float, default=1)
        return parser.parse_args(args_list)

    def __init__(
        self,
        args: DictConfig,
        algorithm_name: str = "FedProto",
        unique_model=False,
        use_fedavg_client_cls=False,
        return_diff=False,
    ):
        super().__init__(
            args, algorithm_name, unique_model, use_fedavg_client_cls, return_diff
        )
        self.global_prototypes: dict[int, torch.Tensor] = {}
        self.init_trainer(FedProtoClient)

    def package(self, client_id: int):
        server_package = super().package(client_id)
        server_package["global_prototypes"] = self.global_prototypes
        return server_package

    def train_one_round(self):
        client_packages = self.trainer.train()
        self.aggregate_prototypes(
            [package["prototypes"] for package in client_packages.values()]
        )

    def aggregate_prototypes(
        self, client_prototypes_list: list[dict[int, torch.Tensor]]
    ):
        self.global_prototypes = {}
        for i in range(NUM_CLASSES[self.args.dataset.name]):
            size = 0
            prototypes = torch.zeros(self.model.classifier.in_features)
            for client_prototypes in client_prototypes_list:
                if i in client_prototypes.keys():
                    prototypes += client_prototypes[i]
                    size += 1

            if size > 0:
                self.global_prototypes[i] = prototypes / size
