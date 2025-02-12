import math
from argparse import ArgumentParser, Namespace
from typing import Any

import torch

from src.client.flute import FLUTEClient
from src.server.fedavg import FedAvgServer
from src.utils.constants import NUM_CLASSES


class FLUTEServer(FedAvgServer):
    algorithm_name: str = "FLUTE"
    all_model_params_personalized = False  # `True` indicates that clients have their own fullset of personalized model parameters.
    return_diff = False  # `True` indicates that clients return `diff = W_global - W_local` as parameter update; `False` for `W_local` only.
    client_cls = FLUTEClient

    @staticmethod
    def get_hyperparams(args_list=None) -> Namespace:
        parser = ArgumentParser()
        parser.add_argument("--rep_round", type=int, default=1)
        parser.add_argument("--lamda1", type=float, default=0.25)
        parser.add_argument("--lamda2", type=float, default=0.0025)
        parser.add_argument("--lamda3", type=float, default=0.0005)
        parser.add_argument("--gamma1", type=float, default=1)
        parser.add_argument("--gamma2", type=float, default=1)
        parser.add_argument("--nc2_lr", type=float, default=0.5)
        return parser.parse_args(args_list)

    def train_one_round(self):
        clients_package = self.trainer.train()
        self.aggregate_client_updates(clients_package)
        self.update_neural_collapse(clients_package)

    def update_neural_collapse(self, client_packages: dict[int, dict[str, Any]]):
        num_classes = NUM_CLASSES[self.args.dataset.name]

        classifier_weights = torch.stack(
            [
                package["regular_model_params"]["classifier.weight"]
                for package in client_packages.values()
            ],
            dim=0,
        ).requires_grad_(True)

        optimizer = torch.optim.SGD([classifier_weights], lr=self.args.flute.nc2_lr)
        loss = 0

        for i, client_id in enumerate(self.selected_clients):
            labels = client_packages[client_id]["data_labels"]

            weight_i = classifier_weights[i]
            weight_i_matmul = weight_i @ weight_i.t()
            norm = torch.norm(weight_i_matmul)
            loss += torch.norm(
                weight_i_matmul / norm
                - torch.mul(
                    (
                        torch.eye(num_classes)
                        - torch.ones((num_classes, num_classes)) / num_classes
                    ),
                    labels @ labels.t(),
                )
                / math.sqrt(num_classes - 1)
            ) / len(self.selected_clients)

        loss.backward()
        optimizer.step()

        for i, client_id in enumerate(self.selected_clients):
            self.clients_personal_model_params[client_id]["classifier.weight"] = (
                classifier_weights[i].detach().clone()
            )
