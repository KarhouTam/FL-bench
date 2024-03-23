from argparse import Namespace
from copy import deepcopy
from typing import Dict, List

import torch

from fedavg import FedAvgServer, get_fedavg_argparser
from src.client.fedproto import FedProtoClient
from src.utils.constants import NUM_CLASSES


def get_fedproto_argparser():
    parser = get_fedavg_argparser()
    parser.add_argument("--lamda", type=float, default=1)
    return parser


class FedProtoServer(FedAvgServer):
    def __init__(
        self,
        algo: str = "FedProto",
        args: Namespace = None,
        unique_model=False,
        default_trainer=False,
    ):
        if args is None:
            args = get_fedproto_argparser().parse_args()
        super().__init__(algo, args, unique_model, default_trainer)
        self.trainer = FedProtoClient(
            deepcopy(self.model), self.args, self.logger, self.device
        )
        self.global_prototypes: Dict[int, torch.Tensor] = {}

    def train_one_round(self):
        client_prototypes = []
        for client_id in self.selected_clients:
            (prototypes, self.client_metrics[client_id][self.current_epoch]) = (
                self.trainer.train(
                    client_id=client_id,
                    local_epoch=self.clients_local_epoch[client_id],
                    global_prototypes=self.global_prototypes,
                    verbose=((self.current_epoch + 1) % self.args.verbose_gap) == 0,
                )
            )

            client_prototypes.append(prototypes)

        self.aggregate_prototypes(client_prototypes)

    def aggregate_prototypes(
        self, client_prototypes_list: List[Dict[int, torch.Tensor]]
    ):
        self.global_prototypes = {}
        for i in range(NUM_CLASSES[self.args.dataset]):
            size = 0
            prototypes = torch.zeros(
                self.model.classifier.in_features, device=self.device
            )
            for client_prototypes in client_prototypes_list:
                if i in client_prototypes.keys():
                    prototypes += client_prototypes[i]
                    size += 1

            if size > 0:
                self.global_prototypes[i] = prototypes / size


if __name__ == "__main__":
    server = FedProtoServer()
    server.run()
