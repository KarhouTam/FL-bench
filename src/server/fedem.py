from argparse import ArgumentParser, Namespace
from collections import OrderedDict
from copy import deepcopy
from typing import Any, Dict

import torch
from omegaconf import DictConfig

from src.client.fedem import EnsembleModel, FedEMClient
from src.server.fedavg import FedAvgServer
from src.utils.models import MODELS


class FedEMServer(FedAvgServer):
    @staticmethod
    def get_hyperparams(args_list=None) -> Namespace:
        parser = ArgumentParser()
        parser.add_argument("--n_learners", type=int, default=3)
        return parser.parse_args(args_list)

    def __init__(
        self,
        args: DictConfig,
        algorithm_name: str = "FedEM",
        unique_model=False,
        use_fedavg_client_cls=False,
        return_diff=False,
    ):
        if args.common.test.server.interval <= args.common.global_epoch:
            raise NotImplementedError(
                "FedEM is not supported for global model testing on server side."
            )
        super().__init__(
            args, algorithm_name, unique_model, use_fedavg_client_cls, return_diff
        )
        self.init_model(
            model=EnsembleModel(
                learners=[
                    MODELS[self.args.model.name](
                        dataset=self.args.dataset.name,
                        pretrained=self.args.model.use_torchvision_pretrained_weights,
                    )
                    for _ in range(self.args.fedem.n_learners)
                ],
                learners_weights=torch.ones(self.args.fedem.n_learners)
                / self.args.fedem.n_learners,
            )
        )

        self.model.check_and_preprocess(self.args)
        self.client_data_sample_weights = {i: None for i in range(self.client_num)}
        self.client_learner_weights = {i: None for i in range(self.client_num)}
        self.init_trainer(FedEMClient)

    def package(self, client_id):
        server_package = super().package(client_id)
        server_package["data_sample_weights"] = self.client_data_sample_weights[
            client_id
        ]
        server_package["learner_weights"] = self.client_learner_weights[client_id]
        return server_package
