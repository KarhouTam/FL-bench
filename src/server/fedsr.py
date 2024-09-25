import os
from argparse import ArgumentParser, Namespace

import torch
import torch.distributions as distrib
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import DictConfig

from src.client.fedsr import FedSRClient
from src.server.fedavg import FedAvgServer
from src.utils.constants import FLBENCH_ROOT, NUM_CLASSES
from src.utils.models import DecoupledModel


class FedSRModel(DecoupledModel):
    # modify base model to suit FedSR
    def __init__(self, base_model: DecoupledModel, dataset) -> None:
        super().__init__()
        self.z_dim = base_model.classifier.in_features
        out_dim = 2 * self.z_dim
        self.base = base_model.base
        self.map_layer = nn.Linear(self.z_dim, out_dim)
        self.classifier = base_model.classifier
        self.r_mu = nn.Parameter(torch.zeros(NUM_CLASSES[dataset], self.z_dim))
        self.r_sigma = nn.Parameter(torch.ones(NUM_CLASSES[dataset], self.z_dim))
        self.C = nn.Parameter(torch.ones([]))

    def featurize(self, x, num_samples=1, return_dist=False):
        # designed for FedSR
        z_params = F.relu(self.map_layer(F.relu(self.base(x))))
        z_mu = z_params[:, : self.z_dim]
        z_sigma = F.softplus(z_params[:, self.z_dim :])
        z_dist = distrib.Independent(distrib.normal.Normal(z_mu, z_sigma), 1)
        z = z_dist.rsample([num_samples]).view([-1, self.z_dim])

        if return_dist:
            return z, (z_mu, z_sigma)
        else:
            return z

    def forward(self, x):
        z = self.featurize(x)
        logits = self.classifier(z)
        return logits


class FedSRServer(FedAvgServer):
    @staticmethod
    def get_hyperparams(args_list=None) -> Namespace:
        parser = ArgumentParser()
        parser.add_argument("--L2R_coeff", type=float, default=1e-2)
        parser.add_argument("--CMI_coeff", type=float, default=5e-4)
        return parser.parse_args(args_list)

    def __init__(
        self,
        args: DictConfig,
        algorithm_name: str = "FedSR",
        unique_model=False,
        use_fedavg_client_cls=False,
        return_diff=False,
    ):
        super().__init__(
            args, algorithm_name, unique_model, use_fedavg_client_cls, return_diff
        )
        # reload the model
        self.model = FedSRModel(self.model, self.args.dataset.name)
        self.model.check_and_preprocess(self.args)

        _init_global_params, _init_global_params_name = [], []
        for key, param in self.model.named_parameters():
            _init_global_params.append(param.data.clone())
            _init_global_params_name.append(key)
        self.public_model_param_names = list(self.public_model_params.keys())

        if self.args.model.external_model_weights_path is not None:
            file_path = str(
                (FLBENCH_ROOT / self.args.model.external_model_weights_path).absolute()
            )
            if os.path.isfile(file_path) and file_path.find(".pt") != -1:
                self.public_model_params.update(
                    torch.load(file_path, map_location="cpu")
                )

        self.init_trainer(FedSRClient)
