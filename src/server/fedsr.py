import os
from argparse import ArgumentParser, Namespace
from collections import OrderedDict

import torch
import torch.distributions as distrib
import torch.nn as nn
import torch.nn.functional as F

from src.server.fedavg import FedAvgServer
from src.client.fedsr import FedSRClient
from src.utils.models import DecoupledModel
from src.utils.constants import NUM_CLASSES, FLBENCH_ROOT
from src.utils.tools import NestedNamespace


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
        args: NestedNamespace,
        algo: str = "FedSR",
        unique_model=False,
        use_fedavg_client_cls=False,
        return_diff=False,
    ):
        super().__init__(args, algo, unique_model, use_fedavg_client_cls, return_diff)
        # reload the model
        self.model = FedSRModel(self.model, self.args.common.dataset)
        self.model.check_and_preprocess(self.args)

        _init_global_params, _init_global_params_name = [], []
        for key, param in self.model.named_parameters():
            _init_global_params.append(param.data.clone())
            _init_global_params_name.append(key)
        self.public_model_param_names = list(self.public_model_params.keys())

        model_params_file_path = str(
            (FLBENCH_ROOT / self.args.common.external_model_params_file).absolute()
        )
        if (
            os.path.isfile(model_params_file_path)
            and model_params_file_path.find(".pt") != -1
        ):
            self.public_model_params.update(
                torch.load(model_params_file_path, map_location="cpu")
            )

        self.init_trainer(FedSRClient)
