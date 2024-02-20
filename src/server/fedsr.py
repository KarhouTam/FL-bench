import os
from argparse import ArgumentParser, Namespace
from collections import OrderedDict
from copy import deepcopy

import torch
import torch.distributions as distrib
import torch.nn as nn
import torch.nn.functional as F


from fedavg import FedAvgServer, get_fedavg_argparser
from src.client.fedsr import FedSRClient
from src.utils.models import DecoupledModel
from src.utils.constants import NUM_CLASSES
from src.utils.tools import trainable_params


def get_fedsr_argparser() -> ArgumentParser:
    parser = get_fedavg_argparser()
    parser.add_argument("--L2R_coeff", type=float, default=1e-2)
    parser.add_argument("--CMI_coeff", type=float, default=5e-4)
    return parser


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
    def __init__(
        self,
        algo: str = "FedSR",
        args: Namespace = None,
        unique_model=False,
        default_trainer=False,
    ):
        if args is None:
            args = get_fedsr_argparser().parse_args()
        super().__init__(algo, args, unique_model, default_trainer)

        # reload the model
        self.model = FedSRModel(self.model, self.args.dataset).to(self.device)
        self.model.check_avaliability()
        init_params, self.trainable_params_name = trainable_params(
            self.model, detach=True, requires_name=True
        )
        self.global_params_dict = OrderedDict(
            zip(self.trainable_params_name, init_params)
        )
        if self.args.external_model_params_file and os.path.isfile(
            self.args.external_model_params_file
        ):
            self.global_params_dict = torch.load(
                self.args.external_model_params_file, map_location=self.device
            )

        self.trainer = FedSRClient(
            deepcopy(self.model), self.args, self.logger, self.device
        )


if __name__ == "__main__":
    server = FedSRServer()
    server.run()
