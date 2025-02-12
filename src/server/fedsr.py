from argparse import ArgumentParser, Namespace

import torch
import torch.distributions as distrib
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import DictConfig

from src.client.fedsr import FedSRClient
from src.server.fedavg import FedAvgServer
from src.utils.constants import NUM_CLASSES
from src.utils.models import MODELS, DecoupledModel


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
    algorithm_name: str = "FedSR"
    all_model_params_personalized = False  # `True` indicates that clients have their own fullset of personalized model parameters.
    return_diff = False  # `True` indicates that clients return `diff = W_global - W_local` as parameter update; `False` for `W_local` only.
    client_cls = FedSRClient

    @staticmethod
    def get_hyperparams(args_list=None) -> Namespace:
        parser = ArgumentParser()
        parser.add_argument("--L2R_coeff", type=float, default=1e-2)
        parser.add_argument("--CMI_coeff", type=float, default=5e-4)
        return parser.parse_args(args_list)

    def __init__(self, args: DictConfig):
        super().__init__(args, False, False)
        # reload the model
        self.init_model(
            model=FedSRModel(
                base_model=MODELS[self.args.model.name](
                    dataset=self.args.dataset.name,
                    pretrained=self.args.model.use_torchvision_pretrained_weights,
                ),
                dataset=self.args.dataset.name,
            )
        )
        self.init_trainer()
