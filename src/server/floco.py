from argparse import ArgumentParser
from collections import OrderedDict
from copy import deepcopy

import numpy as np
import torch
from omegaconf import DictConfig
from sklearn import decomposition

from src.client.floco import FlocoClient
from src.server.fedavg import FedAvgServer
from src.utils.constants import NUM_CLASSES
from src.utils.models import MODELS, DecoupledModel
from src.utils.tools import Namespace


class FlocoServer(FedAvgServer):
    @staticmethod
    def get_hyperparams(args_list=None) -> Namespace:
        parser = ArgumentParser()
        parser.add_argument("--endpoints", type=int, default=10)
        parser.add_argument("--tau", type=int, default=10)
        parser.add_argument("--rho", type=float, default=0.1)

        # Floco+ (only used if pers_epoch > 0)
        parser.add_argument("--pers_epoch", type=int, default=0)
        parser.add_argument("--lamda", type=float, default=1)

        return parser.parse_args(args_list)

    def __init__(
        self,
        args: DictConfig,
        algorithm_name: str = "Floco",
        unique_model=False,
        use_fedavg_client_cls=True,
        return_diff=False,
    ):
        super().__init__(
            args, algorithm_name, unique_model, use_fedavg_client_cls, return_diff
        )
        self.model = SimplexModel(self.args)
        self.model.check_and_preprocess(self.args)
        _init_global_params, _init_global_params_name = [], []
        for key, param in self.model.named_parameters():
            _init_global_params.append(param.data.clone())
            _init_global_params_name.append(key)
        self.public_model_param_names = _init_global_params_name
        self.public_model_params: OrderedDict[str, torch.Tensor] = OrderedDict(
            zip(_init_global_params_name, _init_global_params)
        )
        self.init_trainer(FlocoClient)
        self.projected_clients = None

        if self.args.floco.pers_epoch > 0:  # Floco+
            self.clients_personalized_model_params = {
                i: deepcopy(self.model.state_dict()) for i in self.train_clients
            }

    def train_one_round(self):
        if self.args.floco.tau == self.current_epoch:
            print("Projecting gradients ... ")
            selected_clients = self.selected_clients  # save selected clients
            self.selected_clients = self.train_clients  # collect gradients
            client_packages = self.trainer.train()
            self.projected_clients = project_clients(
                client_packages, self.args.floco.endpoints, self.return_diff
            )
            self.selected_clients = selected_clients  # restore selected clients

        client_packages = self.trainer.train()
        if self.args.floco.pers_epoch > 0:  # Floco+
            for client_id in self.selected_clients:
                self.clients_personalized_model_params[client_id] = client_packages[
                    client_id
                ]["personalized_model_params"]
        self.aggregate_client_updates(client_packages)

    def package(self, client_id: int):
        server_package = super().package(client_id)
        server_package["subregion_parameters"] = (
            None
            if self.projected_clients is None
            else (self.projected_clients[client_id], self.args.floco.rho)
        )
        if self.args.floco.pers_epoch > 0:  # Floco+
            server_package["personalized_model_params"] = (
                self.clients_personalized_model_params[client_id]
            )
        return server_package


class SimplexModel(DecoupledModel):
    def __init__(self, args) -> None:
        super().__init__()
        self.args = args
        base_model = MODELS[self.args.model.name](
            dataset=self.args.dataset.name,
            pretrained=self.args.model.use_torchvision_pretrained_weights,
        )
        self.base = base_model.base
        self.classifier = SimplexLinear(
            endpoints=self.args.floco.endpoints,
            in_features=base_model.classifier.in_features,
            out_features=NUM_CLASSES[self.args.dataset.name],
            bias=True,
            seed=self.args.common.seed,
        )
        self.subregion_parameters = None

    def forward(self, x):
        endpoints = self.args.floco.endpoints
        if self.subregion_parameters is None:  # before projection
            if self.training:  # sample uniformly from simplex for training
                sample = np.random.exponential(scale=1.0, size=endpoints)
                self.classifier.alphas = sample / sample.sum()
            else:  # use simplex center for testing
                simplex_center = tuple([1 / endpoints for _ in range(endpoints)])
                self.classifier.alphas = simplex_center
        else:  # after projection
            if self.training:  # sample uniformly from subregion for training
                self.classifier.alphas = _sample_L1_ball(*self.subregion_parameters)
            else:  # use subregion center for testing
                self.classifier.alphas = self.subregion_parameters[0]
        return super().forward(x)


class SimplexLinear(torch.nn.Linear):
    def __init__(self, endpoints: int, seed: int, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.endpoints = endpoints
        self.alphas = tuple([1 / endpoints for _ in range(endpoints)])
        self._weights = torch.nn.ParameterList(
            [_initialize_weight(self.weight, seed + i) for i in range(endpoints)]
        )

    @property
    def weight(self) -> torch.nn.Parameter:
        return sum(alpha * weight for alpha, weight in zip(self.alphas, self._weights))


def project_clients(client_packages, endpoints, return_diff):
    model_grad_type = "model_params_diff" if return_diff else "regular_model_params"
    gradient_dict = {i: None for i in range(len(client_packages))}  # init sorted dict
    for client_id, package in client_packages.items():
        gradient_dict[client_id] = np.concatenate(
            [
                v.cpu().numpy().flatten()
                for k, v in package[model_grad_type].items()
                if "classifier._weights" in k
            ]
        )
    client_stats = np.array(list(gradient_dict.values()))
    kappas = decomposition.PCA(n_components=endpoints).fit_transform(client_stats)

    # Find optimal projection
    lowest_log_energy = np.inf
    best_beta = None
    for i, z in enumerate(np.linspace(1e-4, 1, 1000)):
        betas = _project_client_onto_simplex(kappas, z=z)
        betas /= betas.sum(axis=1, keepdims=True)
        log_energy = _riesz_s_energy(betas)
        if log_energy not in [-np.inf, np.inf] and log_energy < lowest_log_energy:
            lowest_log_energy = log_energy
            best_beta = betas
    return best_beta


def _project_client_onto_simplex(kappas, z):
    sorted_kappas = np.sort(kappas, axis=1)[:, ::-1]
    z = np.ones(len(kappas)) * z
    cssv = np.cumsum(sorted_kappas, axis=1) - z[:, np.newaxis]
    ind = np.arange(kappas.shape[1]) + 1
    cond = sorted_kappas - cssv / ind > 0
    nonzero = np.count_nonzero(cond, axis=1)
    normalized_kappas = cssv[np.arange(len(kappas)), nonzero - 1] / nonzero
    betas = np.maximum(kappas - normalized_kappas[:, np.newaxis], 0)
    return betas


def _riesz_s_energy(simplex_points):
    diff = simplex_points[:, None] - simplex_points[None, :]
    distance = np.sqrt((diff**2).sum(axis=2))
    np.fill_diagonal(distance, np.inf)
    epsilon = 1e-4  # epsilon is the smallest distance possible to avoid overflow during gradient calculation
    distance[distance < epsilon] = epsilon
    # select only upper triangular matrix to have each mutual distance once
    mutual_dist = distance[np.triu_indices(len(simplex_points), 1)]
    mutual_dist[np.argwhere(mutual_dist == 0).flatten()] = epsilon
    energies = 1 / mutual_dist**2
    energy = energies[~np.isnan(energies)].sum()
    log_energy = -np.log(len(mutual_dist)) + np.log(energy)
    return log_energy


def _sample_L1_ball(center, radius):
    u = np.random.uniform(-1, 1, len(center))
    u = np.sign(u) * (np.abs(u) / np.sum(np.abs(u)))
    return center + np.random.uniform(0, radius) * u


def _initialize_weight(init_weight: torch.Tensor, seed: int) -> torch.nn.Parameter:
    weight = torch.nn.Parameter(torch.zeros_like(init_weight))
    torch.manual_seed(seed)
    torch.nn.init.xavier_normal_(weight)
    return weight
