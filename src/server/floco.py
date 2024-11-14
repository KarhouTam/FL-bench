from typing import Any, Union, Literal
from argparse import ArgumentParser
from copy import deepcopy

import torch
import numpy as np
from omegaconf import DictConfig
from sklearn import decomposition

from src.utils.tools import Namespace
from src.client.floco import FlocoClient
from src.server.fedavg import FedAvgServer
from src.utils.constants import NUM_CLASSES
from src.utils.models import MODELS, DecoupledModel

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
            num_endpoints=self.args.floco.num_endpoints, in_features=base_model.classifier.in_features, 
            out_features=NUM_CLASSES[self.args.dataset.name], bias=True, seed=self.args.common.seed)
        self.sample_from = "simplex_center"
    def set_subregion(self, sample_from: str, subregion_parameters: tuple):
        self.sample_from = sample_from
        self.center, self.radius = subregion_parameters
    def forward(self, x):
        if self.sample_from == "simplex_center":
            sampled_alpha = np.ones(self.args.floco.num_endpoints) / np.ones(self.args.floco.num_endpoints).sum()
        else:
            sampled_alpha = sample_L1_ball(self.center, self.radius, 1)
        set_net_alpha(self.classifier, sampled_alpha)
        return self.classifier(self.base(x))

class FlocoServer(FedAvgServer):

    @staticmethod
    def get_hyperparams(args_list=None) -> Namespace:
        parser = ArgumentParser()
        parser.add_argument("--num_endpoints", type=int, default=1)  # TODO improve terminology
        parser.add_argument("--tau", type=int, default=100)  # TODO improve terminology
        parser.add_argument("--rho", type=float, default=0.1)  # TODO improve terminology
        parser.add_argument("--finetune_region", type=str, default='simplex_center')
        parser.add_argument("--evaluate_region", type=str, default='simplex_center')

        # Floco+ (only used if pers_epoch > 0)
        parser.add_argument("--pers_epoch", type=int, default=0)  # TODO improve terminology
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
        super().__init__(args, algorithm_name, unique_model, use_fedavg_client_cls, return_diff)
        self.model = SimplexModel(self.args)
        self.model.check_and_preprocess(self.args)
        self.init_trainer(FlocoClient)
        self.projected_points = None

        if self.args.floco.pers_epoch > 0:
            self.clients_personalized_model_params = {i: deepcopy(self.model.state_dict()) for i in self.train_clients}

    def train_one_round(self):
        if self.args.floco.tau == self.current_epoch:
            selected_clients = self.selected_clients  # save selected clients
            self.selected_clients = self.train_clients  # collect gradients from all clients
            client_packages = self.trainer.train()
            self.projected_points = compute_projected_points(client_packages, self.args.floco.num_endpoints, self.return_diff)
            self.selected_clients = selected_clients  # restore selected clients

        client_packages = self.trainer.train()

        if self.args.floco.pers_epoch > 0:
            for client_id in self.selected_clients:
                self.clients_personalized_model_params[client_id] = client_packages[client_id]["personalized_model_params"]

        self.aggregate(client_packages)

    def package(self, client_id: int):
        server_package = super().package(client_id)
        if self.projected_points is None:
            server_package["sample_from"] = "simplex_center" if self.testing else "simplex_uniform"
            server_package["subregion_parameters"] = None
        else:
            server_package["sample_from"] = "region_center" if self.testing else "region_uniform"
            server_package["subregion_parameters"] = (self.projected_points[client_id], self.args.floco.rho)

        if self.args.floco.pers_epoch > 0:
            server_package["personalized_model_params"] = self.clients_personalized_model_params[client_id]
        return server_package


def compute_projected_points(client_packages, num_endpoints, return_diff):
    model_grad_type = "model_params_diff" if return_diff else "regular_model_params"
    client_gradient_dict = {}
    for client_id, package in client_packages.items():
        # TODO fix
        weights = [params.cpu().numpy() for params in package[model_grad_type].values()]
        client_grads = [weights[-i].flatten() for i in range(1, num_endpoints + 1)]
        client_grads = np.concatenate(client_grads)
        client_gradient_dict[client_id] = client_grads

    # Sort results
    client_statistics = np.array([client_gradient_dict[key] for key in sorted(client_gradient_dict.keys())])

    print('Projecting gradients ... ')
    client_statistics = decomposition.PCA(n_components=num_endpoints).fit_transform(client_statistics)
    print('... finished PCA')

    # Offset z optimization
    statistics_over_z = []
    energies_over_z = []
    best_z = None
    last_log_energy = np.inf
    z_grid = np.linspace(1e-4, 1, 1000)
    for i, z in enumerate(z_grid):
        # 2. Optimized Simplex projection
        final_client_statistics = projection_simplex(client_statistics, z=z)
        print(final_client_statistics.shape)
        final_client_statistics /= final_client_statistics.sum(1).reshape(-1, 1)
        statistics_over_z.append(final_client_statistics)
        log_energy = compute_riesz_s_energy(final_client_statistics, d=2)
        if log_energy not in [-np.inf, np.inf]:
            energies_over_z.append(log_energy)
            if log_energy < last_log_energy:
                best_z = i
                last_log_energy = log_energy
    print('... finished z optimization')
    return np.array(statistics_over_z)[best_z]


def projection_simplex(V, z=1):  # TODO simplify
    """
    Projection of x onto the simplex, scaled by z:
        P(x; z) = argmin_{y >= 0, sum(y) = z} ||y - x||^2
    z: float or array
        If array, len(z) must be compatible with V
    axis: None or int
        axis=None: project V by P(V.ravel(); z)
        axis=1: project each V[i] by P(V[i]; z[i])
        axis=0: project each V[:, j] by P(V[:, j]; z[j])
    """
    n_features = V.shape[1]
    U = np.sort(V, axis=1)[:, ::-1]
    z = np.ones(len(V)) * z
    cssv = np.cumsum(U, axis=1) - z[:, np.newaxis]
    ind = np.arange(n_features) + 1
    cond = U - cssv / ind > 0
    rho = np.count_nonzero(cond, axis=1)
    theta = cssv[np.arange(len(V)), rho - 1] / rho
    return np.maximum(V - theta[:, np.newaxis], 0)


def compute_riesz_s_energy(simplex_points, d=2):
    diff = (simplex_points[:, None] - simplex_points[None, :])
    # calculate the squared euclidean from each point to another
    dist = np.sqrt((diff ** 2).sum(axis=2))
    # make sure the distance to itself does not count
    np.fill_diagonal(dist, np.inf)
    # epsilon which is the smallest distance possible to avoid an overflow during gradient calculation
    # eps = 10 ** (-320 / (d + 2))
    epsilon = 1e-4
    dist[dist < epsilon] = epsilon
    # select only upper triangular matrix to have each mutual distance once
    mutual_dist = dist[np.triu_indices(len(simplex_points), 1)]
    mutual_dist[np.argwhere(mutual_dist == 0).flatten()] = 1e-4
    # calculate the energy by summing up the squared distances
    energies = (1 / mutual_dist ** d)
    energy = energies[~np.isnan(energies)].sum()
    log_energy = -np.log(len(mutual_dist)) + np.log(energy)
    return log_energy


def sample_L1_ball(center, radius, num_samples):
    dim = len(center)
    samples = np.zeros((num_samples, dim))
    # Generate a point on the surface of the L1 unit ball
    u = np.random.uniform(-1, 1, dim)
    u = np.sign(u) * (np.abs(u) / np.sum(np.abs(u)))
    # Scale the point to fit within the radius
    r = np.random.uniform(0, radius)
    samples = center + r * u
    return samples


def set_net_alpha(net, alphas: tuple[float, ...]):
    for m in net.modules():
        if isinstance(m, SimplexLayer):
            m.set_alphas(alphas)


def seed_weights(weights: list, seed: int) -> None:
    """Seed the weights of a list of nn.Parameter objects."""
    for i, weight in enumerate(weights):
        torch.manual_seed(seed + i)
        torch.nn.init.xavier_normal_(weight)

class StandardConv(torch.nn.Conv2d):
    def __int__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def seed(self, s):
        seed_weights([self.weight], s)
        return self
    

class StandardLinear(torch.nn.Linear):
    def __int__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def seed(self, s):
        seed_weights([self.weight], s)
        return self


class SimplexLayer:
    def __init__(self, init_weight: torch.Tensor, num_endpoints: int, seed: int):
        self.num_endpoints = num_endpoints
        self._alphas = tuple([1/num_endpoints for _ in range(num_endpoints)])  # set by the train() method each round
        self._weights = torch.nn.ParameterList([_initialize_weight(init_weight, seed + i) for i in range(num_endpoints)])

    @property
    def weight(self) -> torch.nn.Parameter:
        return sum(alpha * weight for alpha, weight in zip(self._alphas, self._weights))

    def set_alphas(self, alphas: Union[tuple[float], Literal["center"]]):
        if len(alphas) == len(self._weights):
            self._alphas = alphas
        else:
            raise ValueError(f"alphas must match number of simplex endpoints ({self.num_endpoints})")


class SimplexLinear(torch.nn.Linear, SimplexLayer):
    def __init__(self, num_endpoints: int, seed: int, *args, **kwargs):
        super().__init__(*args, **kwargs)
        SimplexLayer.__init__(self, init_weight=self.weight, num_endpoints=num_endpoints, seed=seed)


class SimplexConv(torch.nn.Conv2d, SimplexLayer):
    def __init__(self, num_endpoints: int, seed: int, *args, **kwargs):
        super().__init__(*args, **kwargs)
        SimplexLayer.__init__(self, init_weight=self.weight, num_endpoints=num_endpoints, seed=seed)


def _initialize_weight(init_weight: torch.Tensor, seed: int) -> torch.nn.Parameter:
    weight = torch.nn.Parameter(torch.zeros_like(init_weight))
    torch.manual_seed(seed)
    torch.nn.init.xavier_normal_(weight)
    return weight