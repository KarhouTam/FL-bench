import torch
import numpy as np
from typing import Union, Literal

import torch
import torch.nn as nn
from src.utils.models import MODELS
from src.utils.models import DecoupledModel
from src.utils.constants import NUM_CLASSES

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
    def forward(self, x, center, radius):
        sampled_alpha = sample_L1_ball(center, radius, 1)
        set_net_alpha(self.classifier, sampled_alpha)
        return self.classifier(self.base(x))


def map_simplex_to_cartesian(simplex_point):
    """
    Map a point from standard 2-simplex with vertices: (0,0,1), (0,1,0), (1,0,0),
    to cartesian triangle: triangle(ABC), A=(0,0), B=(1,0), C=(0,1)
    """
    return simplex_point[1:]


# A class that defines a region in the solution simplex with a center, and coordinate mappings between the coordinates simplex <-> cartesian
class SimplexRegion:
    """
    Simplex Region/Clustering Region class. Is defined through its cluster center in both simplex & cartesian coordinates. Can sample from its region given certain rho parameter.
    """

    def __init__(
        self,
        region_id,
        center,
        rho
    ):
        self.region_id = region_id
        self.center_simplex = center
        self.center_cartesian = map_simplex_to_cartesian(self.center_simplex)
        # Radius of the ball to sample around from
        self.rho = rho

        # Presample alphas to accelerate training
        sampled_alphas = sample_L1_ball(self.center_simplex, self.rho, 100000)
        # Normalization, IMPORTANT!
        self.sampled_alphas = sampled_alphas / sampled_alphas.sum(axis=1).reshape(-1, 1)
        # Append center
        self.sampled_alphas = np.concatenate([self.sampled_alphas, [self.center_simplex]])
        self.alphas_cartesian = [map_simplex_to_cartesian(alpha) for alpha in self.sampled_alphas]

    def get_client_subregion(self):
        return self.sampled_alphas, self.alphas_cartesian

    def get_region_center_simplex(self):
        return self.center_simplex

    def get_region_center_cartesian(self):
        return self.center_cartesian

class SolutionSimplex:
    """
    Solution simplex that keeps track of all simplex regions.
    """

    def __init__(self, args):
        self.args = args
        self.rho = self.args.floco.rho
        self.num_endpoints = self.args.floco.num_endpoints

    def set_solution_simplex_regions(
        self, projected_points, rho
    ):  

        self.simplex_regions = _compute_solution_simplex_(
            projected_points=projected_points,
            rho=self.rho 
        )
        self.rho = rho
        self.client_to_simplex_region_mapping = {}
        for i, simplex_region in enumerate(self.simplex_regions):
            self.client_to_simplex_region_mapping[i] = simplex_region.region_id
    
    def get_client_subregion(self, client_id):
        # Get correct simplex region
        sampled_region_id = self.client_to_simplex_region_mapping[client_id]
        client_simplex_region = self.simplex_regions[sampled_region_id]
        sampled_alpha_simplex, sampled_alpha_cartesian = client_simplex_region.get_client_subregion()
        return sampled_alpha_simplex, sampled_alpha_cartesian, sampled_region_id

    def sample_uniform(self, client_id):
        # Get correct simplex region
        sampled_region_id = client_id
        alpha = np.random.exponential(scale=1.0, size=(100000, self.num_endpoints))
        sampled_alpha_simplex = alpha / alpha.sum(1).reshape(-1, 1)
        simplex_center = np.ones(self.num_endpoints) / np.ones(self.num_endpoints).sum()
        sampled_alpha_simplex = np.concatenate([sampled_alpha_simplex, [simplex_center]])
        sampled_alpha_cartesian = [map_simplex_to_cartesian(alpha) for alpha in sampled_alpha_simplex]
        return sampled_alpha_simplex, sampled_alpha_cartesian, sampled_region_id

    def get_client_center(self, client_id):
        sampled_region_id = self.client_to_simplex_region_mapping[client_id]
        alpha_simplex = self.simplex_regions[sampled_region_id].get_region_center_simplex()
        alpha_cartesian = self.simplex_regions[sampled_region_id].get_region_center_cartesian()
        return [alpha_simplex], alpha_cartesian, sampled_region_id

    def get_simplex_region_centers_cartesian(self):
        return [simplex_region.get_region_center_cartesian() for simplex_region in self.simplex_regions]

    def get_simplex_region_centers_simplex(self):
        return [simplex_region.get_region_center_simplex() for simplex_region in self.simplex_regions]

def _compute_solution_simplex_(projected_points, rho):
    simplex_regions = []
    for i, tmp_center in enumerate(projected_points):
        simplex_region = SimplexRegion(
            region_id=i,
            center=tmp_center,
            rho=rho
        )
        simplex_regions.append(simplex_region)
    return simplex_regions

def sample_L1_ball(center, radius, num_samples):
    dim = len(center)
    samples = np.zeros((num_samples, dim))
    for i in range(num_samples):
        # Generate a point on the surface of the L1 unit ball
        u = np.random.uniform(-1, 1, dim)
        u = np.sign(u) * (np.abs(u) / np.sum(np.abs(u)))
        # Scale the point to fit within the radius
        r = np.random.uniform(0, radius)
        samples[i] = center + r * u
    return samples


def set_net_alpha(net, alphas: tuple[float, ...]):
    for m in net.modules():
        if isinstance(m, SimplexLayer):
            m.set_alphas(alphas)


def projection_simplex(V, z=1, axis=None):
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
    if axis == 1:
        n_features = V.shape[1]
        U = np.sort(V, axis=1)[:, ::-1]
        z = np.ones(len(V)) * z
        cssv = np.cumsum(U, axis=1) - z[:, np.newaxis]
        ind = np.arange(n_features) + 1
        cond = U - cssv / ind > 0
        rho = np.count_nonzero(cond, axis=1)
        theta = cssv[np.arange(len(V)), rho - 1] / rho
        return np.maximum(V - theta[:, np.newaxis], 0)

    elif axis == 0:
        return projection_simplex(V.T, z, axis=1).T

    else:
        V = V.ravel().reshape(1, -1)
        return projection_simplex(V, z, axis=1).ravel()


def compute_riesz_s_energy(simplex_points, d=2):
    diff = (simplex_points[:, None] - simplex_points[None, :])
    # calculate the squared euclidean from each point to another
    dist = np.sqrt((diff ** 2).sum(axis=2))
    # make sure the distance to itself does not count
    np.fill_diagonal(dist, np.inf)
    # epsilon which is the smallest distance possible to avoid an overflow during gradient calculation
    # eps = 10 ** (-320 / (d + 2))
    eps = 1e-4
    b = dist < eps
    dist[b] = eps
    # select only upper triangular matrix to have each mutual distance once
    mutual_dist = dist[np.triu_indices(len(simplex_points), 1)]    
    mutual_dist[np.argwhere(mutual_dist == 0).flatten()] = 1e-4
    # calculate the energy by summing up the squared distances
    energies = (1 / mutual_dist**d)
    energies = energies[~np.isnan(energies)]
    energy = energies.sum()
    log_energy = - np.log(len(mutual_dist)) + np.log(energy)
    return energy, log_energy

def sample_model_point_estimate(model, new_model, sampling_point):
    """
    Creates a new model with the same architecture as the input model, 
    but replaces the last layer's ParameterList with a Linear layer using 
    the (weighted) average of its weights.
    
    Parameters:
    model (nn.Module): The PyTorch model to be processed.
    
    Returns:
    nn.Module: A new model with a Linear layer in place of the ParameterList.
    """    
    # Extract the last layer (assuming it's a fully connected layer with ParameterList)
    last_layer = copy.deepcopy(model.classifier._weights)
    last_layer_bias = copy.deepcopy(model.classifier.bias)

    # Check if last layer is nn.ParameterList
    if not isinstance(last_layer, torch.nn.ParameterList):
        raise TypeError("The last layer must be a ParameterList containing weight tensors.")
    
    # Compute the average of the tensors in the ParameterList
    final_weight = 0
    for factor, weight in zip(sampling_point, last_layer):
        final_weight += factor * weight
    
    # Replace the last layer in the new model with a Linear layer
    # Assuming the input features of the linear layer are the same as the last Parameter tensor
    in_features, out_features = final_weight.size()
    new_model.classifier = torch.nn.Linear(in_features, out_features, bias=True)
    
    # Set the new Linear layer's weights to the averaged weights
    with torch.no_grad():
        new_model.classifier.weight = torch.nn.Parameter(final_weight)
        new_model.classifier.bias = last_layer_bias
    
    return new_model


def seed_weights(weights: list, seed: int) -> None:
    """Seed the weights of a list of nn.Parameter objects."""
    for i, weight in enumerate(weights):
        torch.manual_seed(seed + i)
        torch.nn.init.xavier_normal_(weight)

class StandardConv(nn.Conv2d):
    def __int__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def seed(self, s):
        seed_weights([self.weight], s)
        return self
    

class StandardLinear(nn.Linear):
    def __int__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def seed(self, s):
        seed_weights([self.weight], s)
        return self


class SimplexLayer:
    def __init__(self, init_weight: torch.Tensor, num_endpoints: int, seed: int):
        self.num_endpoints = num_endpoints
        self._alphas = tuple([1/num_endpoints for _ in range(num_endpoints)])  # set by the train() method each round
        self._weights = nn.ParameterList([_initialize_weight(init_weight, seed + i) for i in range(num_endpoints)])

    @property
    def weight(self) -> nn.Parameter:
        return sum(alpha * weight for alpha, weight in zip(self._alphas, self._weights))

    def set_alphas(self, alphas: Union[tuple[float], Literal["center"]]):
        if len(alphas) == len(self._weights):
            self._alphas = alphas
        else:
            raise ValueError(f"alphas must match number of simplex endpoints ({self.num_endpoints})")


class SimplexLinear(nn.Linear, SimplexLayer):
    def __init__(self, num_endpoints: int, seed: int, *args, **kwargs):
        super().__init__(*args, **kwargs)
        SimplexLayer.__init__(self, init_weight=self.weight, num_endpoints=num_endpoints, seed=seed)


class SimplexConv(nn.Conv2d, SimplexLayer):
    def __init__(self, num_endpoints: int, seed: int, *args, **kwargs):
        super().__init__(*args, **kwargs)
        SimplexLayer.__init__(self, init_weight=self.weight, num_endpoints=num_endpoints, seed=seed)


def _initialize_weight(init_weight: torch.Tensor, seed: int) -> nn.Parameter:
    weight = nn.Parameter(torch.zeros_like(init_weight))
    torch.manual_seed(seed)
    torch.nn.init.xavier_normal_(weight)
    return weight