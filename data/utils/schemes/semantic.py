import random
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Set, Union

import numpy as np
import torch
from rich.console import Console
from scipy.optimize import linear_sum_assignment
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from torch.distributions import MultivariateNormal, kl_divergence
from torch.utils.data import DataLoader, Dataset
from torchvision import models

# FL_BENCH_ROOT is the base directory for the project
FL_BENCH_ROOT = Path(__file__).parent.parent.parent.parent.absolute()

# Append the FL_BENCH_ROOT directory to the system path
sys.path.append(FL_BENCH_ROOT.as_posix())

from src.utils.tools import get_optimal_cuda_device

# Define efficient net models and their weights
EFFICIENT_NETS = [
    (models.efficientnet_b0, models.EfficientNet_B0_Weights.DEFAULT),
    (models.efficientnet_b1, models.EfficientNet_B1_Weights.DEFAULT),
    (models.efficientnet_b2, models.EfficientNet_B2_Weights.DEFAULT),
    (models.efficientnet_b3, models.EfficientNet_B3_Weights.DEFAULT),
    (models.efficientnet_b4, models.EfficientNet_B4_Weights.DEFAULT),
    (models.efficientnet_b5, models.EfficientNet_B5_Weights.DEFAULT),
    (models.efficientnet_b6, models.EfficientNet_B6_Weights.DEFAULT),
    (models.efficientnet_b7, models.EfficientNet_B7_Weights.DEFAULT),
]


def subsample(embeddings: Union[torch.Tensor, np.ndarray], num_samples: int):
    """Subsample the embeddings to a specified number of samples."""
    if len(embeddings) < num_samples:
        return embeddings
    subsample_indices = random.sample(range(len(embeddings)), num_samples)
    return embeddings[subsample_indices]


def pairwise_kl_div(
    means_1: torch.Tensor,
    trils_1: torch.Tensor,
    means_2: torch.Tensor,
    trils_2: torch.Tensor,
    batch_size: int,
    device: torch.device,
) -> torch.Tensor:
    """Calculate pairwise KL divergence between two sets of distributions."""
    num_dist_1, num_dist_2 = means_1.shape[0], means_2.shape[0]
    pairwise_kl_matrix = torch.zeros((num_dist_1, num_dist_2), device=device)

    for i in range(0, num_dist_1, batch_size):
        for j in range(0, num_dist_2, batch_size):
            pairwise_kl_matrix[i : i + batch_size, j : j + batch_size] = kl_divergence(
                MultivariateNormal(
                    means_1[i : i + batch_size].unsqueeze(1),
                    scale_tril=trils_1[i : i + batch_size].unsqueeze(1),
                ),
                MultivariateNormal(
                    means_2[j : j + batch_size].unsqueeze(0),
                    scale_tril=trils_2[j : j + batch_size].unsqueeze(0),
                ),
            )
    return pairwise_kl_matrix


def semantic_partition(
    dataset: Dataset,
    targets: np.ndarray,
    target_indices: np.ndarray,
    label_set: Set[int],
    efficient_net_type: int,
    client_num: int,
    pca_components: int,
    seed: int,
    gmm_max_iter: int,
    gmm_init_params: str,
    use_cuda: bool,
    partition: Dict[str, List[np.ndarray]],
    stats: Dict[int, Dict[str, Any]],
):
    """Partition the dataset semantically using embeddings from a trained
    EfficientNet.

    Args:
        dataset (Dataset): The input dataset.
        targets (np.ndarray): Array of data labels.
        target_indices (np.ndarray): Indices of targets. If not set to `--iid`, it will be np.arange(len(targets)).
                                      Otherwise, it contains the absolute indices of the full targets.
        label_set (Set[int]): Set of unique labels from the dataset.
        efficient_net_type (int): Index of the EfficientNet model to use.
        client_num (int): Number of clients.
        pca_components (int): Number of PCA components to reduce dimensions.
        seed (int): Random seed for reproducibility.
        gmm_max_iter (int): Maximum number of iterations for the Gaussian Mixture Model.
        gmm_init_params (str): Parameters to initialize GMM.
        use_cuda (bool): Whether to use CUDA for computations.
        partition (Dict[str, List[np.ndarray]]): Output data indices for each client.
        stats (Dict[int, Dict[str, Any]]): Dictionary to record clients' data distribution statistics.
    """
    device = get_optimal_cuda_device(use_cuda)
    client_ids = list(range(client_num))
    logger = Console()

    # Build the pre-trained EfficientNet model
    logger.log(f"Using model: EfficientNet-B{efficient_net_type}")
    model, weights = EFFICIENT_NETS[efficient_net_type]
    efficient_net = model(weights=weights)
    efficient_net.classifier = torch.nn.Flatten()
    efficient_net = efficient_net.to(device)
    efficient_net.eval()

    # Compute embeddings
    logger.log("Computing embeddings...")
    with torch.no_grad():
        dataloader = DataLoader(dataset, batch_size=256)
        embeddings = []
        for x, _ in dataloader:
            x = x.to(device)
            # Broadcast to 3 channels if input is grayscale
            if x.shape[1] == 1:
                x = x.broadcast_to((x.shape[0], 3, *x.shape[2:]))
            embeddings.append(efficient_net(x).cpu().numpy())

    embeddings = np.concatenate(embeddings)
    embeddings_scaled = StandardScaler(with_std=False).fit_transform(embeddings)
    del embeddings

    # PCA transformation
    if pca_components is None or 0 < pca_components < embeddings_scaled.shape[1]:
        logger.log("Performing PCA transformation...")
        pca = PCA(n_components=pca_components, random_state=seed)
        pca.fit(subsample(embeddings_scaled, 100000))  # Fit PCA to a subsample
        embeddings_scaled = pca.transform(embeddings_scaled)

    # Initialize structures for clustering
    label_cluster_means: Dict[int, torch.Tensor] = {}
    label_cluster_trils: Dict[int, torch.Tensor] = {}
    label_cluster_list = [
        [[] for _ in range(client_num)] for _ in range(len(label_set))
    ]

    for current_label in label_set:
        logger.log(f"Building clusters for label {current_label}")

        idx_current_label = np.where(targets == current_label)[0]
        embeddings_of_current_label = subsample(
            embeddings_scaled[idx_current_label], 10000
        )

        # Fit Gaussian Mixture Model
        gmm = GaussianMixture(
            n_components=client_num,
            max_iter=gmm_max_iter,
            reg_covar=1e-4,
            init_params=gmm_init_params,
            random_state=seed,
        )
        gmm.fit(embeddings_of_current_label)

        cluster_list = gmm.predict(embeddings_of_current_label)

        for idx, cluster in zip(idx_current_label.tolist(), cluster_list):
            label_cluster_list[current_label][cluster].append(idx)

        label_cluster_means[current_label] = torch.tensor(
            gmm.means_, device=device, dtype=torch.float
        )
        label_cluster_trils[current_label] = torch.linalg.cholesky(
            torch.from_numpy(gmm.covariances_).float().to(device)
        )

    # Cluster assignment
    cluster_assignment = [
        [None for _ in range(client_num)] for _ in range(len(label_set))
    ]
    unmatched_labels = list(label_set)

    # Initial assignment
    latest_matched_label = random.choice(unmatched_labels)
    cluster_assignment[latest_matched_label] = client_ids
    unmatched_labels.remove(latest_matched_label)

    # Match labels iteratively
    while unmatched_labels:
        label_to_match = random.choice(unmatched_labels)

        logger.log(
            f"Computing pairwise KL-divergence between label {latest_matched_label} and {label_to_match}"
        )

        cost_matrix = (
            pairwise_kl_div(
                means_1=label_cluster_means[latest_matched_label],
                trils_1=label_cluster_trils[latest_matched_label],
                means_2=label_cluster_means[label_to_match],
                trils_2=label_cluster_trils[label_to_match],
                batch_size=10,
                device=device,
            )
            .cpu()
            .numpy()
        )

        optimal_local_assignment = linear_sum_assignment(cost_matrix)

        for client_id in client_ids:
            cluster_assignment[label_to_match][
                optimal_local_assignment[1][client_id]
            ] = cluster_assignment[latest_matched_label][
                optimal_local_assignment[0][client_id]
            ]

        unmatched_labels.remove(label_to_match)
        latest_matched_label = label_to_match

    # Allocate data indices to clients based on cluster assignments
    for current_label in label_set:
        for client_id in client_ids:
            partition["data_indices"][
                cluster_assignment[current_label][client_id]
            ].extend(label_cluster_list[current_label][client_id])

    # Gather statistics for each client
    for client_id in range(client_num):
        client_data_indices = partition["data_indices"][client_id]
        stats[client_id] = {
            "x": len(targets[client_data_indices]),
            "y": dict(Counter(targets[client_data_indices].tolist())),
        }
        # Update indices to use the original target indices
        partition["data_indices"][client_id] = target_indices[client_data_indices]

    # Calculate statistics for number of samples per client
    num_samples = np.array([stat["x"] for stat in stats.values()])
    stats["samples_per_client"] = {
        "mean": num_samples.mean().item(),
        "stddev": num_samples.std().item(),
    }

    logger.log("All operations completed successfully!")
