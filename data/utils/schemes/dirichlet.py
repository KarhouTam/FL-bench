from collections import Counter
from typing import Any, Dict, Set

import numpy as np


def dirichlet(
    targets: np.ndarray,
    target_indices: np.ndarray,
    label_set: Set[int],
    client_num: int,
    alpha: float,
    min_samples_per_client: int,
    partition: Dict[str, Any],
    stats: Dict[int, Dict[str, Any]],
):
    """Partition the dataset according to the Dirichlet distribution using a
    specified concentration parameter, `alpha`.

    Args:
        targets (np.ndarray): Array of data labels.
        target_indices (np.ndarray): Indices of targets. If not set to `--iid`, it will be np.arange(len(targets)).
                                      Otherwise, it holds the absolute indices of the full targets.
        label_set (Set[int]): Set of unique labels.
        client_num (int): Number of clients.
        alpha (float): Concentration parameter; smaller values indicate stronger data heterogeneity.
        min_samples_per_client (int): Minimum number of data samples each client should have.
        partition (Dict[str, Any]): Dictionary to hold output data indices for each client.
        stats (Dict[int, Dict[str, Any]]): Dictionary to record clients' data distribution.
    """

    min_size = 0
    # Map each label to its corresponding indices in the target array
    indices_per_label = {label: np.where(targets == label)[0] for label in label_set}

    while min_size < min_samples_per_client:
        # Initialize empty lists to hold data indices for each client
        partition["data_indices"] = [[] for _ in range(client_num)]

        # Iterate through each label in the label_set
        for label in label_set:
            # Shuffle the indices corresponding to the current label
            np.random.shuffle(indices_per_label[label])

            # Generate a Dirichlet distribution for partitioning data among clients
            distribution = np.random.dirichlet(np.repeat(alpha, client_num))

            # Calculate the cumulative distribution to get split indices
            cumulative_distribution = np.cumsum(distribution) * len(
                indices_per_label[label]
            )
            split_indices_position = cumulative_distribution.astype(int)[:-1]

            # Split the indices based on the calculated positions
            split_indices = np.split(indices_per_label[label], split_indices_position)

            # Assign the split indices to each client
            for client_id in range(client_num):
                partition["data_indices"][client_id].extend(split_indices[client_id])

        # Update the minimum number of samples across all clients
        min_size = min(len(indices) for indices in partition["data_indices"])

    # Gather statistics and prepare the output for each client
    for client_id in range(client_num):
        stats[client_id]["x"] = len(targets[partition["data_indices"][client_id]])
        stats[client_id]["y"] = dict(
            Counter(targets[partition["data_indices"][client_id]].tolist())
        )

        # Update the data indices to use the original target indices
        partition["data_indices"][client_id] = target_indices[
            partition["data_indices"][client_id]
        ]

    # Calculate the number of samples for each client and update statistics
    sample_counts = np.array([stat["x"] for stat in stats.values()])
    stats["samples_per_client"] = {
        "mean": sample_counts.mean().item(),
        "stddev": sample_counts.std().item(),
    }
