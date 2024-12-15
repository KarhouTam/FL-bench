import random
from collections import Counter
from typing import Any, Dict, List, Set

import numpy as np


def iid_partition(
    targets: np.ndarray,
    target_indices: np.ndarray,
    label_set: Set[int],
    client_num: int,
    partition: Dict[str, List[np.ndarray]],
    stats: Dict[int, Dict[str, Any]],
):
    """Partitions the dataset into IID (Independent and Identically
    Distributed) segments for each client.

    Args:
        targets (np.ndarray): Array of data labels.
        target_indices (np.ndarray): Indices of targets. If not set to `--iid`, it will be np.arange(len(targets)).
                                      Otherwise, it contains the absolute indices of the full targets.
        label_set (Set[int]): Set of unique labels to be included in the partition.
        client_num (int): Total number of clients.
        partition (Dict[str, List[np.ndarray]]): Output data indices for each client.
        stats (Dict[int, Dict[str, Any]]): Dictionary to record clients' data distribution statistics.
    """

    # Filter valid target indices that belong to the label set
    valid_target_indices = [
        i for i in range(len(target_indices)) if targets[i] in label_set
    ]
    random.shuffle(valid_target_indices)  # Shuffle to ensure randomness

    # Calculate the size of each client's partition
    partition_size = int(len(valid_target_indices) / client_num)
    num_samples_per_client = []

    # Allocate data indices to each client
    for client_id in range(client_num):
        # Determine the data indices for the current client
        client_partition_indices = valid_target_indices[
            partition_size * client_id : partition_size * (client_id + 1)
        ]

        # Store indices in partition dictionary
        partition["data_indices"][client_id] = client_partition_indices

        # Initialize stats for the current client
        stats[client_id] = {"x": len(client_partition_indices), "y": None}
        stats[client_id]["y"] = dict(
            Counter(targets[client_partition_indices].tolist())
        )

        # Keep track of the number of samples allocated
        num_samples_per_client.append(len(client_partition_indices))

    # Calculate statistics for the number of samples per client
    num_samples_array = np.array(num_samples_per_client)
    stats["samples_per_client"] = {
        "mean": num_samples_array.mean().item(),
        "stddev": num_samples_array.std().item(),
    }
