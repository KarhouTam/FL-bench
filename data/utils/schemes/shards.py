import random
from collections import Counter
from typing import Any, Dict, List, Set

import numpy as np


def allocate_shards(
    targets: np.ndarray,
    target_indices: np.ndarray,
    label_set: Set[int],
    client_num: int,
    shard_num: int,
    partition: Dict[str, List[np.ndarray]],
    stats: Dict[int, Dict[str, Any]],
):
    """Allocate data shards to clients based on the method described in the
    FedAvg paper. The data is sorted by label and then split into `shard_num *
    client_num` segments, with each client receiving `shard_num` shards.

    Args:
        targets (np.ndarray): Array of data labels.
        target_indices (np.ndarray): Indices of targets. If not set to `--iid`, it will be np.arange(len(targets)).
                                      Otherwise, it contains the absolute indices of the full targets.
        label_set (Set[int]): Set of unique labels.
        client_num (int): Total number of clients.
        shard_num (int): Number of shards to allocate to each client.
        partition (Dict[str, List[np.ndarray]]): Output data indices for each client.
        stats (Dict[int, Dict[str, Any]]): Dictionary to record clients' data distribution statistics.
    """

    # Calculate the total number of shards
    total_shards = client_num * shard_num

    # Filter the indices to only include valid targets within the label set
    valid_indices = [i for i in range(len(targets)) if targets[i] in label_set]
    targets = targets[valid_indices]  # Filter targets correspondingly
    shards_size = len(targets) // total_shards  # Calculate the size of each shard

    # Create a sorting index based on labels
    sorted_indices = np.argsort(targets)

    # Initialize shard indices
    available_shard_indices = list(range(total_shards))

    # Allocate shards to each client
    for client_id in range(client_num):
        # Randomly select shard indices for the current client
        selected_shards = random.sample(available_shard_indices, shard_num)
        available_shard_indices = list(
            set(available_shard_indices) - set(selected_shards)
        )  # Update available shards

        # Assign the selected shards to the current client's data
        for shard in selected_shards:
            shard_data = sorted_indices[shard * shards_size : (shard + 1) * shards_size]
            partition["data_indices"][client_id].extend(shard_data)

    # Gather statistics for each client
    for client_id in range(client_num):
        client_data_indices = partition["data_indices"][client_id]
        stats[client_id] = {
            "x": len(targets[client_data_indices]),
            "y": dict(Counter(targets[client_data_indices].tolist())),
        }
        # Update the data indices to point to the original target indices
        partition["data_indices"][client_id] = target_indices[client_data_indices]

    # Calculate statistics for number of samples per client
    sample_counts = np.array([stat["x"] for stat in stats.values()])
    stats["samples_per_client"] = {
        "mean": sample_counts.mean().item(),
        "stddev": sample_counts.std().item(),
    }
