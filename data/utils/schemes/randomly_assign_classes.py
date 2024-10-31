import random
from collections import Counter
from typing import Any, Dict, List, Set

import numpy as np


def randomly_assign_classes(
    targets: np.ndarray,
    target_indices: np.ndarray,
    label_set: Set[int],
    client_num: int,
    class_num: int,
    partition: Dict[str, List[np.ndarray]],
    stats: Dict[int, Dict[str, Any]],
):
    """Partition data to ensure each client has a nearly equal distribution of
    classes.

    Args:
        targets (np.ndarray): Array of data labels.
        target_indices (np.ndarray): Indices of targets. If not set to `--iid`, it will be np.arange(len(targets)).
                                      Otherwise, it contains the absolute indices of the full targets.
        label_set (Set[int]): Set of unique labels from the dataset.
        client_num (int): Total number of clients.
        class_num (int): Number of classes to assign to each client.
        partition (Dict[str, List[np.ndarray]]): Output data indices for each client.
        stats (Dict[int, Dict[str, Any]]): Dictionary to record clients' data distribution.
    """

    # Create a mapping from each label to its indices in the targets
    class_indices = {
        label: sorted(np.where(targets == label)[0].tolist()) for label in label_set
    }

    # Initialize structures to track assigned labels and selected times for each label
    assigned_labels = []
    label_selection_counts = {label: 0 for label in label_set}
    sorted_labels = sorted(label_set)

    # Randomly assign classes to each client
    for client_id in range(client_num):
        # Sample class labels for the current client
        sampled_labels = random.sample(sorted_labels, class_num)
        assigned_labels.append(sampled_labels)

        # Update selection counts for the sampled labels
        for label in sampled_labels:
            label_selection_counts[label] += 1

    # Count occurrences of each label in targets
    label_counts = Counter(targets)

    # Calculate batch sizes for the classes based on selection counts
    batch_sizes = {
        label: (
            0
            if label_selection_counts[label] == 0
            else int(label_counts[label] / label_selection_counts[label])
        )
        for label in label_set
    }

    # Assign data indices to each client based on sampled labels and batch sizes
    for client_id in range(client_num):
        for label in assigned_labels[client_id]:
            # Determine batch size for the current label
            if len(class_indices[label]) < 2 * batch_sizes[label]:
                batch_size = len(class_indices[label])
            else:
                batch_size = batch_sizes[label]

            # Randomly select indices for the current label
            selected_indices = random.sample(class_indices[label], batch_size)

            # Update the partition for the current client
            partition["data_indices"][client_id].extend(selected_indices)

            # Remove selected indices from the available class indices
            class_indices[label] = list(
                set(class_indices[label]) - set(selected_indices)
            )

    # Gather statistics and prepare the output for each client
    for client_id in range(client_num):
        client_data_indices = partition["data_indices"][client_id]
        stats[client_id] = {
            "x": len(client_data_indices),
            "y": dict(Counter(targets[client_data_indices].tolist())),
        }

        # Update the data indices to use the original target indices
        partition["data_indices"][client_id] = target_indices[client_data_indices]

    # Calculate the number of samples for each client and update statistics
    sample_counts = np.array([stat["x"] for stat in stats.values()])
    stats["samples_per_client"] = {
        "mean": sample_counts.mean().item(),
        "stddev": sample_counts.std().item(),
    }
