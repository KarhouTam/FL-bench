import random
from collections import Counter
from typing import Dict

import numpy as np


def randomly_assign_classes(
    targets: np.ndarray,
    target_indices: np.ndarray,
    label_set: set,
    client_num: int,
    class_num: int,
    partition: dict,
    stats: dict,
):
    """Partition data to make each client has almost `client_num` class of
    data.

    Args:
        targets (np.ndarray): Data label array.
        target_indices (np.ndarray): Indices of targets. If you haven't set `--iid`, then it will be np.arange(len(targets))
        Otherwise, it will be the absolute indices of the full targets.
        label_set (set): Label set.
        client_num (int): Number of clients.
        class_num (int): Class num.
        partition (Dict): Output data indices dict.
        stats (Dict): Output dict that recording clients data distribution.
    """

    class_indices = {i: sorted(np.where(targets == i)[0].tolist()) for i in label_set}
    assigned_labels = []
    selected_times = {i: 0 for i in label_set}
    label_sequence = sorted(label_set)
    for i in range(client_num):
        sampled_labels = random.sample(label_sequence, class_num)
        assigned_labels.append(sampled_labels)
        for j in sampled_labels:
            selected_times[j] += 1

    labels_count = Counter(targets)

    batch_sizes = {i: 0 for i in label_set}
    for i in label_set:
        if selected_times[i] == 0:
            batch_sizes[i] = 0
        else:
            batch_sizes[i] = int(labels_count[i] / selected_times[i])

    for i in range(client_num):
        for cls in assigned_labels[i]:
            if len(class_indices[cls]) < 2 * batch_sizes[cls]:
                batch_size = len(class_indices[cls])
            else:
                batch_size = batch_sizes[cls]
            selected_idxs = random.sample(class_indices[cls], batch_size)
            partition["data_indices"][i] = np.concatenate(
                [partition["data_indices"][i], selected_idxs]
            ).astype(np.int64)
            class_indices[cls] = list(set(class_indices[cls]) - set(selected_idxs))

    for i in range(client_num):
        stats[i] = {"x": None, "y": None}
        stats[i]["x"] = len(partition["data_indices"][i])
        stats[i]["y"] = dict(Counter(targets[partition["data_indices"][i]].tolist()))
        partition["data_indices"][i] = target_indices[partition["data_indices"][i]]

    num_samples = np.array(list(map(lambda stat_i: stat_i["x"], stats.values())))
    stats["samples_per_client"] = {
        "std": num_samples.mean().item(),
        "stddev": num_samples.std().item(),
    }
