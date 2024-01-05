import random
from collections import Counter
from typing import Dict

import numpy as np


def randomly_assign_classes(
    targets: np.ndarray,
    label_set: set,
    client_num: int,
    class_num: int,
    partition: Dict,
    stats: Dict,
):
    """Partition data to make each client has almost `client_num` class of data.

    Args:
        targets (np.ndarray): Data label array.
        label_set (set): Label set.
        client_num (int): Number of clients.
        class_num (int): Class num.
        partition (Dict): Output data indices dict.
        stats (Dict): Output dict that recording clients data distribution.
    """

    data_idx_for_each_label = {i: np.where(targets == i)[0].tolist() for i in label_set}
    assigned_labels = []
    selected_times = {i: 0 for i in label_set}
    for i in range(client_num):
        sampled_labels = random.sample(label_set, class_num)
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
            if len(data_idx_for_each_label[cls]) < 2 * batch_sizes[cls]:
                batch_size = len(data_idx_for_each_label[cls])
            else:
                batch_size = batch_sizes[cls]
            selected_idx = random.sample(data_idx_for_each_label[cls], batch_size)
            partition["data_indices"][i] = np.concatenate(
                [partition["data_indices"][i], selected_idx]
            ).astype(np.int64)
            data_idx_for_each_label[cls] = list(
                set(data_idx_for_each_label[cls]) - set(selected_idx)
            )

        partition["data_indices"][i] = partition["data_indices"][i].tolist()

    for i in range(client_num):
        stats[i] = {"x": None, "y": None}
        stats[i]["x"] = len(targets[partition["data_indices"][i]])
        stats[i]["y"] = dict(Counter(targets[partition["data_indices"][i]].tolist()))

    num_samples = np.array(list(map(lambda stat_i: stat_i["x"], stats.values())))
    stats["sample per client"] = {
        "std": num_samples.mean().item(),
        "stddev": num_samples.std().item(),
    }
