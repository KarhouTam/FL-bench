import random
from collections import Counter
from typing import Set

import numpy as np


def iid_partition(
    targets: np.ndarray,
    target_indices: np.ndarray,
    label_set: Set[int],
    client_num: int,
    partition: dict,
    stats: dict,
):
    target_indices = [i for i in range(len(target_indices)) if targets[i] in label_set]
    random.shuffle(target_indices)
    size = int(len(target_indices) / client_num)
    num_samples = []

    for i in range(client_num):
        partition_i = np.array(
            target_indices[size * i : size * (i + 1)], dtype=np.int32
        )
        partition["data_indices"][i] = partition_i
        stats[i] = {"x": None, "y": None}
        stats[i]["x"] = len(partition_i)
        stats[i]["y"] = dict(Counter(targets[partition_i].tolist()))
        num_samples.append(len(partition_i))

    num_samples = np.array(num_samples)
    stats["samples_per_client"] = {
        "std": num_samples.mean().item(),
        "stddev": num_samples.std().item(),
    }
