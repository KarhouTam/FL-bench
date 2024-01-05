import random
from collections import Counter
from typing import Dict, Set

import numpy as np


def iid_partition(
    targets: np.ndarray,
    label_set: Set[int],
    client_num: int,
    partition: Dict,
    stats: Dict,
):
    indices = [i for i in range(len(targets)) if targets[i] in label_set]
    random.shuffle(indices)
    size = int(len(indices) / client_num)

    for i in range(client_num):
        partition["data_indices"][i] = np.array(
            indices[size * i : size * (i + 1)], dtype=np.int64
        )
        stats[i] = {"x": None, "y": None}
        stats[i]["x"] = len(partition["data_indices"][i])
        stats[i]["y"] = dict(Counter(targets[partition["data_indices"][i]].tolist()))

    num_samples = np.array(list(map(lambda stat_i: stat_i["x"], stats.values())))
    stats["sample per client"] = {
        "std": num_samples.mean().item(),
        "stddev": num_samples.std().item(),
    }
