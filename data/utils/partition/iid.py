import random
from collections import Counter
from typing import Dict, List, Tuple

import numpy as np
from torch.utils.data import Dataset


def iid_partition(
    ori_dataset: Dataset, num_clients: int
) -> Tuple[List[List[int]], Dict]:
    partition = {"separation": None, "data_indices": None}
    stats = {}
    data_indices = [[] for _ in range(num_clients)]
    targets_numpy = np.array(ori_dataset.targets, dtype=np.int64)
    idx = list(range(len(targets_numpy)))
    random.shuffle(idx)
    size = int(len(idx) / num_clients)

    for i in range(num_clients):
        data_indices[i] = idx[size * i : size * (i + 1)]
        stats[i] = {"x": None, "y": None}
        stats[i]["x"] = len(data_indices[i])
        stats[i]["y"] = Counter(targets_numpy[data_indices[i]].tolist())

    num_samples = np.array(list(map(lambda stat_i: stat_i["x"], stats.values())))
    stats["sample per client"] = {
        "std": num_samples.mean(),
        "stddev": num_samples.std(),
    }

    partition["data_indices"] = data_indices

    return partition, stats
