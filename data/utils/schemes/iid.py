import random
from collections import Counter
from typing import Dict, List, Tuple

import numpy as np
from torch.utils.data import Dataset


def iid_partition(dataset: Dataset, client_num: int) -> Tuple[List[List[int]], Dict]:
    partition = {"separation": None, "data_indices": None}
    stats = {}
    data_indices = [[] for _ in range(client_num)]
    targets_numpy = np.array(dataset.targets, dtype=np.int64)
    idx = list(range(len(targets_numpy)))
    random.shuffle(idx)
    size = int(len(idx) / client_num)

    for i in range(client_num):
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
