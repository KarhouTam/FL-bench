from collections import Counter
from typing import Dict, List, Tuple

import numpy as np
from torch.utils.data import Dataset


def dirichlet(
    ori_dataset: Dataset, num_clients: int, alpha: float, least_samples: int
) -> Tuple[List[List[int]], Dict]:
    num_classes = len(ori_dataset.classes)
    min_size = 0
    stats = {}
    partition = {"separation": None, "data_indices": None}

    targets_numpy = np.array(ori_dataset.targets, dtype=np.int32)
    idx = [np.where(targets_numpy == i)[0] for i in range(num_classes)]

    while min_size < least_samples:
        data_indices = [[] for _ in range(num_clients)]
        for k in range(num_classes):
            np.random.shuffle(idx[k])
            distrib = np.random.dirichlet(np.repeat(alpha, num_clients))
            distrib = np.array(
                [
                    p * (len(idx_j) < len(targets_numpy) / num_clients)
                    for p, idx_j in zip(distrib, data_indices)
                ]
            )
            distrib = distrib / distrib.sum()
            distrib = (np.cumsum(distrib) * len(idx[k])).astype(int)[:-1]
            data_indices = [
                np.concatenate((idx_j, idx.tolist())).astype(np.int64)
                for idx_j, idx in zip(data_indices, np.split(idx[k], distrib))
            ]
            min_size = min([len(idx_j) for idx_j in data_indices])

    for i in range(num_clients):
        stats[i] = {"x": None, "y": None}
        stats[i]["x"] = len(targets_numpy[data_indices[i]])
        stats[i]["y"] = Counter(targets_numpy[data_indices[i]].tolist())

    num_samples = np.array(list(map(lambda stat_i: stat_i["x"], stats.values())))
    stats["sample per client"] = {
        "std": num_samples.mean(),
        "stddev": num_samples.std(),
    }

    partition["data_indices"] = data_indices

    return partition, stats
