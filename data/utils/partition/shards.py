import random
from collections import Counter
from typing import Dict, List, Tuple

import numpy as np
from torch.utils.data import Dataset


def allocate_shards(
    ori_dataset: Dataset, num_clients: int, num_shards: int
) -> Tuple[List[List[int]], Dict[str, Dict[str, int]]]:
    partition = {"separation": None, "data_indices": None}

    shards_total = num_clients * num_shards
    # one shard's length indicate how many data samples that belongs to one class that one client can obtain.
    size_of_shards = int(len(ori_dataset) / shards_total)

    data_indices = [[] for _ in range(num_clients)]

    targets_numpy = np.array(ori_dataset.targets, dtype=np.int32)
    idx = np.arange(len(ori_dataset), dtype=np.int64)

    # sort sample indices according to labels
    idx_targets = np.vstack((idx, targets_numpy))
    # corresponding labels after sorting are [0, .., 0, 1, ..., 1, ...]
    idx_targets = idx_targets[:, idx_targets[1, :].argsort()]
    idx = idx_targets[0, :].tolist()

    # assign
    idx_shard = list(range(shards_total))
    for i in range(num_clients):
        rand_set = random.sample(idx_shard, num_shards)
        idx_shard = list(set(idx_shard) - set(rand_set))
        for rand in rand_set:
            data_indices[i] = np.concatenate(
                [
                    data_indices[i],
                    idx[rand * size_of_shards : (rand + 1) * size_of_shards],
                ],
                axis=0,
            ).astype(np.int64)
        data_indices[i] = data_indices[i].tolist()

    stats = {}
    for i, idx in enumerate(data_indices):
        stats[i] = {"x": None, "y": None}
        stats[i]["x"] = len(idx)
        stats[i]["y"] = Counter(targets_numpy[idx].tolist())

    num_samples = np.array(list(map(lambda stat_i: stat_i["x"], stats.values())))
    stats["sample per client"] = {
        "std": num_samples.mean(),
        "stddev": num_samples.std(),
    }

    partition["data_indices"] = data_indices

    return partition, stats
