import random
from collections import Counter
from typing import Dict, List, Tuple

import numpy as np
from torch.utils.data import Dataset


def allocate_shards(
    ori_dataset: Dataset, client_num: int, shard_num: int
) -> Tuple[List[List[int]], Dict[str, Dict[str, int]]]:
    partition = {"separation": None, "data_indices": None}

    shards_total = client_num * shard_num
    # one shard's length indicate how many data samples that belongs to one class that one client can obtain.
    size_of_shards = int(len(ori_dataset) / shards_total)

    data_indices = [[] for _ in range(client_num)]

    targets_numpy = np.array(ori_dataset.targets, dtype=np.int32)

    # sort sample indices according to labels
    idxs_labels = np.vstack(
        (np.arange(len(ori_dataset), dtype=np.int64), targets_numpy)
    )
    # corresponding labels after sorting are [0, .., 0, 1, ..., 1, ...]
    idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
    idxs_argsorted = idxs_labels[0, :].tolist()

    # assign
    idx_shard = list(range(shards_total))
    for i in range(client_num):
        rand_set = random.sample(idx_shard, shard_num)
        idx_shard = list(set(idx_shard) - set(rand_set))
        for rand in rand_set:
            data_indices[i] = np.concatenate(
                [
                    data_indices[i],
                    idxs_argsorted[rand * size_of_shards : (rand + 1) * size_of_shards],
                ],
                axis=0,
            ).astype(np.int64)
        data_indices[i] = data_indices[i].tolist()

    stats = {}
    for i, idxs in enumerate(data_indices):
        stats[i] = {"x": None, "y": None}
        stats[i]["x"] = len(idxs)
        stats[i]["y"] = Counter(targets_numpy[idxs].tolist())

    num_samples = np.array(list(map(lambda stat_i: stat_i["x"], stats.values())))
    stats["sample per client"] = {
        "std": num_samples.mean(),
        "stddev": num_samples.std(),
    }

    partition["data_indices"] = data_indices

    return partition, stats
