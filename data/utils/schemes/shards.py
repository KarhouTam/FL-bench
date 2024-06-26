import random
from collections import Counter
from typing import Dict, Set

import numpy as np


def allocate_shards(
    targets: np.ndarray,
    target_indices: np.ndarray,
    label_set: Set[int],
    client_num: int,
    shard_num: int,
    partition: dict,
    stats: dict,
):
    """Refer to the mehtod used in FedAvg paper. Sort data by label first and
    split them into `shard_num * client_num` and allocate each client
    `shard_num` shards.

    Args:
        targets (np.ndarray): Data label array.
        target_indices (np.ndarray): Indices of targets. If you haven't set `--iid`, then it will be np.arange(len(targets))
        Otherwise, it will be the absolute indices of the full targets.
        label_set (set): Label set.
        client_num (int): Number of clients.
        shard_num (int): Number of shards. A shard means a sub-list of data indices.
        partition (Dict): Output data indices dict.
        stats (Dict): Output dict that recording clients data distribution.
    """
    shards_total = client_num * shard_num
    # one shard's length indicate how many data samples that belongs to one class that one client can obtain.
    indices = [i for i in range(len(targets)) if targets[i] in label_set]
    targets = targets[indices]
    size_of_shards = int(len(targets) / shards_total)

    # sort sample indices according to labels
    idxs_labels = np.vstack((np.arange(len(targets), dtype=np.int64), targets))
    # corresponding labels after sorting are [0, .., 0, 1, ..., 1, ...]
    idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
    idxs_argsorted = idxs_labels[0, :].tolist()

    # assign
    idx_shard = list(range(shards_total))
    for i in range(client_num):
        rand_set = random.sample(idx_shard, shard_num)
        idx_shard = list(set(idx_shard) - set(rand_set))
        for rand in rand_set:
            partition["data_indices"][i] = np.concatenate(
                [
                    partition["data_indices"][i],
                    idxs_argsorted[rand * size_of_shards : (rand + 1) * size_of_shards],
                ]
            ).astype(np.int64)

    for i in range(client_num):
        stats[i] = {"x": None, "y": None}
        stats[i]["x"] = len(targets[partition["data_indices"][i]])
        stats[i]["y"] = dict(Counter(targets[partition["data_indices"][i]].tolist()))
        partition["data_indices"][i] = target_indices[partition["data_indices"][i]]

    num_samples = np.array(list(map(lambda stat_i: stat_i["x"], stats.values())))
    stats["samples_per_client"] = {
        "std": num_samples.mean().item(),
        "stddev": num_samples.std().item(),
    }
