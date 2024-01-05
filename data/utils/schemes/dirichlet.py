from collections import Counter
from typing import Dict

import numpy as np


def dirichlet(
    targets: np.ndarray,
    label_set: set,
    client_num: int,
    alpha: float,
    least_samples: int,
    partition: Dict,
    stats: Dict,
):
    """Partition dataset according to Dirichlet with concentration parameter `alpha`.

    Args:
        targets (np.ndarray): Data label array.
        label_set (set): Label set.
        client_num (int): Number of clients.
        alpha (float): Concentration parameter. Smaller alpha indicates strong data heterogeneity.
        least_samples (int): Lease number of data samples each client should have.
        partition (Dict): Output data indices dict.
        stats (Dict): Output dict that recording clients data distribution.

    """
    min_size = 0
    indices_4_labels = {i: np.where(targets == i)[0] for i in label_set}

    while min_size < least_samples:
        partition["data_indices"][:client_num] = [[] for _ in range(client_num)]

        for k in label_set:
            np.random.shuffle(indices_4_labels[k])
            distrib = np.random.dirichlet(np.repeat(alpha, client_num))
            distrib = np.array(
                [
                    p * (len(idx_j) < len(targets) / client_num)
                    for p, idx_j in zip(distrib, partition["data_indices"])
                ]
            )
            distrib = distrib / distrib.sum()
            distrib = (np.cumsum(distrib) * len(indices_4_labels[k])).astype(int)[:-1]
            partition["data_indices"][:client_num] = [
                np.concatenate((idx_j, idx.tolist())).astype(np.int64)
                for idx_j, idx in zip(
                    partition["data_indices"], np.split(indices_4_labels[k], distrib)
                )
            ]
            min_size = min(
                [len(idx_j) for idx_j in partition["data_indices"][:client_num]]
            )

    for i in range(client_num):
        stats[i] = {"x": None, "y": None}
        stats[i]["x"] = len(targets[partition["data_indices"][i]])
        stats[i]["y"] = dict(Counter(targets[partition["data_indices"][i]].tolist()))

    sample_num = np.array(list(map(lambda stat_i: stat_i["x"], stats.values())))
    stats["sample per client"] = {
        "std": sample_num.mean().item(),
        "stddev": sample_num.std().item(),
    }
