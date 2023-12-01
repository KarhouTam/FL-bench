from collections import Counter
from typing import Dict, List, Tuple

import numpy as np
from torch.utils.data import Dataset
from collections import Counter


def dirichlet_4_domainnet(
    partition, args, clients_4_train, domain_indices_bound, targets_numpy
):
    domain_2_domain_label = {
        "clipart": 0,
        "infograph": 1,
        "painting": 2,
        "quickdraw": 3,
        "real": 4,
        "sketch": 5,
    }
    # the indices of all training data on all training clients
    all_training_data_indices = []
    for client_idx in clients_4_train:
        all_training_data_indices = (
            all_training_data_indices + partition["data_indices"][client_idx]["train"]
        )
    all_training_data_indices.sort()

    def index_2_domain_label(index):
        # get the domain label of a specific data index
        for domain, bound in domain_indices_bound.items():
            if index >= bound["begin"] and index < bound["end"]:
                return domain_2_domain_label[domain]

    all_data_domain_label = list(map(index_2_domain_label, all_training_data_indices))
    # generate heterogeneous partition
    min_size = 0
    while min_size < args.least_samples:
        idx_batch = [[] for _ in range(len(clients_4_train))]
        for domain_label in range(5):
            proportions = np.random.dirichlet(
                np.repeat(args.alpha, len(clients_4_train))
            )
            idx = np.where(np.array(all_data_domain_label) == domain_label)[0]
            np.random.shuffle(idx)
            proportions = np.array(
                [
                    p
                    * (
                        len(idx_j)
                        < len(all_training_data_indices) / len(clients_4_train)
                    )
                    for p, idx_j in zip(proportions, idx_batch)
                ]
            )  # balance proportion
            proportions = proportions / proportions.sum()
            proportions = (np.cumsum(proportions) * len(idx)).astype(int)[:-1]
            idx_batch = [
                idx_j + idx.tolist()
                for idx_j, idx in zip(idx_batch, np.split(idx, proportions))
            ]
            min_size = min([len(idx_j) for idx_j in idx_batch])
    for client_idx, sample_idx in zip(clients_4_train, idx_batch):
        partition["data_indices"][client_idx]["train"] = sample_idx
    # generate stats
    client_num = args.client_num
    stats = {}
    for i in clients_4_train:
        stats[i] = {"x": None, "y": None}
        stats[i]["x"] = len(partition["data_indices"][i]["train"]) + len(
            partition["data_indices"][i]["test"]
        )
        stats[i]["y"] = Counter(
            targets_numpy[partition["data_indices"][i]["train"]].tolist()
            + targets_numpy[partition["data_indices"][i]["test"]].tolist()
        )

    num_samples = np.array(
        [
            len(
                partition["data_indices"][j]["train"]
                + partition["data_indices"][j]["test"]
            )
            for j in clients_4_train
        ]
    )
    stats["sample per client"] = {
        "std": num_samples.mean(),
        "stddev": num_samples.std(),
    }
    stats["domain_info"] = {}
    partition_data_indices = partition["data_indices"]
    for client_idx, indices in enumerate(partition_data_indices):
        indices = indices["train"]
        domain_label_list = list(map(index_2_domain_label, indices))
        domain_label_count = Counter(domain_label_list)
        stats["domain_info"][client_idx] = domain_label_count
    return partition, stats
