import random
from collections import Counter
from typing import Dict, List, Tuple

import numpy as np
from torch.utils.data import Dataset


def randomly_assign_classes(
    dataset: Dataset, client_num: int, class_num: int
) -> Tuple[List[List[int]], Dict[str, Dict[str, int]]]:
    partition = {"separation": None, "data_indices": None}
    data_indices = [[] for _ in range(client_num)]
    targets_numpy = np.array(dataset.targets, dtype=np.int32)
    label_list = list(range(len(dataset.classes)))

    data_idx_for_each_label = [
        np.where(targets_numpy == i)[0].tolist() for i in label_list
    ]
    assigned_labels = []
    selected_times = [0 for _ in label_list]
    for i in range(client_num):
        sampled_labels = random.sample(label_list, class_num)
        assigned_labels.append(sampled_labels)
        for j in sampled_labels:
            selected_times[j] += 1

    labels_count = Counter(targets_numpy)

    batch_sizes = np.zeros_like(label_list)
    for i in label_list:
        if selected_times[i] == 0:
            batch_sizes[i] = 0
        else:
            batch_sizes[i] = int(labels_count[i] / selected_times[i])
            
    for i in range(client_num):
        for cls in assigned_labels[i]:
            if len(data_idx_for_each_label[cls]) < 2 * batch_sizes[cls]:
                batch_size = len(data_idx_for_each_label[cls])
            else:
                batch_size = batch_sizes[cls]
            selected_idx = random.sample(data_idx_for_each_label[cls], batch_size)
            data_indices[i] = np.concatenate(
                [data_indices[i], selected_idx], axis=0
            ).astype(np.int64)
            data_idx_for_each_label[cls] = list(
                set(data_idx_for_each_label[cls]) - set(selected_idx)
            )

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
