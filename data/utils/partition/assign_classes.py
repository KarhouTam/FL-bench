import random
from collections import Counter
from typing import Dict, List, Tuple

import numpy as np
from torch.utils.data import Dataset


def randomly_assign_classes(
    ori_dataset: Dataset, num_clients: int, num_classes: int
) -> Tuple[List[List[int]], Dict[str, Dict[str, int]]]:
    partition = {"separation": None, "data_indices": None}
    data_indices = [[] for _ in range(num_clients)]
    targets_numpy = np.array(ori_dataset.targets, dtype=np.int32)
    classes_label = list(range(len(ori_dataset.classes)))
    idx = [np.where(targets_numpy == i)[0].tolist() for i in classes_label]
    assigned_classes = [[] for _ in range(num_clients)]
    selected_classes = list(range(len(ori_dataset.classes)))
    if num_classes * num_clients > len(selected_classes):
        selected_classes.extend(
            np.random.choice(
                classes_label, num_classes * num_clients - len(selected_classes)
            ).tolist()
        )
    random.shuffle(selected_classes)
    for i, cls in enumerate(range(0, num_clients * num_classes, num_classes)):
        assigned_classes[i] = selected_classes[cls : cls + num_classes]

    selected_times = Counter(selected_classes[: num_clients * num_classes])
    labels_count = Counter(targets_numpy)
    batch_size = np.zeros_like(classes_label)

    for cls in selected_times.keys():
        batch_size[cls] = int(labels_count[cls] / selected_times[cls])

    for i in range(num_clients):
        for cls in assigned_classes[i]:
            selected_idx = random.sample(idx[cls], batch_size[cls])
            data_indices[i] = np.concatenate(
                [data_indices[i], selected_idx], axis=0
            ).astype(np.int64)
            idx[cls] = list(set(idx[cls]) - set(selected_idx))

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
