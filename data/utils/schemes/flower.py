import json
from collections import Counter

import datasets
import numpy as np

from data.utils.process import partitioner_class_from_flwr_datasets


def flower_partition(
    targets: np.ndarray,
    target_indices: np.ndarray,
    label_set: set,
    client_num: int,
    flower_partitioner_class: str,
    flower_partitioner_kwargs: str,
    partition: dict,
    stats: dict,
):
    target_indices = [i for i in range(len(target_indices)) if targets[i] in label_set]
    targets = targets[target_indices]
    data = {"data_indices": target_indices, "label": targets}

    # Create a Hugging Face Dataset
    dataset = datasets.Dataset.from_dict(data)

    flower_partitioner_kwargs = json.loads(flower_partitioner_kwargs)
    partitioner_class = partitioner_class_from_flwr_datasets(flower_partitioner_class)
    partitioner = partitioner_class(
        num_partitions=client_num, **flower_partitioner_kwargs
    )

    # Assign the dataset to the partitioner
    partitioner.dataset = dataset
    num_samples = []

    # Print each partition and the samples it contains
    for i in range(client_num):
        partition_i = partitioner.load_partition(i)
        indices = partition_i["data_indices"]
        partition["data_indices"][i] = indices
        stats[i] = {"x": None, "y": None}
        stats[i]["x"] = len(indices)
        stats[i]["y"] = dict(Counter(targets[indices].tolist()))
        num_samples.append(len(partition_i))

    num_samples = np.array(num_samples)
    stats["samples_per_client"] = {
        "std": num_samples.mean().item(),
        "stddev": num_samples.std().item(),
    }
