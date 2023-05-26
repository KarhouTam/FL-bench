import pickle
import random
import os
import json
import sys
from pathlib import Path

import torch

CURRENT_DIR = Path(__file__).parent.absolute()

sys.path.append(CURRENT_DIR.parent.as_posix())


if __name__ == "__main__":
    if not os.path.isdir(CURRENT_DIR / "raw"):
        raise RuntimeError(
            "Using `data/download/domain.sh` to download the dataset first."
        )

    class_num = int(input("How many classes you want (1 ~ 345): "))
    seed = input("Fix the random seed (42 by default): ")
    img_size = input("What image size you want (64 by default): ")
    ratio = input("Ratio of data in each class you gather (100 by default): ")
    client_num_foreach_domain = input(
        "How many client share one domain dataset (1 by default): "
    )
    seed = 42 if not seed else int(seed)
    img_size = 64 if not img_size else int(img_size)
    ratio = 1 if not ratio else float(ratio) / 100
    client_num_foreach_domain = (
        1 if not client_num_foreach_domain else int(client_num_foreach_domain)
    )

    random.seed(seed)
    torch.manual_seed(seed)
    if not (1 <= class_num <= 345):
        raise ValueError(f"Invalid value of class num {class_num}")

    domains = ["clipart", "infograph", "painting", "quickdraw", "real", "sketch"]

    classes = os.listdir(CURRENT_DIR / "raw" / "clipart")
    selected_classes = sorted(random.sample(sorted(classes), class_num))
    target_mapping = dict(zip(selected_classes, range(class_num)))

    # record each domain's data indices
    original_partition = []

    stats = {
        i: {"x": 0, "y": {c: 0 for c in range(len(selected_classes))}}
        for i in range(len(domains))
    }

    targets = []
    filename_list = []
    old_count = 0
    new_count = 0

    for i, domain in enumerate(domains):
        for c, cls in enumerate(selected_classes):
            folder = CURRENT_DIR / "raw" / domain / cls
            filenames = sorted(os.listdir(folder))
            if 0 < ratio < 1:
                filenames = random.sample(filenames, int(len(filenames) * ratio))
            for name in filenames:
                filename_list.append(str(folder / name))
                targets.append(target_mapping[cls])
                stats[i]["x"] += 1
                stats[i]["y"][c] += 1
                new_count += 1
        print(f"Indices of data from {domain} [{old_count}, {new_count})")
        data_idxs = list(range(old_count, new_count))
        subset_size = len(data_idxs) // client_num_foreach_domain
        for j in range(client_num_foreach_domain):
            subset_idxs = random.sample(data_idxs, subset_size)
            data_idxs = list(set(data_idxs) - set(subset_idxs))
            original_partition.append(subset_idxs)
        # If data_idxs % client_num > 0, residual indices would be allocated to the final client
        if len(data_idxs) > 0:
            original_partition[-1].extend(data_idxs)
        old_count = new_count

    torch.save(torch.tensor(targets, dtype=torch.long), "targets.pt")

    with open("original_partition.pkl", "wb") as f:
        pickle.dump(original_partition, f)

    with open("filename_list.pkl", "wb") as f:
        pickle.dump(filename_list, f)

    with open("metadata.json", "w") as f:
        json.dump(
            {
                "class_num": class_num,
                "client_num": client_num_foreach_domain * 6,
                "data_amount": len(targets),
                "image_size": img_size,
                "classes": {
                    cls: sum([stats[i]["y"][c] for i in range(len(domains))])
                    for c, cls in enumerate(selected_classes)
                },
                "seed": seed,
            },
            f,
        )

    with open("original_stats.json", "w") as f:
        json.dump(stats, f)

    os.system(f"cd ../utils; python run.py -d domain")
