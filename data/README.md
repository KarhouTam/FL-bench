# Generating Federeated dataset

This benckmark offers 3 methods to partition dataset.


- *Dirichlet* (`-a`): Refer to [Measuring the Effects of Non-Identical Data Distribution for Federated Visual Classification (*FedAvgM*)](https://arxiv.org/abs/1909.06335). Dataset would be splitted according to $Dir$(`-a`). Smaller `-a`, more severe Non-IID.

- *Shards* (`-s`): Refer to [Communication-Efficient Learning of Deep Networks from Decentralized Data (*FedAvg*)](https://arxiv.org/abs/1602.05629). The whole dataset would be evenly splitted into many equal-size shards, and each client would be allocated `-s` shards.

- *Randomly assign classes* (`-c`): Each client would be allocated data that belongs to `-c` classes. And classes for each client are randomly choosed.


👉 Go check [`data/utils/run.py`](https://github.com/KarhouTam/FL-bench/blob/master/data/utils/run.py) for more details of arguments.


## Download Datasets

Most of the datasets supported by this benchmark are integrated into `torchvision.datasets`. So there is no extra work to be done for downloading.

This benchmark also integrates [*LEAF*](https://github.com/TalwalkarLab/leaf), and supports *FEMNIST*, *CelebA*. For these datasets, this benchmark does not partition them further.

Expect *Tiny-Imagenet-200*, *Covid-19*, *Organ-S/A/CMNIST*. For these datasets, I prepare download scripts (at [`data/download`](https://github.com/KarhouTam/FL-bench/blob/master/data/download)) for you. 🤗

e.g.

```shell
cd data/download
sh tiny_imagenet.sh
```

## Usage

e.g.

```shell
cd data/utils
python run.py -d cifar10 -a 0.1 -cn 100
```

The command above splits the *CIFAR-10* dataset into 100 pieces (for 100 clients) according to $Dir(0.1)$.

You can set `--iid 1` to split the dataset in IID. If `--iid` value is non-zero, `-a`, `-s`, `-c` are disabled.

## Acknowledgement

[`data/femnist`](https://github.com/KarhouTam/FL-bench/tree/master/data/femnist), [`data/celeba`](https://github.com/KarhouTam/FL-bench/tree/master/data/celeba), [`data/leaf_utils`](https://github.com/KarhouTam/FL-bench/tree/master/data/leaf_utils) are copied from [*LEAF*](https://github.com/TalwalkarLab/leaf) with subtle modifications to be integrated into this benchmark. [`data/femnist/README.md`](https://github.com/KarhouTam/FL-bench/tree/master/data/femnist#readme) and [`data/celeba/README.md`](https://github.com/KarhouTam/FL-bench/tree/master/data/celeba#readme) for full partition details.

About *Tiny-Imagenet-200*, because the data in the test set are unlabeled, so the test set is not used and the val set is considered as the test set in FL.

