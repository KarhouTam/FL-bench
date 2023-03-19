# Generating Federeated dataset

This benckmark offers 3 methods to partition dataset.


- *Dirichlet* (`-a`): Refer to [Measuring the Effects of Non-Identical Data Distribution for Federated Visual Classification (*FedAvgM*)](https://arxiv.org/abs/1909.06335). Dataset would be splitted according to $Dir$(`-a`). Smaller `-a`, more severe Non-IID.

- *Shards* (`-s`): Refer to [Communication-Efficient Learning of Deep Networks from Decentralized Data (*FedAvg*)](https://arxiv.org/abs/1602.05629). The whole dataset would be evenly splitted into many equal-size shards, and each client would be allocated `-s` shards.

- *Randomly assign classes* (`-c`): Each client would be allocated data that belongs to `-c` classes. And classes for each client are randomly choosed.


## Download Datasets

Most of the datasets supported by this benchmark are integrated into `torchvision.datasets`. So there is no extra work to be done for downloading.

This benchmark also integrates [*LEAF*](https://github.com/TalwalkarLab/leaf), and supports *FEMNIST*, *CelebA*. For these datasets, this benchmark does not partition them further.

Expect *Tiny-ImageNet-200*, *Covid-19*, *Organ-S/A/CMNIST*. For these datasets, I prepare download scripts (at [`data/download`](https://github.com/KarhouTam/FL-bench/blob/master/data/download)) for you. 🤗

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

The command above splits the *CIFAR-10* dataset into 100 subsets (for 100 clients) according to $Dir(0.1)$.

## Arguments
📢 All arguments have default value.
| Arguments for general datasets | Description                                                                                                                                             |
| ------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `--dataset`, `-d`              | The name of dataset you wanna partition.                                                                                                                |
| `--iid`                        | Non-zero value for randomly partitioning data and disabling all other Non-IID partition methods.                                                        |
| `--client_num_in_total`, `-cn` | The number of clients.                                                                                                                                  |
| `--split`                      | Chooses from `[sample, user]`.  `user`: partition clients into train-test groups; `sample`: partition each client's data samples into train-test groups |
| `--fraction`                   | Propotion of train data/clients (depends on `--split`).                                                                                                 |
| `--classes`, `-c`              | Number of classes that one client's data belong to.                                                                                                     |
| `--alpha`, `-a`                | Controls data heterogeneity degree while performing Dirichlet partition.                                                                                |
| `--least_samples`              | Specifies the minimum number of data samples that each client should hold, specifically for Dirichlet partitioning.                                     |
| `--shards`, `-s`               | Number of data shards that each client holds. The same partition method as in *FedAvg.*                                                                 |

🤖 This benchmark also supports *synthetic datasets* from [(Li et al., 2020)](https://arxiv.org/abs/1812.06127). The  matched arguments are `[--gamma, --beta, --dimension]`.

⭐ For *CIFAR-100* specifically, this benchmark supports partitioning it into the superclass category (*CIFAR-100*'s 100 classes can also be classified into 20 superclasses) by setting `--super_class` to non-zero.



## Acknowledgement

[`data/femnist`](https://github.com/KarhouTam/FL-bench/tree/master/data/femnist), [`data/celeba`](https://github.com/KarhouTam/FL-bench/tree/master/data/celeba), [`data/leaf_utils`](https://github.com/KarhouTam/FL-bench/tree/master/data/leaf_utils) are copied from [*LEAF*](https://github.com/TalwalkarLab/leaf) with subtle modifications to be integrated into this benchmark. [`data/femnist/README.md`](https://github.com/KarhouTam/FL-bench/tree/master/data/femnist#readme) and [`data/celeba/README.md`](https://github.com/KarhouTam/FL-bench/tree/master/data/celeba#readme) for full partition details.

About *Tiny-ImageNet-200*, because the data in the test set are unlabeled, so the test set is not used.

