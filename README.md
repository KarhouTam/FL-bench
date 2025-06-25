<p align="center">
<img src=".github/images/logo.svg" alt="Image"/>
</p>
<!-- 
<p align="center">
  <a href="https://github.com/KarhouTam/FL-bench/blob/master/LICENSE">
    <img alt="GitHub License" src="https://img.shields.io/github/license/KarhouTam/FL-bench?style=for-the-badge&logo=github&color=8386e0"/>
  </a>
  <a href="https://github.com/KarhouTam/FL-bench/issues?q=is%3Aissue+is%3Aclosed">
    <img alt="GitHub closed issues" src="https://img.shields.io/github/issues-closed-raw/KarhouTam/FL-bench?style=for-the-badge&logo=github&color=8386e0">
  </a>
  <a href="https://github.com/KarhouTam/FL-bench/stargazers">
    <img alt="GitHub Repo stars" src="https://img.shields.io/github/stars/KarhouTam/FL-bench?style=for-the-badge&logo=github&color=8386e0">
  </a>
  <a href="https://github.com/KarhouTam/FL-bench/forks">
    <img alt="GitHub Repo forks" src="https://img.shields.io/github/forks/KarhouTam/FL-bench?style=for-the-badge&logo=github&color=8386e0">
  </a>
</p>
   -->
<h4 align="center"><i>

Benchmarking Federated Learning Methods.

Realizing Your Brilliant Ideas.

Having Fun with Federated Learning.

FL-bench welcomes PR on everything that can make this project better.

</i></h4>

<p align="center">
<a href=https://zhuanlan.zhihu.com/p/703576051>FL-bench 的简单介绍</a>
</p>


## Methods 🧬



<!-- <details> -->
<summary><b>Traditional FL Methods</b></summary>

- ***FedAvg*** -- [Communication-Efficient Learning of Deep Networks from Decentralized Data](https://arxiv.org/abs/1602.05629) (AISTATS'17)
- ***FedAvgM*** -- [Measuring the Effects of Non-Identical Data Distribution for Federated Visual Classification](https://arxiv.org/abs/1909.06335) (ArXiv'19)
- ***FedProx*** -- [Federated Optimization in Heterogeneous Networks](https://arxiv.org/abs/1812.06127) (MLSys'20)
- ***SCAFFOLD*** -- [SCAFFOLD: Stochastic Controlled Averaging for Federated Learning](https://arxiv.org/abs/1910.06378) (ICML'20)
- ***MOON*** -- [Model-Contrastive Federated Learning](http://arxiv.org/abs/2103.16257) (CVPR'21)
- ***FedDyn*** -- [Federated Learning Based on Dynamic Regularization](http://arxiv.org/abs/2111.04263) (ICLR'21)
- ***FedLC*** -- [Federated Learning with Label Distribution Skew via Logits Calibration](http://arxiv.org/abs/2209.00189) (ICML'22)
- ***FedGen*** -- [Data-Free Knowledge Distillation for Heterogeneous Federated Learning](https://arxiv.org/abs/2105.10056) (ICML'21)
- ***CCVR*** -- [No Fear of Heterogeneity: Classifier Calibration for Federated Learning with Non-IID Data](https://arxiv.org/abs/2106.05001) (NeurIPS'21)
- ***FedOpt*** -- [Adaptive Federated Optimization](https://arxiv.org/abs/2003.00295) (ICLR'21)
- ***FedADMM*** -- [FedADMM: A robust federated deep learning framework with adaptivity to system heterogeneity](https://ieeexplore.ieee.org/abstract/document/9835545) (ICDE'22)
- ***Elastic Aggregation*** -- [Elastic Aggregation for Federated Optimization](https://openaccess.thecvf.com/content/CVPR2023/html/Chen_Elastic_Aggregation_for_Federated_Optimization_CVPR_2023_paper.html) (CVPR'23)
- ***FedFed*** -- [FedFed: Feature Distillation against Data Heterogeneity in Federated Learning](http://arxiv.org/abs/2310.05077) (NeurIPS'23)

<!-- </details> -->

<!-- <details> -->
<summary><b>Personalized FL Methods</b></summary>

- ***pFedSim (My Work⭐)*** -- [pFedSim: Similarity-Aware Model Aggregation Towards Personalized Federated Learning](https://arxiv.org/abs/2305.15706) (ArXiv'23)
- ***Local-Only*** -- Local training only (without communication).
- ***FedMD*** -- [FedMD: Heterogenous Federated Learning via Model Distillation](http://arxiv.org/abs/1910.03581) (NeurIPS'19)
- ***APFL*** -- [Adaptive Personalized Federated Learning](http://arxiv.org/abs/2003.13461) (ArXiv'20)
- ***LG-FedAvg*** -- [Think Locally, Act Globally: Federated Learning with Local and Global Representations](https://arxiv.org/abs/2001.01523) (ArXiv'20)
- ***FedBN*** -- [FedBN: Federated Learning On Non-IID Features Via Local Batch Normalization](http://arxiv.org/abs/2102.07623) (ICLR'21)
- ***FedPer*** -- [Federated Learning with Personalization Layers](http://arxiv.org/abs/1912.00818) (AISTATS'20)
- ***FedRep*** -- [Exploiting Shared Representations for Personalized Federated Learning](http://arxiv.org/abs/2102.07078) (ICML'21)
- ***Per-FedAvg*** -- [Personalized Federated Learning with Theoretical Guarantees: A Model-Agnostic Meta-Learning Approach](https://proceedings.neurips.cc/paper/2020/hash/24389bfe4fe2eba8bf9a9203a44cdad-Abstract.html) (NeurIPS'20)
- ***pFedMe*** -- [Personalized Federated Learning with Moreau Envelopes](http://arxiv.org/abs/2006.08848) (NeurIPS'20)
- ***FedEM*** -- [Federated Multi-Task Learning under a Mixture of Distributions](https://arxiv.org/abs/2108.10252) (NIPS'21)
- ***Ditto*** -- [Ditto: Fair and Robust Federated Learning Through Personalization](http://arxiv.org/abs/2012.04221) (ICML'21)
- ***pFedHN*** -- [Personalized Federated Learning using Hypernetworks](http://arxiv.org/abs/2103.04628) (ICML'21)
- ***pFedLA*** -- [Layer-Wised Model Aggregation for Personalized Federated Learning](https://openaccess.thecvf.com/content/CVPR2022/html/Ma_Layer-Wised_Mdel_Aggregation_for_Personalized_Federated_Learning_CVPR_2022_paper.html) (CVPR'22)
- ***CFL*** -- [Clustered Federated Learning: Model-Agnostic Distributed Multi-Task Optimization under Privacy Constraints](https://arxiv.org/abs/1910.01991) (ArXiv'19)
- ***FedFomo*** -- [Personalized Federated Learning with First Order Model Optimization](http://arxiv.org/abs/2012.08565) (ICLR'21)
- ***FedBabu*** -- [FedBabu: Towards Enhanced Representation for Federated Image Classification](https://arxiv.org/abs/2106.06042) (ICLR'22)
- ***FedAP*** -- [Personalized Federated Learning with Adaptive Batchnorm for Healthcare](https://arxiv.org/abs/2112.00734) (IEEE'22)
- ***kNN-Per*** -- [Personalized Federated Learning through Local Memorization](http://arxiv.org/abs/2111.09360) (ICML'22)
- ***MetaFed*** -- [MetaFed: Federated Learning among Federations with Cyclic Knowledge Distillation for Personalized Healthcare](http://arxiv.org/abs/2206.08516) (IJCAI'22)
- ***FedRoD*** -- [On Bridging Generic and Personalized Federated Learning for Image Classification](https://arxiv.org/abs/2107.00778) (ICLR'22)
- ***FedProto*** -- [FedProto: Federated prototype learning across heterogeneous clients](https://arxiv.org/abs/2105.00243) (AAAI'22)
- ***FedPAC*** -- [Personalized Federated Learning with Feature Alignment and Classifier Collaboration](https://arxiv.org/abs/2306.11867v1) (ICLR'23)
- ***FedALA*** -- [FedALA: Adaptive Local Aggregation for Personalized Federated Learning](https://arxiv.org/abs/2212.01197) (AAAI'23)
- ***PeFLL*** -- [PeFLL: Personalized Federated Learning by Learning to Learn](https://openreview.net/forum?id=MrYiwlDRQO) (ICLR'24)
- ***FLUTE*** -- [Federated Representation Learning in the Under-Parameterized Regime](https://openreview.net/forum?id=LIQYhV45D4) (ICML'24)
- ***FedAS*** -- [FedAS: Bridging Inconsistency in Personalized Federated Learning](https://openaccess.thecvf.com/content/CVPR2024/html/Yang_FedAS_Bridging_Inconsistency_in_Personalized_Federated_Learning_CVPR_2024_paper.html) (CVPR'24)
- ***pFedFDA*** -- [pFedFDA: Personalized Federated Learning via Feature Distribution Adaptation](http://arxiv.org/abs/2411.00329) (NeurIPS 2024)
- ***Floco*** -- [Federated Learning over Connected Modes](https://openreview.net/forum?id=JL2eMCfDW8) (NeurIPS'24)
- ***FedAH*** -- [FedAH: Aggregated Head for Personalized Federated Learning](https://arxiv.org/abs/2412.01295) (ArXiv'24)
<!-- </details> -->


<!-- <details> -->
<summary><b>FL Domain Generalization Methods</b></summary>

- ***FedSR*** -- [FedSR: A Simple and Effective Domain Generalization Method for Federated Learning](https://openreview.net/forum?id=mrt90D00aQX) (NeurIPS'22)
- ***ADCOL*** -- [Adversarial Collaborative Learning on Non-IID Features](https://proceedings.mlr.press/v202/li23j.html) (ICML'23)
- ***FedIIR*** -- [Out-of-Distribution Generalization of Federated Learning via Implicit Invariant Relationships](https://openreview.net/pdf?id=JC05k0E2EM) (ICML'23)
<!-- </details> -->

## Environment Preparation 🧩

### PyPI 🐍
```sh
pip install -r .env/requirements.txt
```

### Poetry 🎶
For those China mainland users
```sh
poetry install --no-root -C .env
```
For others
```sh
cd .env && sed -i "10,14d" pyproject.toml && poetry lock --no-update && poetry install --no-root
```

### Docker 🐳

```sh
docker pull ghcr.io/karhoutam/fl-bench:master
```

An example of building container
```sh
docker run -it --name fl-bench -v path/to/FL-bench:/root/FL-bench --privileged --gpus all ghcr.io/karhoutam/fl-bench:master
```


## Easy Run 🏃‍♂️

ALL classes of methods are inherited from `FedAvgServer` and `FedAvgClient`. If you wanna figure out the entire workflow and detail of variable settings, go check [`src/server/fedavg.py`](src/server/fedavg.py) and [`src/client/fedavg.py`](src/client/fedavg.py).

### Step 1. Generate FL Dataset
Partition the MNIST according to Dir(0.1) for 100 clients
```shell
python generate_data.py -d mnist -a 0.1 -cn 100
```
About methods of generating federated dastaset, go check [`data/README.md`](data/#readme) for full details.


### Step 2. Run Experiment

```sh
python main.py [--config-path, --config-name] [method=<METHOD_NAME> args...]
```

- `method`: The algorithm's name, e.g., `method=fedavg`. 
>   \[!NOTE\]
>   `method` should be identical to the `.py` file name in `src/server`.

- `--config-path`: Relative path to the directory of the config file. Defaults to `config`.
- `--config-name`: Name of `.yaml` config file (w/o the `.yaml` extension). Defaults to `defaults`, which points to `config/defaults.yaml`.

Such as running FedAvg with all defaults. 
```sh
python main.py method=fedavg
```
Defaults are set in both [`config/defaults.yaml`](config/defaults.yaml) and [`src/utils/constants.py`](src/utils/constants.py).

### How To Customize FL method Arguments 🤖
- By modifying config file.
- By explicitly setting in CLI, e.g., `python main.py --config-name my_cfg.yaml method=fedprox fedprox.mu=0.01`.
- By modifying the default value in `config/defaults.yaml` or `get_hyperparams()` in `src/server/<method>.py`

> \[!NOTE\]
> For the same FL method argument, the priority of argument setting is **CLI > Config file > Default value**. 
> 
> For example, the default value of `fedprox.mu` is `1`, 
> ```python
> # src/server/fedprox.py
> class FedProxServer(FedAvgServer):
> 
>     @staticmethod
>     def get_hyperparams(args_list=None) -> Namespace:
>         parser = ArgumentParser()
>         parser.add_argument("--mu", type=float, default=1.0)
>         return parser.parse_args(args_list)
> 
> ```
> and your `.yaml` config file has
> ```yaml
> # config/your_config.yaml
> ...
> fedprox:
>   mu: 0.01
> ```
> 
> ```shell
> python main.py method=fedprox                                            # fedprox.mu = 1
> python main.py --config-name your_config method=fedprox                  # fedprox.mu = 0.01
> python main.py --config-name your_config method=fedprox fedprox.mu=0.001 # fedprox.mu = 0.001
> ``` 

### Monitor 📈
FL-bench supports `visdom` and `tensorboard`.

#### Activate

```yaml
# your_config.yaml
common:
  ...
  monitor: tensorboard # options: [null, visdom, tensorboard]
```
> \[!NOTE\]
> You needs to launch `visdom` / `tensorboard` server by yourself.

#### Launch `visdom` / `tensorboard` Server

##### `visdom`
1. Run `python -m visdom.server` on terminal.
2. Go check `localhost:8097` on your browser.

#### `tensorboard`
1. Run `tensorboard --logdir=<your_log_dir>` on terminal.
2. Go check `localhost:6006` on your browser.

## Parallel Training via `Ray` 🚀
This feature can **vastly improve your training efficiency**. At the same time, this feature is user-friendly and easy to use!!!
### Activate (What You ONLY Need To Do)
```yaml
# your_config.yaml
mode: parallel
parallel:
  num_workers: 2 # any positive integer that larger than 1
  ...
...
```
### Manually Create `Ray` Cluster (Optional)
A `Ray` cluster would be created implicitly everytime you run experiment in parallel mode.

> \[!TIP\]
> You can create it manually by the command shown below to avoid creating and destroying cluster every time you run experiment.
> ```shell
> ray start --head [OPTIONS]
> ```

> \[!NOTE\]
> You need to keep `num_cpus: null` and `num_gpus: null` in your config file for connecting to a existing `Ray` cluster.
> 
> ```yaml
> # your_config_file.yaml
> # Connect to an existing Ray cluster in localhost.
> mode: parallel
> parallel:
>   ...
>   num_gpus: null
>   num_cpus: null
> ...
> ```





## Arguments 🔧

FL-bench highly recommend through config file to customize your FL method and experiment settings. 

FL-bench offers a default config file [`config/defaults.yaml`](config/defaults.yaml) that contains all required arguments and corresponding comments.

All common arguments have their default value. Go check [`config/defaults.yaml`](config/defaults.yaml) or [`DEFAULTS`](src/utils/constants.py) in `src/utils/constants.py` for all argument defaults.

> \[!NOTE\]
> If your custom config file does not contain all required arguments, FL-bench will fill those missing arguments with their defaults that loaded from [`DEFAULTS`](src/utils/constants.py).

About the default values of specific FL method arguments, go check corresponding `src/server/<method>.py` for the full details.

> \[!TIP\]
> FL-bench also supports CLI arguments for quick changings. Here are some examples:
> ```
> # Using config/defaults.yaml but change the method to FedProx and set its mu to 0.1.
> python main.py method=fedprox fedprox.mu=0.1
> 
> # Change learning rate to 0.1.
> python main.py optimizer.lr=0.1
> 
> # Change batch size to 128.
> python main.py common.batch_size=128
> ```




## Models 🤖

This benchmark supports bunch of models that common and integrated in Torchvision (check [here](src/utils/models.py) for all):

- ResNet family
- EfficientNet family
- DenseNet family
- MobileNet family
- LeNet5
- ...


> \[!TIP\]
> You can define your own custom model by filling the `CustomModel` class in [`src/utils/models.py`](src/utils/models.py) and use it by defining `model: custom` in your `.yaml` config file.

## Datasets and [Partition Strategies](data/README.md) 🎨

Regular Image Datasets

- *MNIST* (1 x 28 x 28, 10 classes)

- *CIFAR-10/100* (3 x 32 x 32, 10/100 classes)

- *EMNIST* (1 x 28 x 28, 62 classes)

- *FashionMNIST* (1 x 28 x 28, 10 classes)

- [*Syhthetic Dataset*](https://arxiv.org/abs/1812.06127)

- [*FEMNIST*](https://leaf.cmu.edu/) (1 x 28 x 28, 62 classes)

- [*CelebA*](https://leaf.cmu.edu/) (3 x 218 x 178, 2 classes)

- [*SVHN*](http://ufldl.stanford.edu/housenumbers/) (3 x 32 x 32, 10 classes)

- [*USPS*](https://ieeexplore.ieee.org/document/291440) (1 x 16 x 16, 10 classes)

- [*Tiny-ImageNet-200*](https://arxiv.org/pdf/1707.08819.pdf) (3 x 64 x 64, 200 classes)

- [*CINIC-10*](https://datashare.ed.ac.uk/handle/10283/3192) (3 x 32 x 32, 10 classes)

Domain Generalization Image Datasets

- [*DomainNet*](http://ai.bu.edu/DomainNet/) (3 x ? x ?, 345 classes) 
  - Go check [`data/README.md`](data#readme) for the full process guideline 🧾.
  
Medical Image Datasets

- [*COVID-19*](https://www.researchgate.net/publication/344295900_Curated_Dataset_for_COVID-19_Posterior-Anterior_Chest_Radiography_Images_X-Rays) (3 x 244 x 224, 4 classes)

- [*Organ-S/A/CMNIST*](https://medmnist.com/) (1 x 28 x 28, 11 classes)

## Customization Tips 💡

### Implementing FL Method

The `package()` at server-side class is used for assembling all parameters server need to send to clients. Similarly, `package()` at client-side class is for parameters clients need to send back to server. You should always has `super().package()` in your override implementation. 

- Consider to inherit your method classes from [`FedAvgServer`](src/server/fedavg.py) and [`FedAvgClient`](src/client/fedavg.py) for maximum utilizing FL-bench's workflow.

- You can also inherit your method classes from advanced methods, e.g., FedBN, FedProx, etc. Which will inherit all functions, variables and hyperparamter settings. If you do that, you need to careful design your method in order to avoid potential hyperparamters and workflow conflicts.
```python
class YourServer(FedBNServer):
  ...

class YourClient(FedBNClient):
  ...
```


- For customizing your server-side process, consider to override the `package()` and `aggregate_client_updates()`.

- For customizing your client-side training, consider to override the `fit()`, `set_parameters()` and `package()`.

You can find all details in [`FedAvgClient`](src/client/fedavg.py) and [`FedAvgServer`](src/server/fedavg.py), which are the bases of all implementations in FL-bench.

### Integrating Dataset

- Inherit your own dataset class from `BaseDataset` in [`data/utils/datasets.py`](data/utils/datasets.py) and add your class in dict `DATASETS`. Highly recommend to refer to the existing dataset classes for guidance.

### Customizing Model

- I offer the `CustomModel` class in [`src/utils/models.py`](src/utils/models.py) and you just need to define your model arch.
- If you want to use your customized model within FL-bench's workflow, the `base` and `classifier` must be defined. (Tips: You can define one of them as `torch.nn.Identity()` for bypassing it.)

## Citation 🧐

```bibtex
@software{Tan_FL-bench,
  author = {Tan, Jiahao and Wang, Xinpeng},
  license = {GPL-3.0},
  title = {{FL-bench: A federated learning benchmark for solving image classification tasks}},
  url = {https://github.com/KarhouTam/FL-bench}
}
```

```bibtex
@misc{tan2023pfedsim,
  title={pFedSim: Similarity-Aware Model Aggregation Towards Personalized Federated Learning}, 
  author={Jiahao Tan and Yipeng Zhou and Gang Liu and Jessie Hui Wang and Shui Yu},
  year={2023},
  eprint={2305.15706},
  archivePrefix={arXiv},
  primaryClass={cs.LG}
}

```
