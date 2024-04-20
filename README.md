<p align="center">
<img src=".github/images/logo.svg" alt="Image"/>
</p>

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
<h4 align="center"><i>
  
Evaluating Federated Learning Methods.

Realizing Your Brilliant Ideas.

Having Fun with Federated Learning.

</i></h4>

<h5 align="center"><i>🎉 FL-bench now can perform FL training in parallel (with the help of <a href https://github.com/ray-project/ray>ray</a>)！ 🎉</i></h5> 



## Methods 🧬



<details>
<summary><b>Traditional FL Methods</b></summary>

- ***FedAvg*** -- [Communication-Efficient Learning of Deep Networks from Decentralized Data](https://arxiv.org/abs/1602.05629) (AISTATS'17)

- ***FedAvgM*** -- [Measuring the Effects of Non-Identical Data Distribution for Federated Visual Classification](https://arxiv.org/abs/1909.06335) (ArXiv'19)

- ***FedProx*** -- [Federated Optimization in Heterogeneous Networks](https://arxiv.org/abs/1812.06127) (MLSys'20)

- ***SCAFFOLD*** -- [SCAFFOLD: Stochastic Controlled Averaging for Federated Learning](https://arxiv.org/abs/1910.06378) (ICML'20)

- ***MOON*** -- [Model-Contrastive Federated Learning](http://arxiv.org/abs/2103.16257) (CVPR'21)
  
- ***FedDyn*** -- [Federated Learning Based on Dynamic Regularization](http://arxiv.org/abs/2111.04263) (ICLR'21)

- ***FedLC*** -- [Federated Learning with Label Distribution Skew via Logits Calibration](http://arxiv.org/abs/2209.00189) (ICML'22)

- ***FedGen*** -- [Data-Free Knowledge Distillation for Heterogeneous Federated Learning](https://arxiv.org/abs/2105.10056) (ICML'21)

- ***CCVR*** -- [No Fear of Heterogeneity: Classifier Calibration for Federated Learning with Non-IID Data](https://arxiv.org/abs/2106.05001) (NIPS'21)

- ***FedOpt*** -- [Adaptive Federated Optimization](https://arxiv.org/abs/2003.00295) (ICLR'21)

- ***Elastic Aggregation*** -- [Elastic Aggregation for Federated Optimization](https://openaccess.thecvf.com/content/CVPR2023/html/Chen_Elastic_Aggregation_for_Federated_Optimization_CVPR_2023_paper.html) (CVPR'23)
</details>

<details>
<summary><b>Personalized FL Methods</b></summary>

- ***pFedSim (My Work⭐)*** -- [pFedSim: Similarity-Aware Model Aggregation Towards Personalized Federated Learning](https://arxiv.org/abs/2305.15706) (ArXiv'23)

- ***Local-Only*** -- Local training only (without communication).

- ***FedMD*** -- [FedMD: Heterogenous Federated Learning via Model Distillation](http://arxiv.org/abs/1910.03581) (NIPS'19)

- ***APFL*** -- [Adaptive Personalized Federated Learning](http://arxiv.org/abs/2003.13461) (ArXiv'20)

- ***LG-FedAvg*** -- [Think Locally, Act Globally: Federated Learning with Local and Global Representations](https://arxiv.org/abs/2001.01523) (ArXiv'20)

- ***FedBN*** -- [FedBN: Federated Learning On Non-IID Features Via Local Batch Normalization](http://arxiv.org/abs/2102.07623) (ICLR'21)

- ***FedPer*** -- [Federated Learning with Personalization Layers](http://arxiv.org/abs/1912.00818) (AISTATS'20)

- ***FedRep*** -- [Exploiting Shared Representations for Personalized Federated Learning](http://arxiv.org/abs/2102.07078) (ICML'21)

- ***Per-FedAvg*** -- [Personalized Federated Learning with Theoretical Guarantees: A Model-Agnostic Meta-Learning Approach](https://proceedings.neurips.cc/paper/2020/hash/24389bfe4fe2eba8bf9aa9203a44cdad-Abstract.html) (NIPS'20)

- ***pFedMe*** -- [Personalized Federated Learning with Moreau Envelopes](http://arxiv.org/abs/2006.08848) (NIPS'20)

- ***Ditto*** -- [Ditto: Fair and Robust Federated Learning Through Personalization](http://arxiv.org/abs/2012.04221) (ICML'21)

- ***pFedHN*** -- [Personalized Federated Learning using Hypernetworks](http://arxiv.org/abs/2103.04628) (ICML'21)
  
- ***pFedLA*** -- [Layer-Wised Model Aggregation for Personalized Federated Learning](https://openaccess.thecvf.com/content/CVPR2022/html/Ma_Layer-Wised_Model_Aggregation_for_Personalized_Federated_Learning_CVPR_2022_paper.html) (CVPR'22)

- ***CFL*** -- [Clustered Federated Learning: Model-Agnostic Distributed Multi-Task Optimization under Privacy Constraints](https://arxiv.org/abs/1910.01991) (ArXiv'19)

- ***FedFomo*** -- [Personalized Federated Learning with First Order Model Optimization](http://arxiv.org/abs/2012.08565) (ICLR'21)

- ***FedBabu*** -- [FedBabu: Towards Enhanced Representation for Federated Image Classification](https://arxiv.org/abs/2106.06042) (ICLR'22)

- ***FedAP*** -- [Personalized Federated Learning with Adaptive Batchnorm for Healthcare](https://arxiv.org/abs/2112.00734) (IEEE'22)

- ***kNN-Per*** -- [Personalized Federated Learning through Local Memorization](http://arxiv.org/abs/2111.09360) (ICML'22)

- ***MetaFed*** -- [MetaFed: Federated Learning among Federations with Cyclic Knowledge Distillation for Personalized Healthcare](http://arxiv.org/abs/2206.08516) (IJCAI'22)

- ***FedRoD*** -- [On Bridging Generic and Personalized Federated Learning for Image Classification](https://arxiv.org/abs/2107.00778) (ICLR'22)

- ***FedProto*** -- [FedProto: Federated prototype learning across heterogeneous clients](https://arxiv.org/abs/2105.00243) (AAAI'22)

</details>


<details>
<summary><b>FL Domain Generalization Methods</b></summary>

- ***FedSR*** -- [FedSR: A Simple and Effective Domain Generalization Method for Federated Learning](https://openreview.net/forum?id=mrt90D00aQX) (NIPS'22)
  
- ***ADCOL*** -- [Adversarial Collaborative Learning on Non-IID Features](https://proceedings.mlr.press/v202/li23j.html) (ICML'23)
  
- ***FedIIR*** -- [Out-of-Distribution Generalization of Federated Learning via Implicit Invariant Relationships](https://openreview.net/pdf?id=JC05k0E2EM) (ICML'23)

</details>

## Environment Preparation 🧩

Just select one of them.

### PyPI 🐍
```sh
pip install -r .environment/requirements.txt
```

### Conda 💻
```sh
conda env create -f .environment/environment.yml
```

### Poetry 🎶

```sh
# For those China mainland users
cd .environment && poetry install --no-root

# For those oversea users
cd .environment && sed -i "10,14d" pyproject.toml && poetry lock --no-update && poetry install --no-root
```

### Docker 🐳

```shell
# For those China mainland users
docker pull registry.cn-hangzhou.aliyuncs.com/karhoutam/fl-bench:master

# For those oversea users
docker pull ghcr.io/karhoutam/fl-bench:master
# or
docker pull docker.io/karhoutam/fl-bench:master

# An example for building container
docker run -it --name fl-bench -v path/to/FL-bench:/root/FL-bench --privileged --gpus all ghcr.io/karhoutam/fl-bench:master
```


## Easy Run 🏃‍♂️

ALL classes of methods are inherited from `FedAvgServer` and `FedAvgClient`. If you wanna figure out the entire workflow and detail of variable settings, go check [`src/server/fedavg.py`](src/server/fedavg.py) and [`src/client/fedavg.py`](src/client/fedavg.py).

### Step 1. Generate FL Dataset
```shell
# Partition the MNIST according to Dir(0.1) for 100 clients
python generate_data.py -d mnist -a 0.1 -cn 100
```
About methods of generating federated dastaset, go check [`data/README.md`](data/#readme) for full details.


### Step 2. Run Experiment
`python main.py <method> [your_config_file.yml] [method_args...]`

❗ Method name should be identical to the `.py` file name in `src/server`.

```
# Run FedAvg with default settings. 
python main.py fedavg
```


### How To Customize Experiment Arguments 🤖
- By modifying config file
- By explicitly setting in CLI, e.g., `python main.py fedprox config/my_cfg.yml --mu 0.01`.
- By modifying the default value in `src/utils/constants.py/DEFAULT_COMMON_ARGS` or `src/server/<method>.py/get_<method>_args()`

⚠ For the same FL method argument, the priority of argument setting is **CLI > Config file > Default value**. 

For example, the default value of `fedprox.mu` is `1`, 
```python
def get_fedprox_args(args_list=None) -> Namespace:
    parser = ArgumentParser()
    parser.add_argument("--mu", type=float, default=1.0)
    return parser.parse_args(args_list)
```
and you set
```yaml
# your_config.yml
...
fedprox:
  mu: 0.01
```
in your config file. If you run
```shell
python main.py fedprox                           # fedprox.mu = 1
python main.py fedprox your_config.yml           # fedprox.mu = 0.01
python main.py fedprox your_config.yml --mu 10   # fedprox.mu = 10
``` 

### Monitor 📈
1. Run `python -m visdom.server` on terminal.
2. Set `visible` as `true`.
3. Go check `localhost:8097` on your browser.

### Using `Ray` for Parallel Training
You need to set
```yaml
# your_config_file.yml
mode: parallel
parallel:
  num_workers: 2 # any positive integer that larger than 1
  ...
...
```
for parallel training, which will **vastly improve your training efficiency**.


#### Creating a `Ray` Cluster
A `Ray` cluster would be created implicitly by `python main.py <method> ...`.
Or you can manually launch it to avoid creating cluster each time by running experiment.
```yaml
# your_config_file.yml
mode: parallel
parallel:
  ray_cluster_addr: null
  ...
...
```

```shell
ray start --head [OPTIONS]
```




## Arguments 🔧

All common arguments have their default value. Go check [`DEFAULT_COMMON_ARGS`](src/utils/constants.py) in `src/utils/constants.py` for full details of common arguments. 

⚠ Common arguments cannot be set via CLI.

You can also write your own `.yml` config file. I offer you a [template](config/template.yml) in `config` and recommend you to save your config files there also. 

One example: `python main.py fedavg config/template.yaml [cli_method_args...]`

About the default values of specific FL method arguments, go check corresponding `FL-bench/src/server/<method>.py` for the full details.
| Arguments                    | Type    | Description                                                                                                                                                                                                                                  |
| ---------------------------- | ------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `dataset`                    | `str`   | The name of dataset that experiment run on.                                                                                                                                                                                                  |
| `model`                      | `str`   | The model backbone experiment used.                                                                                                                                                                                                          |
| `seed`                       | `int`   | Random seed for running experiment.                                                                                                                                                                                                          |
| `join_ratio`                 | `float` | Ratio for (client each round) / (client num in total).                                                                                                                                                                                       |
| `global_epoch`               | `int`   | Global epoch, also called communication round.                                                                                                                                                                                               |
| `local_epoch`                | `int`   | Local epoch for client local training.                                                                                                                                                                                                       |
| `finetune_epoch`             | `int`   | Epoch for clients fine-tunning their models before test.                                                                                                                                                                                     |
| `test_interval`              | `int`   | Interval round of performing test on clients.                                                                                                                                                                                                |
| `eval_test`                  | `bool`  | Non-zero value for performing evaluation on joined clients' testset before and after local training.                                                                                                                                         |
| `eval_val`                   | `bool`  | Non-zero value for performing evaluation on joined clients' valset before and after local training.                                                                                                                                          |
| `eval_train`                 | `bool`  | Non-zero value for performing evaluation on joined clients' trainset before and after local training.                                                                                                                                        |
| `optimizer`                  | `dict`  | Client-side optimizer.  Argument request is the same as Optimizers in `torch.optim`.                                                                                                                                                         |
| `lr_scheduler`                  | `dict`  | Client-side learning rate scheduler.  Argument request is the same as schedulers in `torch.optim.lr_scheduler`.                                                                                                                                                         |
| `verbose_gap`                | `int`   | Interval round of displaying clients training performance on terminal.                                                                                                                                                                       |
| `batch_size`                 | `int`   | Data batch size for client local training.                                                                                                                                                                                                   |
| `use_cuda`                   | `bool`  | Non-zero value indicates that tensors are in gpu.                                                                                                                                                                                            |
| `visible`                    | `bool`  | Non-zero value for using Visdom to monitor algorithm performance on `localhost:8097`.                                                                                                                                                        |
| `straggler_ratio`            | `float` | The ratio of stragglers (set in `[0, 1]`). Stragglers would not perform full-epoch local training as normal clients. Their local epoch would be randomly selected from range `[straggler_min_local_epoch, local_epoch)`.                     |
| `straggler_min_local_epoch`  | `int`   | The minimum value of local epoch for stragglers.                                                                                                                                                                                             |
| `external_model_params_file` | `str`   | The relative file path of external model parameters. Please confirm whether the shape of parameters compatible with the model by yourself. ⚠ This feature is enabled only when `unique_model=False`, which is pre-defined by each FL method. |
| `save_log`                   | `bool`  | Non-zero value for saving algorithm running log in `out/<method>/<start_time>`.                                                                                                                                                              |
| `save_model`                 | `bool`  | Non-zero value for saving output model(s) parameters in `out/<method>/<start_time>`.pt`.                                                                                                                                                     |
| `save_fig`                   | `bool`  | Non-zero value for saving the accuracy curves showed on Visdom into a `.pdf` file at `out/<method>/<start_time>`.                                                                                                                            |
| `save_metrics`               | `bool`  | Non-zero value for saving metrics stats into a `.csv` file at `out/<method>/<start_time>`.                                                                                                                                                   |
| `viz_win_name`               | `str`   | Custom visdom window name (active when setting `visible` as a non-zero value).                                                                                                                                                               |
| `check_convergence`          | `bool`  | Non-zero value for checking convergence after training.                                                                                                                                                                                      |

### Arguments of Parallel Training 👯‍♂️

| Arguments                 | Type  | Description                                                                                                                                                                                                                                                                              |
| ------------------------- | ----- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `num_workers`             | `int` | The number of parallel workers. Need to be set as an integer that larger than `1`.                                                                                                                                                                                                       |
| `ray_cluster_addr`        | `str` | The IP address of the selected ray cluster. Default as `null`, which means `ray` will build a new cluster everytime you running an experiment and destroy it at the end. More details can be found in the [official docs](https://docs.ray.io/en/latest/ray-core/api/doc/ray.init.html). |
| `num_cpus` and `num_gpus` | `int` | The amount of computational resources you allocate. Default as `null`, which means all.                                                                                                                                                                                                  |



## Supported Models 🚀

This benchmark supports bunch of models that common and integrated in Torchvision:

- ResNet family
- EfficientNet family
- DenseNet family
- MobileNet family
- LeNet5
...

🤗 You can define your own custom model by filling the `CustomModel` class in [`src/utils/models.py`](src/utils/models.py) and use it by setting `model` to `custom` when running.

## Supported Datasets 🎨

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

- For customizing your server-side process, consider to override the `package()` and `aggregate()`.

- For customizing your client-side training, consider to override the `fit()` or `package()`.

You can find all details in [`FedAvgClient`](src/client/fedavg.py) and [`FedAvgServer`](src/server/fedavg.py), which are the bases of all implementations in FL-bench.

### Integrating Dataset

- Inherit your own dataset class from `BaseDataset` in [`data/utils/datasets.py`](data/utils/datasets.py) and add your class in dict `DATASETS`.

### Customizing Model

- I offer the `CustomModel` class in [`src/utils/models.py`](src/utils/models.py) and you just need to define your model arch.
- If you want to use your customized model within FL-bench's workflow, the `base` and `classifier` must be defined. (Tips: You can define one of them as `torch.nn.Identity()` for bypassing it.)

## Citation 🧐

```bibtex
@software{Tan_FL-bench,
  author = {Tan, Jiahao and Wang, Xinpeng},
  license = {GPL-2.0},
  title = {{FL-bench: A federated learning benchmark for solving image classification tasks}},
  url = {https://github.com/KarhouTam/FL-bench}
}

@misc{tan2023pfedsim,
  title={pFedSim: Similarity-Aware Model Aggregation Towards Personalized Federated Learning}, 
  author={Jiahao Tan and Yipeng Zhou and Gang Liu and Jessie Hui Wang and Shui Yu},
  year={2023},
  eprint={2305.15706},
  archivePrefix={arXiv},
  primaryClass={cs.LG}
}

```
