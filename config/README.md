# Generic Arguments 🛠
| Argument| Description        |
| - | - |
| `dataset`                            | The name of dataset that experiment run on.        |
| `model`                              | The model backbone experiment used.                |
| `seed`| Random seed for running experiment.                |
| `join_ratio`                         | Ratio for (client each round) / (client num in total).                            |
| `global_epoch`                       | Global epoch, also called communication round.     |
| `local_epoch`                        | Local epoch for client local training.             |
| `finetune_epoch`                     | Epoch for clients fine-tunning their models before test.                          |
| `test_gap`                           | Interval round of performing test on clients.      |
| `eval_test`                          | `true` for performing evaluation on joined clients' testset before and after local training.             |
| `eval_train`                         | `true` for performing evaluation on joined clients' trainset before and after local training.            |
| `local_lr`                           | Learning rate for client local training.           |
| `momentum`                           | Momentum for client local opitimizer.              |
| `weight_decay`                       | Weight decay for client local optimizer.           |
| `verbose_gap`                        | Interval round of displaying clients training performance on terminal.            |
| `batch_size`                         | Data batch size for client local training.         |
| `use_cuda`                           | `true` indicates that tensors are in gpu.  |
| `visible`                            | `true` for using Visdom to monitor algorithm performance on `localhost:8097`.                            |
| `global_testset`                     | `true` for evaluating client models over the global testset before and after local training, instead of evaluating over clients own testset. The global testset is the union set of all client's testset. |
| `save_log`                           | `true` for saving algorithm running log in `FL-bench/out/${algo}`.        |
| `straggler_ratio`                    | The ratio of stragglers (set in `[0, 1]`). Stragglers would not perform full-epoch local training as normal clients. Their local epoch would be randomly selected from range `[straggler_min_local_epoch, local_epoch)`.                 |
| `straggler_min_local_epoch`          | The minimum value of local epoch for stragglers.   |
| `external_model_params_file`         | The relative file path of external (pretrained) model parameters (`*.pt`). e.g., `../../out/FedAvg/mnist_100_lenet5.pt`. This feature is enabled only when `unique_model=False`.                          |
| `save_model`                         | `true` for saving output model(s) parameters in `FL-bench/out/${algo}`.  The default file name pattern is `${dataset}_${global_epoch}_${model}.pt`.                    |
| `save_fig`                           | `true` for saving the accuracy curves showed on Visdom into a `.jpeg` file at `FL-bench/out/${algo}`.    |
| `save_metrics`                       | `true` for saving metrics stats into a `.csv` file at `FL-bench/out/${algo}`.                            |
| `viz_win_name`                       | Custom visdom window name (active when setting `visible` as `true`).  |
| `--check_convergence` | `true` for checking convergence after training.  |