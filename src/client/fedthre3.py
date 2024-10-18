from copy import deepcopy
import math
import time
from typing import Any
import torch

from src.client.fedavg import FedAvgClient
from src.utils.my_utils import LayerFilter, cos_similar, save_model_param, weight_sub


class FedThre3Client(FedAvgClient):
    def __init__(self, **commons):
        # 多出一个output_dir参数，将output_dir参数从commons中删除
        self.output_dir = commons['output_dir']
        del commons['output_dir']
        super().__init__(**commons)
        self.thre_lf = LayerFilter(unselect_keys=['bn', 'running', 'num_batches_tracked'])
        # self.thre_lf = LayerFilter()
        self.total_weight = 1

        self.last_grad_norm = dict()
        

    @torch.no_grad()
    def set_parameters(self, package: dict[str, Any]):
        self.local_round_idx = package['local_round_idx']
        self.total_opt_diff = package['total_opt_diff']
        # grad_stack: dict[str, list[dict[str, torch.Tensor]]], 
        # example: {'layer1': [{'weight': tensor, 'bias': tensor}, {'weight': tensor, 'bias': tensor}]}
        self.grad_stack_len = package['grad_stack_len']
        self.grad_stack: dict[str, list[dict[str, torch.Tensor]]] = \
            package['grad_stack']
        self.threshold = package['threshold']
        self.min_cos = package['min_cos']
        self.total_weight = package['total_weight']
        
        for layer_name, layer_param in package["regular_model_params"].items():
            if layer_name in self.total_opt_diff:
                layer_param.data += (self.total_opt_diff[layer_name]*len(self.trainset)/self.total_weight).to(layer_param.device)
                # 置零
                # self.total_opt_diff[layer_name] = torch.zeros_like(self.total_opt_diff[layer_name], requires_grad=False)

        super().set_parameters(package)
        self.old_params = deepcopy(self.model.state_dict())

    def get_threshold_cos(self, u, ratio, min_cos):
        '''
        u: float, control the threshold
        p: float, the ratio of the gradient, last gradient divide current gradient
        min_cos: float, if the cos value is less than min_cos, return min_cos
        '''
        if ratio is None:
            return min_cos
        
        sina = (1+ratio)*u
        if sina > 1:
            return min_cos
        else:
            return math.sqrt(1 - sina**2)

    @torch.no_grad()
    def update_threshold_diff(self, G_params: dict[str,list[dict]], g_param):
        res = {}
        opt_x = {}
        opt_Gx = {}
        timestr = time.strftime("[%Y-%m-%d %H:%M:%S]", time.localtime())

        print(f'{timestr} Client:{self.client_id} Local round:{self.local_round_idx} Update threshold diff')
        for layer_name, layer_param in self.thre_lf(g_param).items():
            # print(f'client {self.client_id} update_threshold_diff layer_name: {layer_name}')
            if layer_name not in self.last_grad_norm:
                ratio = None
            else:
                ratio = torch.norm(layer_param, p='fro') / self.last_grad_norm[layer_name]
            self.last_grad_norm[layer_name] = torch.norm(layer_param, p='fro')

            if layer_name not in G_params or len(G_params[layer_name]) == 0:
                G_params[layer_name] = [layer_param]
                res[layer_name] = dict(
                    new_diff=True,
                    param=layer_param,
                )
                self.total_opt_diff[layer_name] = torch.zeros_like(layer_param, requires_grad=False)
            else:
                # 判断g_param全0
                if (layer_param == 0).all():
                    opt_x[layer_name] = torch.Tensor([0.0])
                    opt_Gx[layer_name] = torch.zeros_like(layer_param)
                    res[layer_name] = dict(
                        new_diff=False,
                        param=opt_x[layer_name],
                    )
                    self.total_opt_diff[layer_name] += layer_param
                    continue
                
                sp = layer_param.shape
                if len(sp) != 1:
                    G = torch.stack(
                        [p.reshape(sp[0],-1) for p in G_params[layer_name]]
                        ).permute(1, 2, 0)
                    g = layer_param.reshape(sp[0],-1)
                else:
                    G = torch.stack(
                        [p for p in G_params[layer_name]]
                        ).permute(1, 0).unsqueeze(0)
                    g = layer_param.unsqueeze(0)
                    
                flag = True
                try:
                    x = torch.linalg.lstsq(G, g).solution
                except torch._C._LinAlgError:
                    print(f"\tMatrix is rank-deficient, applying Tikhonov regularization")
                    rank = torch.linalg.matrix_rank(G)
                    G[rank < self.grad_stack_len] += 1e-9  # 正则化参数
                    x = torch.linalg.lstsq(G, g).solution
                except Exception as e:
                    flag = False
                    print(f'\tlayer_name: {layer_name}, lstsq error: {e}')
                
                if flag:
                    x[torch.isnan(x)] = 0
                    x[torch.isinf(x)] = 0
                    Gx = torch.bmm(G, x.unsqueeze(-1)).squeeze(-1).reshape(sp)
                    opt_x[layer_name] = x
                    opt_Gx[layer_name] = Gx
                    sim = cos_similar(Gx, g)

                    cos_threshold = self.get_threshold_cos(self.threshold, ratio, self.min_cos)
                    print(f'\tlayer_name: {layer_name}, x shape: {x.shape} cos_similar: {sim} cos_threshold: {cos_threshold} use_min: {cos_threshold == self.min_cos}')
                    if (not (x == 0).all()) and sim > cos_threshold:
                        res[layer_name] = dict(new_diff=False, param=x)
                        # 计算未上传差值
                        self.total_opt_diff[layer_name] += (layer_param - opt_Gx[layer_name])
                        continue
                
                param = layer_param + self.total_opt_diff[layer_name]
                res[layer_name] = dict(new_diff=True, param=param)
                G_params[layer_name].append(param)
                if len(G_params[layer_name]) > self.grad_stack_len:
                    G_params[layer_name].pop(0)
                self.total_opt_diff[layer_name] = torch.zeros_like(layer_param, requires_grad=False)

        print()
        self.thre_diff = res
        return opt_x, opt_Gx

    def train(self, server_package: dict[str, Any]):
        self.set_parameters(server_package)
        self.train_with_eval()
        with torch.no_grad():
            self.grad = weight_sub(self.old_params, self.model.state_dict())
            save_model_param(self.grad,
                            self.local_round_idx,
                            f'c{self.client_id}',
                            is_grad=True,
                            path=self.output_dir)
        opt_x, opt_Gx = self.update_threshold_diff(self.grad_stack, self.grad)
        self.local_round_idx += 1
        client_package = self.package()
        return client_package

    def package(self):
        """Package data that client needs to transmit to the server.
        You can override this function and add more parameters.

        Returns:
            A dict: {
                `weight`: Client weight. Defaults to the size of client training set.
                `regular_model_params`: Client model parameters that will join parameter aggregation.
                `model_params_diff`: The parameter difference between the client trained and the global. `diff = global - trained`.
                `eval_results`: Client model evaluation results.
                `personal_model_params`: Client model parameters that absent to parameter aggregation.
                `optimzier_state`: Client optimizer's state dict.
                `lr_scheduler_state`: Client learning rate scheduler's state dict.
            }
        """        
        client_package = dict(
            weight=len(self.trainset),
            eval_results=self.eval_results,
            model_params_diff=self.thre_diff,
            personal_model_params=self.model.state_dict(),
            optimizer_state=deepcopy(self.optimizer.state_dict()),
            lr_scheduler_state=(
                {}
                if self.lr_scheduler is None
                else deepcopy(self.lr_scheduler.state_dict())
            ),
            local_round_idx=self.local_round_idx
        )
        
        return client_package