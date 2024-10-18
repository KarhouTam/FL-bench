from collections import OrderedDict
from datetime import datetime
import logging
import math
import os
from typing import List
import numpy as np
import re
import torch
from torch import Tensor
TIME = datetime.now().strftime('%Y%m%d%H%M')


def cal_memory(param, set_layout=None):
    '''
    calculate the memory size of a tensor
    param: tensor
    set_layout: can be 'torch.sparse_csr', 'torch.sparse_coo', 'torch.strided'
    '''
    if isinstance(param, int):
        param = Tensor([param])
    assert isinstance(param, torch.Tensor), 'param must be a tensor but a ' + (f'dict {param.keys()}' if isinstance(param, dict) else f'{type(param)}')
    layout = str(param.layout)

    if set_layout == 'bit':
        return math.ceil(param.numel() / 8)

    if set_layout is not None and layout != set_layout:
        print('change layout from', layout, 'to', set_layout)
        if set_layout == 'torch.sparse_csr':
            param = param.to_sparse_csr()
        elif set_layout == 'torch.sparse_coo':
            param = param.to_sparse_coo()
        elif set_layout == 'torch.strided':
            param = param.to_dense()
        else:
            raise ValueError('Unsupported layout', set_layout, 'for tensor layout', layout)

    layout = str(param.layout)
    if layout == 'torch.sparse_csr':
        row = param.crow_indices().numel() * param.crow_indices().element_size()
        col = param.col_indices().numel() * param.col_indices().element_size()
        data = param.values().numel() * param.values().element_size()
        return row + col + data
    elif layout == 'torch.sparse_coo':
        indices = param.indices().numel() * param.indices().element_size()
        data = param.values().numel() * param.values().element_size()
        return indices + data
    elif layout == 'torch.strided':
        return param.numel() * param.element_size()
    else:
        raise ValueError('Unsupported layout', layout, set_layout)
        

def calculate_data_size(param, set_sparse = None, set_layout='torch.sparse_csr'):
    '''
    set_layout: can be 'torch.sparse_csr', 'torch.sparse_coo', 'torch.strided'
    '''
    total = 0
    if set_sparse == 'all':
        sparse_param = param
        dense_param = {}
    elif set_sparse is not None:
        sparse_filter = LayerFilter(any_select_keys=set_sparse)
        sparse_param = sparse_filter(param)
        # print('sparse_param', sparse_param.keys())
        dense_filter = LayerFilter(unselect_keys=list(sparse_param.keys()))
        dense_param = dense_filter(param)
    else:
        sparse_param = {}
        dense_param = param

    for k, v in dense_param.items():
        if isinstance(v, tuple):
            for i in v:
                total += cal_memory(i)
        elif isinstance(v, dict) and 'param' in v and 'new_diff' in v:
            v = v['param']
            total += cal_memory(v)
        else:
            total += cal_memory(v)

    for k, v in sparse_param.items():
        layout = set_layout
        if isinstance(v, tuple):
            for i in v:
                total += cal_memory(i, set_layout=layout)
        elif isinstance(v, dict) and 'param' in v and 'new_diff' in v:
            if not v['new_diff']:
                layout = None
            v = v['param']
            total += cal_memory(v, set_layout=layout)
        else:
            total += cal_memory(v, set_layout=layout)

    return total

# def calculate_data_size(data):
#     # Assume data is in a serializable format, like a dictionary
#     # 若data是字典格式，其中包含tensor在GPU上，需要先将其转移到CPU上
#     data = copy.deepcopy(data)
#     if isinstance(data, torch.Tensor):
#         data = data.cpu()
#     elif isinstance(data, dict):
#         data = {k: v.cpu() if isinstance(v, torch.Tensor) else v for k, v in data.items()}
#     return len(pickle.dumps(data))

def parse_condition(cond_str):
    """Parse the condition string and return a lambda function to evaluate it."""
    range_match = re.match(r'r\[(\d+)-(\d+)\)', cond_str)
    mod_match = re.search(r'%(\d+)', cond_str)
    
    if range_match:
        range_start, range_end = int(range_match.group(1)), int(range_match.group(2))
        
        if mod_match:
            mod_value = int(mod_match.group(1))
            return lambda idx: range_start <= idx < range_end and idx % mod_value == 0
        else:
            return lambda idx: range_start <= idx < range_end
    elif mod_match:
        mod_value = int(mod_match.group(1))
        return lambda idx: idx % mod_value == 0
    else:
        raise ValueError(f"Invalid condition format: {cond_str}")

def get_config_for_round(agg_cond_list, round_idx):
    """Determine the first matching condition and return the corresponding configuration."""
    for config in agg_cond_list:
        cond_str = config['cond']
        condition = parse_condition(cond_str)
        
        if condition(round_idx):
            return config
    return None  # Return None if no conditions are met

@torch.no_grad()
def cal_opt_tensor(gk, g0):
    '''
    首先对gk和g0进行归一化，然后计算gk和g0的叉乘，判断gk和g0是否为同一方向
    若gk和g0为同一方向，则返回gk，否则进行优化
    gk: torch.tensor, shape: [batch_size, channels, height, width]
    g0: torch.tensor, shape: [batch_size, channels, height, width]
    return: torch.tensor, shape: [batch_size, channels, height, width]
    '''
    g0 = g0.to(gk.device)
    # 对gk和g0进行shape: [batch_size, channels, height * width]
    sz = gk.size()
    gk = gk.view(gk.size(0), gk.size(1), -1)
    g0 = g0.view(g0.size(0), g0.size(1), -1)
    # 对gk和g0进行归一化
    gk_norm = torch.norm(gk, p=2, dim=(1,2), keepdim=True)
    g0_norm = torch.norm(g0, p=2, dim=(1,2), keepdim=True)
    gk, g0 = gk / gk_norm, g0 / g0_norm

    b = torch.diagonal(torch.bmm(g0.transpose(1,2), gk), dim1=1, dim2=2).sum(dim=1)
    a = torch.diagonal(torch.bmm(g0.transpose(1,2), g0), dim1=1, dim2=2).sum(dim=1)
    # 判断每个batch是否为同一方向,w0为1表示不同方向, w0为0表示同一方向同时g0权重为0, b shape: [batch_size], w0 shape: [batch_size]
    # 此时w0中不同方向的项权重为b/a, 同一方向的项权重为0
    w0 = b < 0
    w0 = w0 * b / a
    # logging.info(f"w0:{b < 0}, w0*b/a:{b / a}")
    # 对gk进行优化, gk_opt shape: [batch_size, channels, height, width]
    # 其中w0为0的项不需要优化, w0为b/a的项需要优化
    gk_opt = gk - g0 * w0.unsqueeze(-1).unsqueeze(-1)
    return (gk_opt * (gk_norm + g0_norm) / 2).view(sz)

class LayerFilter:

    def __init__(self,
                 unselect_keys: List[str] = None,
                 all_select_keys: List[str] = None,
                 any_select_keys: List[str] = None) -> None:
        self.update_filter(unselect_keys, all_select_keys, any_select_keys)

    def update_filter(self,
                      unselect_keys: List[str] = None,
                      all_select_keys: List[str] = None,
                      any_select_keys: List[str] = None):
        self.unselect_keys = unselect_keys if unselect_keys is not None else []
        self.all_select_keys = all_select_keys if all_select_keys is not None else []
        self.any_select_keys = any_select_keys if any_select_keys is not None else []

    def __call__(self, param_dict, param_dict_template=None):
        if param_dict_template is not None:
            return {
                layer_key: param for layer_key, param in param_dict.items()
                if layer_key in param_dict_template
            }
        
        elif len(self.unselect_keys + self.all_select_keys +
               self.any_select_keys) == 0:
            return param_dict
        
        else:
            d = {}
            for layer_key, param in param_dict.items():
                if isinstance(layer_key, str):
                    if (len(self.unselect_keys) == 0 or all(key not in layer_key for key in self.unselect_keys)) and (
                        len(self.all_select_keys) == 0 or all(key in layer_key for key in self.all_select_keys)) and (
                        len(self.any_select_keys) == 0 or any(key in layer_key for key in self.any_select_keys)):
                        d[layer_key] = param
                elif isinstance(layer_key, int):
                    if (len(self.unselect_keys) == 0 or layer_key not in self.unselect_keys) and (
                        len(self.all_select_keys) == 0 or layer_key in (self.all_select_keys + self.any_select_keys)):
                        d[layer_key] = param
            return d
        
    def __str__(self) -> str:    
        return f"unselect_keys:{self.unselect_keys}  all_select_keys:{self.all_select_keys}  any_select_keys:{self.any_select_keys}"

    def __eq__(self, value: object) -> bool:
        if not isinstance(value, LayerFilter):
            return False
        return self.unselect_keys == value.unselect_keys and self.all_select_keys == value.all_select_keys and self.any_select_keys == value.any_select_keys

    def __hash__(self) -> int:
        return hash(str(self))

@torch.no_grad()
def aggregate_layer(w_locals, layer_name, strict=True):
    '''
    if strict is True, then the layer_name must be in all local models
    else the layer_name can be in some local models
    params:
        w_locals: list of tuple, [(sample_num, model_params), ...]
        layer_name: str, the layer name to be aggregated
        strict: bool, whether the layer_name must be in all local models
    return:
        averaged_layer: torch.tensor, the averaged layer
    '''
    if strict:
        assert all(layer_name in w_local[1] for w_local in w_locals), f"layer_name:{layer_name} not in all local models."
    else:
        w_locals = [(sample_num, model_params) for sample_num, model_params in w_locals if layer_name in model_params]
    if len(w_locals) == 0:
        return None
    training_num = sum(sample_num for sample_num, _ in w_locals)
    averaged_layer = torch.zeros_like(w_locals[0][1][layer_name], dtype=torch.float32)
    for local_sample_number, local_model_params in w_locals:
        w = local_sample_number / training_num
        averaged_layer += (local_model_params[layer_name] * w)
    return averaged_layer



def get_cka_matrix(w_list, layer_name):
    cka_matrix = np.eye(len(w_list))
    for i, (_, w_i) in enumerate(w_list):
        dim = len(w_i[layer_name].shape)
        if dim == 0:
            continue
        for j, (_, w_j) in enumerate(w_list):
            if dim == 4:
                cka_matrix[i, j] = CKA(w_i[layer_name].mean(dim=[-1,-2]), w_j[layer_name].mean(dim=[-1,-2]))
            else:
                cka_matrix[i, j] = CKA(w_i[layer_name], w_j[layer_name])
            if cka_matrix[i, j] > 1:
                cka_matrix[i, j] = 1
    return cka_matrix


def topk_indices(row, k):
    return np.argpartition(row, -k)[-k:]


def save_model_param(model_params,
                     round_idx,
                     path_tag,
                     pre_desc=None,
                     post_desc=None,
                     is_grad=True,
                     path=None):
    # 保存全局权重，作为实验
    if pre_desc is None:
        pre_desc = path_tag

    if path is None:
        path = os.path.dirname(os.path.abspath(__file__))
    save_dir = os.path.join(path,
                            'grad_lists' if is_grad else 'weight_lists',
                            path_tag)
    os.makedirs(save_dir, exist_ok=True)
    if post_desc is None:
        path = os.path.join(save_dir, f'{pre_desc}_round_{round_idx}.pt')
    else:
        path = os.path.join(save_dir,
                            f'{pre_desc}_round_{round_idx}_{post_desc}.pt')

    logging.info(f"Save {path_tag} {round_idx} model params to '{path}'.")
    torch.save(model_params, path)
    return path

@torch.no_grad()
def weight_sub(weight_x, weight_y, strict=True):
    """
    Calculate the difference between two model weights.

    Args:
        weight_x (dict): State dictionary of the first model.
        weight_y (dict): State dictionary of the second model.
    Returns:
        dict: weight_x - weight_y
    """
    device = next(iter(weight_x.values())).device
    # Create a new dictionary to store the weight differences
    weight_diff = {}
    # Iterate through the keys (parameter names) in weight_x
    for key in weight_x.keys():
        if key not in weight_y:
            if strict:
                raise ValueError(f"key:{key} not in weight_y.")
            else:
                continue
        # Compute the difference between corresponding weight tensors
        diff = weight_x[key].to(device) - weight_y[key].to(device)
        # Store the difference in the weight_diff dictionary
        weight_diff[key] = diff
    return weight_diff

@torch.no_grad()
def weight_add(weight_x, weight_y, strict=True):
    """
    Calculate the addition result between two model weights.

    Args:
        weight_x (dict): State dictionary of the first model.
        weight_y (dict): State dictionary of the second model.
        strict (bool): If True, then the keys in weight_x and weight_y must be the same.
    Returns:
        dict: weight_x + weight_y
    """
    # Create a new dictionary to store the weight differences
    device = next(iter(weight_x.values())).device
    weight_add = {}
    # Iterate through the keys (parameter names) in weight_x
    for key in weight_x.keys():
        if key not in weight_y:
            if strict:
                raise ValueError(f"key:{key} not in weight_y.")
            else:
                continue
        # Compute the difference between corresponding weight tensors
        weight_add[key] = weight_x[key].to(device) + weight_y[key].to(device)
    return weight_add


def get_model_gradient(model):
    """
    Description:
        - get norm gradients from model, and store in a OrderDict
    
    Args:
        - model: (torch.nn.Module), torch model
    
    Returns:
        - grads in OrderDict
    """
    grads = OrderedDict()
    for name, params in model.named_parameters():
        grad = params.grad
        if grad is not None:
            grads[name] = grad
    return grads


def linear_kernel(X, Y):
    return np.matmul(X, Y.transpose(0, 1))


def rbf(X, Y, sigma=None):
    """
    Radial-Basis Function kernel for X and Y with bandwith chosen
    from median if not specified.
    """
    GX = np.dot(X, Y.T)
    KX = np.diag(GX) - GX + (np.diag(GX) - GX).T
    if sigma is None:
        mdist = np.median(KX[KX != 0])
        sigma = math.sqrt(mdist)
    KX *= -0.5 / (sigma * sigma)
    KX = np.exp(KX)
    return KX


def HSIC(K, L):
    """
    Calculate Hilbert-Schmidt Independence Criterion on K and L.
    """
    n = K.shape[0]
    H = np.identity(n) - (1. / n) * np.ones((n, n))

    KH = np.matmul(K, H)
    LH = np.matmul(L, H)
    return 1. / ((n - 1)**2) * np.trace(np.matmul(KH, LH))


def CKA(X, Y, kernel=None):
    """
    Calculate Centered Kernel Alingment for X and Y. If no kernel
    is specified, the linear kernel will be used.
    """
    kernel = linear_kernel if kernel is None else kernel
    if len(X.shape) == 1:
        X = X.unsqueeze(1)
        Y = Y.unsqueeze(1)
    K = kernel(X, X)
    L = kernel(Y, Y)

    hsic = HSIC(K, L)
    varK = np.sqrt(HSIC(K, K))
    varL = np.sqrt(HSIC(L, L))
    return hsic / (varK * varL)


# -------------------第二种cka计算方法------------
from torch import Tensor
import torch


def centering(k: Tensor, inplace: bool = True) -> Tensor:
    if not inplace:
        k = torch.clone(k)
    means = k.mean(dim=0)
    means -= means.mean() / 2
    k -= means.view(-1, 1)
    k -= means.view(1, -1)
    return k


def linear_hsic(k: Tensor, l: Tensor, unbiased: bool = True) -> Tensor:
    assert k.shape[0] == l.shape[0], 'Input must have the same size'
    m = k.shape[0]
    if unbiased:
        k.fill_diagonal_(0)
        l.fill_diagonal_(0)
        kl = torch.matmul(k, l)
        score = torch.trace(kl) + k.sum() * l.sum() / (
            (m - 1) * (m - 2)) - 2 * kl.sum() / (m - 2)
        return score / (m * (m - 3))
    else:
        k, l = centering(k), centering(l)
        return (k * l).sum() / ((m - 1)**2)


def cka_score(x1: Tensor, x2: Tensor, gram: bool = False) -> Tensor:
    assert x1.shape[0] == x2.shape[0], 'Input must have the same batch size'
    if len(x1.shape) == 1:
        x1 = x1.unsqueeze(1)
        x2 = x2.unsqueeze(1)
    if not gram:
        x1 = torch.matmul(x1, x1.transpose(0, 1))
        x2 = torch.matmul(x2, x2.transpose(0, 1))
    cross_score = linear_hsic(x1, x2)
    self_score1 = linear_hsic(x1, x1)
    self_score2 = linear_hsic(x2, x2)
    return cross_score / torch.sqrt(self_score1 * self_score2)


# -----------------------余弦相似度计算--------------------
def cos_similar(x1: Tensor, x2: Tensor):
    x1 = x1.flatten()
    x2 = x2.flatten()
    # 两个norm会导致精度问题
    # return torch.sum(x1 * x2) / (torch.norm(x1) * torch.norm(x2))
    w1 = torch.sum(x1 * x1)
    w2 = torch.sum(x2 * x2)
    if w1 == 0 and w2 == 0:
        return torch.Tensor([1.0])
    return torch.sum(x1 * x2) / torch.sqrt(w1 * w2)
