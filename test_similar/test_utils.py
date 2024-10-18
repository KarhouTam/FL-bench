import os
import threading
from typing import Union
import numpy as np
import torch
import seaborn as sns
import matplotlib.pyplot as plt
import sys
sys.path.append('/home/zhengsl/code/FL-bench')
from src.utils.my_utils import *
from concurrent.futures import ThreadPoolExecutor


def load_weight_dict_by_tag(weights_dict_dir, tag = "server", max_round = None):
    if tag is None:
        tag = "server"
    weights_dict_dir = os.path.join(weights_dict_dir,tag)
    weights_dict = {}

    lock = threading.Lock()

    def load_weight(file):
        nonlocal weights_dict
        weight_path = os.path.join(weights_dict_dir, file)
        file_split = file.split('.')[0].split('_')
        if not file_split[-2].isdigit():
            outer_round = int(file_split[-1])
            inner_round = 0
        else:
            outer_round = int(file_split[-2])
            inner_round = int(file_split[-1])
        if max_round is not None and outer_round > max_round:
            return
        weight = torch.load(weight_path)
        with lock:
            if outer_round not in weights_dict:
                weights_dict[outer_round] = {}
            weights_dict[outer_round][inner_round] = weight

    # 创建线程池
    pool = ThreadPoolExecutor(max_workers=10, thread_name_prefix='Thread')
    # 遍历文件夹
    for file in os.listdir(weights_dict_dir):
        pool.submit(load_weight, file)
    pool.shutdown(wait=True)
    for outer_round in weights_dict:
        weights_dict[outer_round] = [weights_dict[outer_round][i] for i in sorted(weights_dict[outer_round].keys())]
    return weights_dict

def plot_and_save_heatmap(matrix, save_path, y_labels, x_labels = "auto", save = True, title = 'Similarity Matrix', show = True, w_mul = 1,vmax = None, vmin = None):
    # Create heatmap
    # 宽，高
    plt.figure(figsize=(matrix.shape[1]*w_mul+2,matrix.shape[0]*0.7))
    sns.heatmap(matrix, cmap='coolwarm', annot=True, fmt=".4f", yticklabels=y_labels, xticklabels = x_labels, vmax=vmax, vmin=vmin)
    plt.title(title)
    plt.xlabel('Rounds')
    plt.ylabel('Layers')
    
    # Save the heatmap
    if save:
        plt.savefig(save_path)
    if show:
        plt.show()
    plt.close()

def cal_similar_layer_round(weights_dict, 
                            cal_score_func = CKA, 
                            begin_round = 0, 
                            round_step = None, 
                            show = False, 
                            save = True, 
                            describe = "", 
                            tag = "server", 
                            cmp:Union[str, int, dict] = 'end',
                            vmax:float = None,
                            vmin:float = None,
                            title = 'Similarity Matrix'):

    # assert all outer_round has the same inner_round
    round_list = sorted(weights_dict.keys())
    first_round = round_list[0]
    assert all(len(weights_dict[outer_round]) == len(weights_dict[first_round]) for outer_round in weights_dict)
    inner_len = len(weights_dict[first_round])
    round_num = len(weights_dict)*inner_len
    
    # the round_step add begin_step can not exceed the round_num, and the begin_round can not exceed the round_num
    # if the round_step add begin_step exceed the round_num, set the round step to the round_num - begin_round
    round_step = round_num - begin_round if round_step is None or round_step + begin_round > round_num else round_step
    num_layers = len(weights_dict[first_round][0])
    score_matrix = np.zeros((num_layers, round_step))

    flag =  isinstance(cmp, dict) and isinstance(list(cmp.keys())[0],int)
        
    for index, layer_key in enumerate(list(weights_dict[first_round][0])):
        dim = len(weights_dict[round_list[0]][0][layer_key].shape)
        if not flag:
            if isinstance(cmp, dict):
                rp = cmp
            else:
                if cmp == 'end':
                    rp = begin_round + round_step - 1
                elif cmp == 'begin':
                    rp = begin_round
                elif cmp == 'first':
                    rp = 0
                elif cmp == 'last':
                    rp = len(weights_dict)*inner_len - 1
                elif isinstance(cmp, int):
                    rp = cmp
                else:
                    raise ValueError(f"cmp should be 'end', 'begin', 'first', 'last', int or dict, but got {type(cmp)} {cmp}")

                rp_inner_round = rp % inner_len
                rp_outer_round = round_list[rp // inner_len]
                rp = weights_dict[rp_outer_round][rp_inner_round]

        for round_num in range(begin_round, begin_round + round_step):
            inner_round = round_num % inner_len
            outer_round = round_list[round_num // inner_len]
            if not flag:
                f, l = rp[layer_key].cpu(), weights_dict[outer_round][inner_round][layer_key].cpu()
            else:
                f, l = cmp[outer_round][inner_round][layer_key].cpu(), weights_dict[outer_round][inner_round][layer_key].cpu().cpu()
            if dim == 4:
                f = f.view(f.size(0), -1)
                l = l.view(l.size(0), -1)
                # f = f / torch.norm(f, dim=1, keepdim=True)
                # l = l / torch.norm(l, dim=1, keepdim=True)
                # similarity_score = cal_score_func(
                # rp[layer_key].mean(dim=[-1,-2]).cpu(), \
                # weights_dict[outer_round][inner_round][layer_key].mean(dim=[-1,-2]).cpu())              
            similarity_score = cal_score_func(f, l)          
            score_matrix[index, round_num - begin_round] = similarity_score
            
    score_func = cal_score_func.__name__
    # Save heatmap
    if tag == "server":
        save_path = f'similar_result/{describe}/{score_func}_similarity_heatmap.png'
    else:
        save_path = f'similar_result/{describe}/{tag}/{score_func}_similarity_heatmap.png'

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    if inner_len == 1:
        x_labels = [f'{round_list[i]}' for i in range(begin_round, begin_round + round_step)]
    else:
        x_labels = [f'{round_list[i // inner_len]}_{i % inner_len}' for i in range(begin_round, begin_round + round_step)]
    plot_and_save_heatmap(score_matrix, save_path, y_labels=list(weights_dict[first_round][0]), x_labels=x_labels, show=show, save=save, vmax=vmax, vmin=vmin, title=title)
    return score_matrix

def plot_score_matrix(score_matrix, save_path = None, show = True, save = True, title = 'Similarity Matrix Plot', x_labels = "auto", layer_labels = "auto", colors = "auto"):
    # Create plot
    # 折线图，x轴为round，y轴为相似度，对于每一层，画出相似度随round变化的曲线
    plt.figure(figsize=(30, 10),dpi=500)
    # 颜色
    xticks = [i for i in range(score_matrix.shape[1])]
    if x_labels == "auto":
        x_labels = xticks
    if layer_labels == "auto":
        layer_labels = [f'layer {i}' for i in range(score_matrix.shape[0])]
    if colors == "auto":
        colors = plt.cm.tab20.colors
        
    for i in range(score_matrix.shape[0]):
        plt.plot(xticks, score_matrix[i],label=layer_labels[i], color=colors[i], )
    plt.xticks(xticks, labels=x_labels, rotation=45)
    plt.title(title)
    plt.xlabel('Rounds')
    plt.ylabel('Similarity')
    plt.legend()
    # Save the plot
    if save and save_path is not None:
        plt.savefig(save_path)
    if show:
        plt.show()
    plt.close()

def cal_similar_by_layer(weights_dict, layer_key, cal_score_func = CKA, describe = "", tag = "server", show = False, save = True):
    # assert all outer_round has the same inner_round
    round_list = sorted(weights_dict.keys())
    first_round = round_list[0]
    assert all(len(weights_dict[outer_round]) == len(weights_dict[first_round]) for outer_round in weights_dict)
    inner_len = len(weights_dict[first_round])
    round_num = len(weights_dict)*inner_len
    
    score_matrix = np.zeros((round_num, round_num))
    dim = len(weights_dict[first_round][0][layer_key].shape)
    for round_x in range(0,round_num):
        outer_round_x = round_list[round_x // inner_len]
        inner_round_x = round_x % inner_len
        for round_y in range(0,round_num):
            outer_round_y = round_list[round_y // inner_len]
            inner_round_y = round_y % inner_len
            if dim == 4:
                similarity_score = cal_score_func(
                    weights_dict[outer_round_x][inner_round_x][layer_key].mean(dim=[-1,-2]), \
                    weights_dict[outer_round_y][inner_round_y][layer_key].mean(dim=[-1,-2]))
            else:
                similarity_score = cal_score_func(
                    weights_dict[outer_round_x][inner_round_x][layer_key], \
                    weights_dict[outer_round_y][inner_round_y][layer_key])
            
            score_matrix[round_num - 1 - round_x, round_y] = similarity_score

    # Normalize score_matrix
    # scaler = MinMaxScaler()
    # score_matrix_normalized = scaler.fit_transform(score_matrix)

    score_func = cal_score_func.__name__
    # Save heatmap
    if tag == "server":
        save_path = f'similar_result/{describe}/each_layer/{score_func}_similarity_{layer_key}.png'
    else:
        save_path = f'similar_result/{describe}/{tag}/each_layer/{score_func}_similarity_{layer_key}.png'

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    if inner_len == 1:
        x_labels = [f'{round_list[i]}' for i in range(round_num)]
        y_labels = [f'{round_list[i]}' for i in range(round_num - 1, 0, -1)]
    else:
        x_labels = [f'{round_list[i // inner_len]}_{i % inner_len}' for i in range(round_num)]
        y_labels = [f'{round_list[i // inner_len]}_{i % inner_len}' for i in range(round_num - 1, 0, -1)]
    plot_and_save_heatmap(score_matrix, 
                          save_path, 
                          y_labels=y_labels, 
                          x_labels=x_labels,
                          title=f'{tag}:{score_func} Similarity By {layer_key}',
                          show = show,
                          save=save)

def filter_params_dict(param_dict, layer_filter):
    filtered_dict = {}
    for outer_round in param_dict:
        filtered_dict[outer_round] = []
        for params in param_dict[outer_round]:
            filtered_dict[outer_round].append(layer_filter(params))
    return filtered_dict

def cal_similay_by_tag(weights_dict_dir,
                       tag,
                       cal_score=CKA,
                       unselect_keys=['bn'],
                       all_select_keys=['weight'],
                       any_select_keys=None,
                       describe='',
                       cmp='end',
                       save=True,
                       begin_round=0,
                       round_step=None):
    weights_dict = load_weight_dict_by_tag(weights_dict_dir, tag)
    filtered_weight_dict = filter_params_dict(weights_dict, LayerFilter(unselect_keys, all_select_keys, any_select_keys))

    if unselect_keys is not None:
        describe += f"+unselect={'&'.join(unselect_keys)}"
    if all_select_keys is not None:
        describe += f"+allselect={'&'.join(all_select_keys)}"
    if any_select_keys is not None:
        describe += f"+anyselect={'&'.join(any_select_keys)}"
    cal_similar_layer_round(filtered_weight_dict,
                            cal_score,
                            show=False,
                            tag=tag,
                            describe=describe,
                            cmp=cmp,
                            save=save,
                            begin_round=begin_round,
                            round_step=round_step)
    # for i in filtered_weight_list[0].keys():
    #     cal_similar_by_layer(filtered_weight_list, i, cal_score, tag=tag, describe=describe)