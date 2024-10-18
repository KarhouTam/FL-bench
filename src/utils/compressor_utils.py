from typing import List
import torch
from torch import Tensor
from typing import Dict


class Compressor:
    # 基类，用于压缩和解压缩
    def __init__(self):
        pass

    def compress(self, tensor:torch.Tensor):
        pass

    def uncompress(self, tensor:torch.Tensor):
        pass
    
    def update_basis(self, update_dict:Dict[int, torch.Tensor]):
        pass

    def update_basis_by_vector(self, vector:torch.Tensor):
        pass

class SVDCompressor(Compressor):
    '''
    SVDCompress类, 用于SVD压缩和解压缩, 用于SVDFed算法
    '''
    def __init__(self, K):
        self.U:torch.Tensor = None
        self.K = K

    def update_basis(self, update_dict:Dict[int, torch.Tensor]):
        if self.U is None:
            assert len(update_dict) == self.K, f"First update_dict length must be {self.K}"
            max_index = max(update_dict.keys())
            key = next(iter(update_dict))
            L = update_dict.get(key).shape[0]
            self.U = torch.zeros(L, max_index+1)
                    
        # key是更新位置，value是更新的值
        for k, v in update_dict.items():
            self.U[:,k] = v.clone().detach()

    def update_basis_by_vector(self, vector:torch.Tensor):
        '''
        Return update_dict
        '''
        update_dict = {}
        vector_t = vector.T
        U, S, V = torch.linalg.svd(vector_t, full_matrices=False)
        self.U = U[:,:self.K]
        # update_dict 为全部U向量
        for i in range(self.K):
            update_dict[i] = self.U[:,i]
        return update_dict
    
    def compress(self, vector:torch.Tensor):
        '''
        Args:
            vector: torch.Tensor, 压缩的张量
        Returns:
            a: torch.Tensor, 张量在基下的投影
            e: torch.Tensor, 压缩张量的误差
        '''
        vector_t = vector.flatten().unsqueeze(0).T
        if self.U is None:
            raise ValueError("Basis is None, please update basis first")
        alpha = self.U.T @ vector_t
        g = self.U @ alpha
        e = vector_t - g
        return alpha, e.T.reshape(vector.shape)
    
    def uncompress(self, alpha:torch.Tensor, shape = None):
        '''
        Args:
            alpha: torch.Tensor, 压缩后的alpha
            shape: tuple, 原始张量的shape
        Returns:
            torch.Tensor, 解压后的张量
        '''
        # 如果a的维度刚好等于shape，直接返回
        if alpha.shape == shape:
            return alpha.clone().detach()
        elif shape is None:
            return (self.U @ alpha).T
        else:
            return (self.U @ alpha).T.reshape(shape)

class SlideSVDCompressor(Compressor):
    def __init__(self, K, D, L):
        self.K = K # U的列数
        self.D = D # 主动更新的维度
        self.L = L # 参数切片的长度
        self.U = None # 基础的U

    def update_basis(self, update_dict:Dict[int, torch.Tensor]):
        if self.U is None:
            assert len(update_dict) == self.K, f"First update_dict length must be {self.K}"
            max_index = max(update_dict.keys())
            self.U = torch.zeros(self.L, max_index+1)
        
        # key是更新位置，value是更新的值
        for k, v in update_dict.items():
            self.U[:,k] = v.clone().detach()

    def update_basis_by_vector(self, vector:torch.Tensor, update_threshold:float=0):
        '''
        Return update_dict
        '''
        # 通过向量更新U
        flatten_L = vector.numel() if len(vector.shape) == 1 else (vector.numel() // vector.shape[0])
        if flatten_L % self.L != 0:
            return {}
        vector = vector.reshape(-1, self.L)
        if self.K > vector.shape[0]:
            raise ValueError(f"K {self.K} must less than vector.shape[0] {vector.shape[0]}")
        update_dict = {}
        vector_t = vector.T
        if self.U is None:
            # 通过SVD分解得到U
            U, S, V = torch.linalg.svd(vector_t, full_matrices=False)
            self.U = U[:,:self.K]
            # update_dict 为全部U向量
            for i in range(self.K):
                update_dict[i] = self.U[:,i]
        
        elif self.D > 0:
            # 通过U重构vector
            e = vector_t - self.U @ self.U.T @ vector_t
            U_e, S_e, V_e = torch.linalg.svd(e, full_matrices=False)
            U_K_e = torch.cat([self.U, U_e[:,:self.D]], dim=1)

            alpha = U_K_e.T @ vector_t
            
            contribution = torch.sum(alpha ** 2, dim=1)  # 计算每个正交向量的贡献度（平方和）
            _, min_indices = torch.topk(contribution, k=self.D, largest=False)

            min_indices_set = set(min_indices.tolist())
            wait_D_update_set = set([i for i in range(self.K, self.K+self.D)])
            sub_index = min_indices_set - wait_D_update_set
            add_index = wait_D_update_set - min_indices_set

            # 交换列
            U_K_e[:,list(sub_index)] = U_K_e[:,list(add_index)]
            alpha[list(sub_index)] = alpha[list(add_index)]
            U_K = U_K_e[:,:self.K]
            alpha_2 = alpha[:self.K]
            # alpha_2 = U_K.T @ vector_t
            # alpha_2不需要重新计算，通过alpha和U_K_e的更新列计算
            e_2 = vector_t - U_K @ alpha_2

            # 若更新后的误差变化小于阈值，则不更新
            # print(f"de {(e.norm() - e_2.norm())/e.norm()}, update_threshold {update_threshold}")
            if (e.norm() - e_2.norm())/e.norm() < update_threshold:
                return {}
            
            self.U = U_K_e[:,:self.K]
            # 返回更新列字典
            for i in sub_index:
                update_dict[i] = U_K_e[:,i].clone().detach()
        
        return update_dict

    def compress(self, vector:torch.Tensor):
        '''
        Args:
            vector: torch.Tensor, 压缩的张量, 若vector的最后一个维度不能被L整除, 则返回自身, 否则返回压缩后的张量
        Returns:
            a: torch.Tensor, 张量在基下的投影
            e: torch.Tensor, 压缩张量的误差
        '''
        flatten_L = vector.numel() if len(vector.shape) == 1 else (vector.numel() // vector.shape[0])
        if flatten_L % self.L != 0:
            print(f"vector.numel() // vector.shape[0] {flatten_L} can't divide L {self.L}. Return itself")
            return vector, torch.zeros_like(vector)
   
        vector_t = vector.reshape(-1, self.L).T
        # 通过U重构vector
        alpha = self.U.T @ vector_t
        g = self.U @ alpha
        e = vector_t - g
        return alpha, e.T.reshape(vector.shape)

    def uncompress(self, alpha:torch.Tensor, shape = None):
        # 如果a的维度刚好等于shape，直接返回
        if alpha.shape == shape:
            return alpha.clone().detach()
        elif shape is None:
            return (self.U @ alpha).T
        else:
            return (self.U @ alpha).T.reshape(shape)

class CompressorCombin:
    def __init__(self, setting_dict:Dict[str, tuple], class_name='SlideSVDCompressor'):
        '''
        CompressorCombin类, 用于组合多个Compress类, 为多层参数提供压缩和解压缩功能
        Args:
            setting_dict: key是参数名, value是元组(K, D, L), K是U的列数, D是主动更新的维度, L是参数切片的长度
        '''
        if not isinstance(setting_dict, dict):
            raise ValueError("setting_dict must be a dict")
        
        compressor = globals()[class_name]

        self.setting_dict = setting_dict
        self.compressor_dict:Dict[str, Compressor] = {}
        for key, value in setting_dict.items():
            self.compressor_dict[key] = compressor(*value)

    def update_basis_by_vector(self, model_params:Dict[str, Tensor]):
        '''
        通过model_params更新全部compressor的基
        Args:
            model_params: 模型参数字典
        Returns:
            dict: 更新字典
        '''
        res = {}
        for key, value in model_params.items():
            if key not in self.compressor_dict:
                continue
            compressor = self.compressor_dict[key]
            res[key] = compressor.update_basis_by_vector(value)
        return res

    def compress(self, model_params:Dict[str, Tensor], can_update_basis_func=None, **kwargs):
        '''
        压缩combine中全部compressor的参数
        Args:
            model_params: 模型参数字典, 如果key不在compressor_dict中, 则不压缩
            can_update_basis_func: 是否可以更新基函数的函数, 返回True或False
        Returns:
            combin_alpha: 压缩后的alpha字典 
            combin_update_dict: 更新字典
        '''
        combin_alpha = {}
        combin_update_dict = {}
        combin_error = {}
        for key, value in model_params.items():
            if key not in self.compressor_dict:
                combin_alpha[key] = value
                combin_update_dict[key] = {}
                combin_error[key] = torch.zeros_like(value)
                continue
            compressor = self.compressor_dict[key]
            if can_update_basis_func is not None:
                if can_update_basis_func(**kwargs):
                    combin_update_dict[key] = compressor.update_basis_by_vector(value)
                else:
                    combin_update_dict[key] = {}
            combin_alpha[key], combin_error[key] = compressor.compress(value)
            # res = compressor.uncompress(combin_alpha[key], value.shape)
            # sim = cos_similar(value, res)
            # print(f"key {key}, cos_similar {sim}")
        return combin_alpha, combin_update_dict, combin_error

    def uncompress(self, combin_alpha:Dict[str, Tensor], templete_model_params:Dict[str, Tensor]):
        '''
        根据combin_alpha解压, 如果key不在compressor_dict中, 则无需解压
        Args:
            combin_alpha: 压缩后的alpha字典
            templete_model_params: 参数模板，用于指定解压后的参数形状
        Returns:
            dict: 解压后的模型参数
        '''
        res = {}
        for key, value in combin_alpha.items():
            if key not in self.compressor_dict:
                res[key] = value
            else:
                res[key] = self.compressor_dict[key].uncompress(value, templete_model_params[key].shape)
        return res
    
    def update(self, combin_update_dict:Dict[str, Dict[int, Tensor]]):
        '''
        更新combine中全部compressor的基
        Args:
            combin_update_dict: key是参数名, value是更新字典, 更新字典的key是更新位置, value是更新的值
        '''
        for key, value in combin_update_dict.items():
            if key not in self.compressor_dict:
                continue
            compressor = self.compressor_dict[key]
            compressor.update_basis(value)

class QSGDQuantizer:
    def __init__(self, num_levels=256):
        self.num_levels = num_levels

    def quantize(self, tensor:torch.Tensor):
        norm = tensor.norm(p=2)
        scale = norm / self.num_levels
        sign = tensor.sign()
        abs_tensor = tensor.abs()
        q = (abs_tensor / scale).floor()
        prob = (abs_tensor / scale) - q
        rand_tensor = torch.rand_like(prob)
        q += torch.where(rand_tensor < prob, torch.ones_like(q), torch.zeros_like(q))
        quantized_tensor = sign * q
        return quantized_tensor, 0, scale
    
    def dequantize(self, quantized_tensor, min_val, scale):
        dequantized_tensor = quantized_tensor * scale
        return dequantized_tensor
