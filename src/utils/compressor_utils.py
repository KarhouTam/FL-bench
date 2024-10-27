from typing import List, Tuple, Union
import numpy as np
import torch
from torch import Tensor
from typing import Dict
from scipy.sparse.linalg import svds

from src.utils.my_utils import cal_memory


class Compressor:
    # 基类，用于压缩和解压缩
    def __init__(self, **kwargs):
        pass

    def compress(self, tensor:torch.Tensor):
        pass

    def uncompress(self, tensor:torch.Tensor):
        pass
    
    def update_basis(self, update_dict:Dict[int, torch.Tensor]):
        pass

    def update_basis_by_vector(self, vector:torch.Tensor):
        pass

class TopkCompressor(Compressor):
    '''
    TopkCompressor类, 用于基于top-k百分比的有损压缩。
    在压缩时，仅保留绝对值最大的K%个元素，并将结果转换为指定的稀疏矩阵格式；
    在解压缩时，将稀疏矩阵转换回密集矩阵。
    '''

    def __init__(self, k_percent: float, sparse_format: str = 'csr', **kwargs):
        super().__init__(**kwargs)
        self.k_percent = k_percent  # 要保留的最大元素的百分比 (0 < k_percent <= 100)
        if sparse_format.lower() not in ['coo', 'csr']:
            raise ValueError("sparse_format must be one of 'coo' or 'csr'")
        self.sparse_format = sparse_format.lower()

    def compress(self, tensor: torch.Tensor) -> Tuple[Union[torch.Tensor, torch.sparse.Tensor], torch.Tensor]:
        '''
        Args:
            tensor: torch.Tensor, 待压缩的张量
        Returns:
            compressed_tensor: torch.sparse.Tensor, 压缩后的稀疏张量
            error: torch.Tensor, 误差张量
        '''
        # 获取张量的形状和元素总数
        original_shape = tensor.shape
        num_elements = tensor.numel()
        
        # 计算要保留的元素数量（基于百分比）
        k = int(num_elements * (self.k_percent / 100.0))
        
        # 将tensor展平，以便找到全局的k个最大值
        flat_tensor = tensor.flatten()
        abs_tensor = torch.abs(flat_tensor)

        _, indices = torch.topk(abs_tensor, k)
        compressed_tensor = torch.zeros_like(flat_tensor)
        compressed_tensor[indices] = flat_tensor[indices]
        compressed_tensor = compressed_tensor.unsqueeze(0)

        # 将压缩后的张量转换为稀疏张量
        if self.sparse_format == 'coo':
            compressed_sparse_tensor = compressed_tensor.to_sparse_coo()
        elif self.sparse_format == 'csr':
            compressed_sparse_tensor = compressed_tensor.to_sparse_csr()
        # 计算误差
        error = tensor - compressed_tensor.reshape(original_shape)
        return compressed_sparse_tensor, error

    def uncompress(self, sparse_tensor: torch.sparse.Tensor, shape: Tuple[int, ...] = None) -> torch.Tensor:
        '''
        Args:
            sparse_tensor: torch.sparse.Tensor, 压缩后的稀疏张量
            shape: tuple, 原始张量的形状（如果需要的话）
        Returns:
            torch.Tensor, 解压后的密集张量
        '''
        # 将稀疏张量转换回密集张量
        if sparse_tensor.layout == torch.sparse_coo or sparse_tensor.layout == torch.sparse_csr:
            dense_tensor = sparse_tensor.to_dense()
            dense_tensor = dense_tensor.squeeze(0)
        
        # 如果提供了形状，重新塑形
        if shape is not None:
            dense_tensor = dense_tensor.reshape(shape)
        
        return dense_tensor

    def update_basis(self, update_dict: dict):
        # 对于top-k压缩，无需更新基
        pass

    def update_basis_by_vector(self, vector: torch.Tensor):
        # 对于top-k压缩，无需通过向量更新基
        return {}

class SVDCompressor(Compressor):
    '''
    SVDCompress类, 用于SVD压缩和解压缩, 用于SVDFed算法
    '''
    def __init__(self, L, R, use_scale=True, **kwargs):
        self.U:torch.Tensor = None
        self.L = L # 调整alpha
        self.R = R # 误差阈值
        self.use_scale = use_scale


    def update_basis(self, update_dict:Dict[int, torch.Tensor]):
        max_index = max(update_dict.keys())
        key = next(iter(update_dict))
        L = update_dict.get(key).shape[0]
        self.U = torch.zeros(L, max_index+1)
                    
        # key是更新位置，value是更新的值
        for k, v in update_dict.items():
            self.U[:,k] = v.clone().detach()

    def get_svd_error(self, vector_t, U):
        # 依次计算grads中每个梯度的误差
        total_error = 0
        for i in range(vector_t.shape[1]):
            g = vector_t[:,i].squeeze()
            alpha = U.T @ g
            # tmp = U @ alpha
            # alpha = min(g.norm() / tmp.norm(), self.L) * alpha
            g_approx = U @ alpha
            error = g - g_approx
            total_error += error.norm()/g.norm()
        return total_error/vector_t.shape[1]
    
    def update_basis_by_vector(self, vector:torch.Tensor):
        '''
        Return update_dict
        '''
        update_dict = {}
        vector_t = vector.T
        U, S, _ = torch.linalg.svd(vector_t, full_matrices=False)

        k = 0
        l = 0
        r = len(S)
        while l < r:
            mid = (l + r) // 2
            t = self.get_svd_error(vector_t, U[:,:mid])
            if t < self.R:
                r = mid
            else:
                l = mid + 1
        k = l

        self.U = U[:,:k]
        print(f"Layer basis selected maximun: {k}, error: {t}, threshold: {self.R}")
        # update_dict 为全部U向量
        for i in range(k):
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
            return vector, torch.zeros_like(vector)
        alpha = self.U.T @ vector_t
        _g = self.U @ alpha

        if self.use_scale:
            scale = min((torch.std(vector_t)  + 1e-8)/ (torch.std(_g)  + 1e-8), self.L)
        else:
            scale = 1

        alpha = scale*alpha
        e = vector_t - scale*_g
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


class QuickSlideSVDCompressor(Compressor):
    def __init__(self, K, max_D, L, u_dtype='float32',**kwargs):
        self.K = K # U的列数
        self.max_D = max_D # 主动更新的维度
        self.L = L # 参数切片的长度
        self.U = None # 基础的U

        self.cur_D = max_D
        if u_dtype == 'float128':
            self.dtype = np.float128
        elif u_dtype == 'float64':
            self.dtype = np.float64
        elif u_dtype== 'float32':
            self.dtype = np.float32
        elif u_dtype == 'float16':
            self.dtype = np.float16
        else:
            raise ValueError(f"Unsupported dtype {u_dtype}")
        # self.dtype = np.float16

    def update_basis(self, update_dict: dict):
        if self.U is None:
            assert len(update_dict) == self.K, f"First update_dict length must be {self.K}"
            max_index = max(update_dict.keys())
            self.U = np.zeros((self.L, max_index + 1), dtype=self.dtype)  # Initialize U as numpy array
        
        # Key is the update index, value is the update value
        for k, v in update_dict.items():
            self.U[:, k] = v.copy()

    def update_basis_by_vector(self, vector:torch.Tensor, update_threshold:float=0):
        '''
        Return update_dict
        '''
        # 通过向量更新U
        flatten_L = vector.numel()
        if flatten_L % self.L != 0:
            return {}
        vector = vector.reshape(-1, self.L)
        if self.K > vector.shape[0]:
            raise ValueError(f"K {self.K} must less than vector.shape[0] {vector.shape[0]}")
        update_dict = {}
        vector_t = vector.T.cpu().numpy()
        if self.U is None:
            # Compute U using SVD decomposition
            U, S, V = svds(vector_t, k=self.K)
            self.U = U.copy().astype(self.dtype)
            # All U vectors are updated
            for i in range(self.K):
                update_dict[i] = self.U[:, i].copy()
        
        elif self.cur_D > 0:
            # Reconstruct vector using U
            e = vector_t - self.U @ self.U.T @ vector_t
            U_e, S_e, V_e = svds(e, k=self.cur_D)
            U_K_e = np.hstack([self.U, U_e], dtype=self.dtype)  # Concatenate the new basis vectors

            alpha = U_K_e.T @ vector_t
            
            contribution = np.sum(alpha ** 2, axis=1)  # Calculate the contribution of each orthogonal vector
            min_indices = np.argsort(contribution)[:self.cur_D]  # Get the D smallest contributing indices

            min_indices_set = set(min_indices.tolist())
            wait_D_update_set = set(range(self.K, self.K + self.cur_D))
            sub_index = min_indices_set - wait_D_update_set
            add_index = wait_D_update_set - min_indices_set

            # 
            # Swap columns
            U_K_e[:, list(sub_index)] = U_K_e[:, list(add_index)]
            alpha[list(sub_index)] = alpha[list(add_index)]
            U_K = U_K_e[:, :self.K]
            if update_threshold > 0:
                alpha_2 = alpha[:self.K]
                e_2 = vector_t - U_K @ alpha_2
                # Check if error difference is below the threshold, and skip update if so
                if (np.linalg.norm(e) - np.linalg.norm(e_2)) / np.linalg.norm(e) < update_threshold:
                    return {}
            
            self.U = U_K.copy()
            # Return updated columns in dictionary form
            for i in sub_index:
                update_dict[i] = U_K_e[:, i].copy()
            
            # 动态调整更新的维度
            actual_D = len(sub_index)   # 实际更新的维度
            if actual_D >= self.cur_D//2:
                self.cur_D = min(self.max_D, self.cur_D * 2)
            
            elif self.cur_D//4 < actual_D < self.cur_D//2:
                self.cur_D = max(2, self.cur_D * 3 // 4)
            else:
                self.cur_D = max(2, self.cur_D // 2)
            print(f"actual_D/cur_D: {actual_D}/{self.cur_D}")
        return update_dict
    
    def compress(self, vector:torch.Tensor):
        '''
        Args:
            vector: torch.Tensor, 压缩的张量, 若vector的最后一个维度不能被L整除, 则返回自身, 否则返回压缩后的张量
        Returns:
            a: torch.Tensor, 张量在基下的投影
            e: torch.Tensor, 压缩张量的误差
        '''
        flatten_L = vector.numel()
        if flatten_L % self.L != 0:
            print(f"vector.numel() {flatten_L} can't divide L {self.L}. Return itself")
            return vector, torch.zeros_like(vector)
   
        vector_t = vector.reshape(-1, self.L).T
        # 通过U重构vector
        U = torch.from_numpy(self.U).to(vector.device, dtype=vector.dtype)
        alpha = U.T @ vector_t
        g = U @ alpha
        e = vector_t - g
        return alpha, e.T.reshape(vector.shape)

    def uncompress(self, alpha:torch.Tensor, shape = None):
        # 如果a的维度刚好等于shape，直接返回
        if alpha.shape == shape:
            return alpha.clone().detach()
        elif shape is None:
            U = torch.from_numpy(self.U).to(alpha.device, dtype=alpha.dtype)
            return (U @ alpha).T
        else:
            U = torch.from_numpy(self.U).to(alpha.device, dtype=alpha.dtype)
            return (U @ alpha).T.reshape(shape)


class SlideSVDCompressor(Compressor):
    def __init__(self, K, D, L, device='cpu'):
        self.K = K # U的列数
        self.D = D # 主动更新的维度
        self.L = L # 参数切片的长度
        self.U = None # 基础的U
        self.device = device

    def update_basis(self, update_dict:Dict[int, torch.Tensor]):
        if self.U is None:
            assert len(update_dict) == self.K, f"First update_dict length must be {self.K}"
            max_index = max(update_dict.keys())
            self.U = torch.zeros(self.L, max_index+1, device=self.device)
        
        # key是更新位置，value是更新的值
        for k, v in update_dict.items():
            self.U[:,k] = v.clone().detach().to(self.device)

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
            self.U = U[:,:self.K].to(self.device)
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
    def __init__(self, setting_dict:Dict[str, tuple], class_name='SlideSVDCompressor', device='cpu', **kwargs):
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
            if isinstance(value, list):
                value = tuple(value)
            self.compressor_dict[key] = compressor(*value, **kwargs, device=device)
        self.device = device

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

    def compress(self, model_params:Dict[str, Tensor], can_update_basis_func=None, **kwargs) -> Tuple[Dict[str, Tensor], dict, Dict[str, Tensor]]:
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

    def uncompress(self, combin_alpha:Dict[str, Tensor], templete_model_params:Dict[str, Tensor]) -> Dict[str, Tensor]:
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
