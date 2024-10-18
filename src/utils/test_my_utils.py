# import unittest
# import torch
# from my_utils import QSGDQuantizer

# class TestQSGDQuantizer(unittest.TestCase):
#     def setUp(self):
#         self.quantizer = QSGDQuantizer(num_levels=512)

#     def test_quantize_shape(self):
#         tensor = torch.randn(5, 5)/10000
#         print(tensor)
#         quantized_tensor, min_val, scale = self.quantizer.quantize(tensor)
#         print(quantized_tensor, min_val, scale)
#         quantized_tensor = quantized_tensor.to(torch.int8)
#         print(quantized_tensor, min_val, scale)
#         tensor2 = self.quantizer.dequantize(quantized_tensor, min_val, scale)
#         print(tensor2)
#     # def test_quantize_values(self):
#     #     tensor = torch.tensor([1.0, -2.0, 3.0, -4.0])
#     #     quantized_tensor, min_val, scale = self.quantizer.quantize(tensor)
#     #     self.assertTrue(torch.all(quantized_tensor.abs() <= self.quantizer.num_levels))
#     #     self.assertTrue(torch.all(quantized_tensor.sign() == tensor.sign()))

#     # def test_quantize_zero_tensor(self):
#     #     tensor = torch.zeros(10, 10)
#     #     quantized_tensor, min_val, scale = self.quantizer.quantize(tensor)
#     #     self.assertTrue(torch.all(quantized_tensor == 0))
#     #     self.assertEqual(scale.item(), 0)

#     # def test_quantize_random_tensor(self):
#     #     tensor = torch.randn(100, 100)
#     #     quantized_tensor, min_val, scale = self.quantizer.quantize(tensor)
#     #     self.assertTrue(torch.all(quantized_tensor.abs() <= self.quantizer.num_levels))
#     #     self.assertTrue(torch.all(quantized_tensor.sign() == tensor.sign()))

# if __name__ == '__main__':
#     unittest.main()

import torch
import unittest
from my_utils import CompressorCombin, SlideSVDCompress

class TestCompressorCombinModel(unittest.TestCase):
    def setUp(self):
        setting_dict = {
            'layer1': (2, 1, 4),
            'layer2': (3, 2, 6)
        }
        self.combin = CompressorCombin(setting_dict)
        self.model_params = {
            'layer1': torch.randn(8, 4),
            'layer2': torch.randn(18, 6)
        }


    def test_compress(self):
        compress_dict, combin_update_dict = self.combin.compress(self.model_params, lambda : True)
        self.assertEqual(set(compress_dict.keys()), set(self.model_params.keys()))
        self.assertEqual(set(combin_update_dict.keys()), set(self.model_params.keys()))

    def test_uncompress(self):
        compress_dict, _ = self.combin.compress(self.model_params, lambda : True)
        target_model_params = {k: torch.zeros_like(v) for k, v in self.model_params.items()}
        self.combin.uncompress(compress_dict, target_model_params)
        for key in self.model_params:
            self.assertTrue(torch.allclose(self.model_params[key], target_model_params[key], atol=1e-5))

    def test_update(self):
        _, combin_update_dict = self.combin.compress(self.model_params, lambda : True)
        self.combin.update(combin_update_dict)
        for key, update_dict in combin_update_dict.items():
            compressor = self.combin.compressor_dict[key]
            for idx, tensor in update_dict.items():
                self.assertTrue(torch.allclose(compressor.U[:, idx], tensor, atol=1e-5))

if __name__ == '__main__':
    unittest.main()
