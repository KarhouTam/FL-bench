from typing import Any
import torch
from src.client.fedavg import FedAvgClient


class FedPAQClient(FedAvgClient):
    """Client implementation for FedPAQ (AISTATS'20).

    Reference:
        Amirhossein Reisizadeh, Aryan Mokhtari, Hamed Hassani, Ali Jadbabaie, and Ramtin Pedarsani,
        "FedPAQ: A Communication-Efficient Federated Learning Method with Periodic Averaging and Quantization,"
        Proceedings of the 23rd International Conference on Artificial Intelligence and Statistics (AISTATS), 2020.
        http://proceedings.mlr.press/v108/reisizadeh20a.html
    """

    def __init__(self, **commons):
        super(FedPAQClient, self).__init__(**commons)

    @torch.no_grad()
    def _quantize_tensor(
        self, tensor: torch.Tensor, num_bits: int, stochastic: bool
    ) -> torch.Tensor:
        if tensor.numel() == 0:
            return tensor

        max_val = torch.max(torch.abs(tensor))
        if max_val <= 1e-8:
            return torch.zeros_like(tensor)

        levels = (1 << (num_bits - 1)) - 1
        scale = max_val / levels
        normalized = tensor / scale

        if stochastic:
            floor = torch.floor(normalized)
            prob = normalized - floor
            noise = torch.bernoulli(prob)
            quantized = floor + noise
        else:
            quantized = torch.round(normalized)

        quantized = torch.clamp(quantized, -levels, levels)
        return quantized * scale

    def package(self) -> dict[str, Any]:
        client_package = super().package()
        if self.return_diff and "model_params_diff" in client_package:
            num_bits = self.args.fedpaq.num_bits
            stochastic = bool(self.args.fedpaq.stochastic)
            for name, diff in client_package["model_params_diff"].items():
                if "num_batches_tracked" not in name:
                    client_package["model_params_diff"][name] = self._quantize_tensor(
                        diff, num_bits=num_bits, stochastic=stochastic
                    )
        return client_package
