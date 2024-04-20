from torch.nn import BatchNorm2d

from src.client.fedavg import FedAvgClient


class FedBNClient(FedAvgClient):
    def __init__(self, **commons):
        super().__init__(**commons)
        self.personal_params_name = []
        for module_name, module in self.model.named_modules():
            if isinstance(module, BatchNorm2d):
                for param_name, _ in module.named_parameters():
                    self.personal_params_name.append(f"{module_name}.{param_name}")
