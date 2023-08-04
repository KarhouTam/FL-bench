from torch.nn import BatchNorm2d

from fedavg import FedAvgClient


class FedBNClient(FedAvgClient):
    def __init__(self, model, args, logger, device):
        super().__init__(model, args, logger, device)
        self.personal_params_name = []
        for module_name, module in self.model.named_modules():
            if isinstance(module, BatchNorm2d):
                for param_name, _ in module.named_parameters():
                    self.personal_params_name.append(f"{module_name}.{param_name}")
        self.init_personal_params_dict = {
            name: param.clone().detach()
            for name, param in self.model.state_dict(keep_vars=True).items()
            if (not param.requires_grad) or (name in self.personal_params_name)
        }
