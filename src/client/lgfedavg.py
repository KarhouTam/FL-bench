from torch.nn import BatchNorm2d, Conv2d, Linear

from fedavg import FedAvgClient


class LGFedAvgClient(FedAvgClient):
    def __init__(self, model, args, logger, device):
        super().__init__(model, args, logger, device)
        self.personal_params_name = []
        trainable_layers = [
            (name, module)
            for name, module in model.named_modules()
            if isinstance(module, Conv2d)
            or isinstance(module, BatchNorm2d)
            or isinstance(module, Linear)
        ]
        personal_layers = trainable_layers[self.args.num_global_layers :]

        for module_name, module in personal_layers:
            for param_name, _ in module.named_parameters():
                self.personal_params_name.append(f"{module_name}.{param_name}")

        self.init_personal_params_dict = {
            name: param.clone().detach()
            for name, param in self.model.state_dict(keep_vars=True).items()
            if (not param.requires_grad) or (name in self.personal_params_name)
        }
