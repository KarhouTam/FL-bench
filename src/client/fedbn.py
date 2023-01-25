from .fedavg import FedAvgClient


class FedBNClient(FedAvgClient):
    def __init__(self, model, args, logger):
        super().__init__(model, args, logger)
        self.personal_params_name = [
            name for name in self.model.state_dict().keys() if "bn" in name
        ]
        self.init_personal_params_dict = {
            name: param.clone().detach()
            for name, param in self.model.state_dict(keep_vars=True).items()
            if (not param.requires_grad) or (name in self.personal_params_name)
        }
