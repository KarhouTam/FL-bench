from torch.nn import BatchNorm2d

from src.client.fedavg import FedAvgClient


class FedBNClient(FedAvgClient):
    def __init__(self, **commons):
        super().__init__(**commons)
        self.personal_params_name.extend(
            name for name in self.model.state_dict().keys() if "bn" in name
        )
        # remove duplicates
        self.personal_params_name = list(set(self.personal_params_name))
