import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.nn.parameter import Parameter


class BatchNormDense(nn.Module):
    def __init__(self, num_features, eps=1e-8):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.gamma = Parameter(torch.Tensor(num_features))
        self.beta = Parameter(torch.Tensor(num_features))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.ones_(self.gamma)
        nn.init.zeros_(self.beta)

    def forward(self, x):
        means = x.mean(dim=0)
        variances = x.var(dim=0)
        x = (x - means) / torch.sqrt(variances + self.eps)
        return self.gamma * x + self.beta


class BatchNormConv(nn.Module):
    def __init__(self, num_channels, eps=1e-8):
        super().__init__()
        self.num_channels = num_channels
        self.eps = eps
        self.gamma = Parameter(torch.Tensor(num_channels))
        self.beta = Parameter(torch.Tensor(num_channels))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.ones_(self.gamma)
        nn.init.zeros_(self.beta)

    def forward(self, x):
        means = x.mean(dim=(0, 2, 3))
        variances = x.var(dim=(0, 2, 3))
        x = (x.permute(0, 2, 3, 1) - means) / torch.sqrt(variances + self.eps)
        x = self.gamma * x + self.beta
        return x.permute(0, 3, 1, 2)


class ConvDenseNN(nn.Module):
    def __init__(self, layer_list):
        super().__init__()
        self.layer_names = []
        for layer in layer_list:
            self.layer_names.append(layer[0])
            setattr(self, layer[0], layer[1])

        self.layer_checker()

    def layer_checker(self):
        pass

    def forward(self, x):
        # Boolean for determining whether linear layers were reached
        seen_linear = False

        for i, ln in enumerate(self.layer_names):
            layer = getattr(self, ln)
            if isinstance(layer, nn.Linear) and not seen_linear:
                seen_linear = True
                x = x.view(x.shape[0], -1)

            x = layer(x)
            # Apply ReLU after BatchNorm if not last layer
            if isinstance(layer, (BatchNormDense, BatchNormConv)
                          ) and not i == len(self.layer_names) - 1:
                x = F.relu(x)

        return x



