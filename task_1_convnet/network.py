import torch
import torch.nn as nn

from torch.nn.parameter import Parameter


class BatchNormDense(nn.Module):
    def __init__(self, num_features, eps):
        super().__init__()
        self.num_features = num_features
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
    def __init__(self):
        super().__init__()
        pass

    def forward(self, *input):
        pass


class ConvDenseNN(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, *input):
        pass
