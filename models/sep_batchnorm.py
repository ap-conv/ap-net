import torch
import torch.nn as nn
from torch.nn.modules.utils import _pair
from torch.nn import init


class SepBatchNorm2d(nn.Module):
    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True,
                 track_running_stats=True, sep_num=3):
        super().__init__()
        if sep_num <= 0 or sep_num >= 10:
            raise ValueError('not a good sep_num, got {}'.format(sep_num))
        self.num_features = num_features
        self.sep_num = sep_num
        self.sep_nfeatures = self.seperate_channels(sep_num, num_features)
        self.sep_bns = nn.ModuleList([
            nn.BatchNorm2d(num_features = sum(self.sep_nfeatures[i:]),
                           eps = eps,
                           momentum = momentum,
                           affine = affine,
                           track_running_stats = track_running_stats)
            for i in range(sep_num)
        ])

    def seperate_channels(self, sep_num, n_channels):
        result = []
        result += [n_channels // sep_num] * (sep_num - 1)
        result += [n_channels // sep_num + n_channels % sep_num]
        return tuple(result)

    def load_pretrained(self, module):
        sep_weight = torch.split(module.weight, self.sep_nfeatures, dim=0)
        sep_bias = torch.split(module.bias, self.sep_nfeatures, dim=0)
        sep_running_mean = torch.split(module.running_mean, self.sep_nfeatures, dim=0)
        sep_running_var = torch.split(module.running_var, self.sep_nfeatures, dim=0)
        for i in range(self.sep_num):
            assert self.sep_bns[i].weight.shape == torch.cat(sep_weight[i:], dim=0).shape
            assert self.sep_bns[i].bias.shape == torch.cat(sep_bias[i:], dim=0).shape
            assert self.sep_bns[i].running_mean.shape == torch.cat(sep_running_mean[i:], dim=0).shape
            assert self.sep_bns[i].running_var.shape == torch.cat(sep_running_var[i:], dim=0).shape
            self.sep_bns[i].weight.data = torch.cat(sep_weight[i:], dim=0).data.clone().detach()
            self.sep_bns[i].bias.data = torch.cat(sep_bias[i:], dim=0).data.clone().detach()
            self.sep_bns[i].running_mean.data = torch.cat(sep_running_mean[i:], dim=0).data.clone().detach()
            self.sep_bns[i].running_var.data = torch.cat(sep_running_var[i:], dim=0).data.clone().detach()

    def extra_repr(self):
        return 'sep_num={sep_num}, sep_nfeatures={sep_nfeatures}'.format(**self.__dict__)

    def forward(self, input, start):
        sep_num = self.sep_num
        if start < 0 or start >= sep_num:
            raise SystemExit("start={}; sep_num={};".format(start, sep_num))
        return self.sep_bns[start](input)
