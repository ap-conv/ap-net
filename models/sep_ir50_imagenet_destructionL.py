import numpy as np
import re
from torch import nn
import torch
from torchvision import models, transforms, datasets
import torch.nn.functional as F
from .iresnet import iresnet50 as base_model
from .sep_iresnet import sep_iresnet50
from .sep_sequential import SepSequential
from .sep_conv import SepConv2d
from .sep_batchnorm import SepBatchNorm2d


class APIResNet50(nn.Module):
    def __init__(self, sep_num=3):
        super(APIResNet50, self).__init__()
        ap_iresnet50 = sep_iresnet50(sep_num=sep_num)
        iresnet50 = base_model(pretrained=False)
        self.stage1_img = nn.Sequential(*list(iresnet50.children())[:5])
        self.stage2_img = nn.Sequential(*list(iresnet50.children())[5:6])
        self.stage3_img = SepSequential(*list(ap_iresnet50.children())[6:7])

    def forward(self, x, start):
        x2 = self.stage1_img(x)
        x3 = self.stage2_img(x2)
        x4 = self.stage3_img(x3, start)

        return x4
