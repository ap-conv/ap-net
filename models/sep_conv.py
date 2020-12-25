import torch
import torch.nn as nn
from torch.nn.modules.utils import _pair
from torch.nn import init
from torch.nn.parallel.data_parallel import DataParallel



class SepConv2d(nn.Module):

    __constants__ = ['stride', 'padding', 'dilation', 'groups', 'bias',
                     'padding_mode', 'output_padding', 'in_channels',
                     'out_channels', 'kernel_size', 
                     'sep_num', 'sep_in_channels', 'sep_out_channels']

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1,
                 bias=True, padding_mode='zeros',
                 sep_num=3):
        super().__init__()
        kernel_size = _pair(kernel_size)
        stride = _pair(stride)
        padding = _pair(padding)
        dilation = _pair(dilation)

        if groups != 1:
            raise ValueError('groups must be 1')

        if not isinstance(sep_num, int):
            raise ValueError('sep_num should be int')       
        if sep_num <= 0 or sep_num >= 10:
            raise ValueError('not a good sep_num, got {}'.format(sep_num))

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.transposed = False
        self.output_padding = _pair(0)
        self.groups = 1
        self.padding_mode = padding_mode
        self.sep_num = sep_num
        self.sep_in_channels = self.seperate_channels(sep_num, in_channels)
        self.sep_out_channels = self.seperate_channels(sep_num, out_channels)
        
        self.sep_convs = nn.ModuleList([
            nn.Conv2d(in_channels = sum(self.sep_in_channels[i:]),
                      out_channels = self.sep_out_channels[i],
                      kernel_size = kernel_size,
                      stride = self.stride,
                      padding = self.padding,
                      dilation = self.dilation,
                      groups = self.groups,
                      bias = bias,
                      padding_mode = self.padding_mode)
            for i in range(sep_num)
        ])
    
    def seperate_channels(self, sep_num, n_channels):
        result = []
        result += [n_channels // sep_num] * (sep_num - 1)
        result += [n_channels // sep_num + n_channels % sep_num]
        return tuple(result)

    def extra_repr(self):
        s = 'sep_num={sep_num}'
        s += ', sep_in_channels={sep_in_channels}'
        s += ', sep_out_channels={sep_out_channels}'
        return s.format(**self.__dict__)
    
    def load_pretrained(self, module):
        sep_weight = torch.split(module.weight.data, self.sep_in_channels, dim=1)
        sep_weight = [torch.split(e, self.sep_out_channels, dim=0) for e in sep_weight]
        if module.bias is not None:
            sep_bias = torch.split(module.bias.data, self.sep_out_channels, dim=0)
        for i in range(self.sep_num):
            current_weight = [e[i] for e in sep_weight]
            current_weight = torch.cat(current_weight[i:], dim=1)
            assert self.sep_convs[i].weight.shape == current_weight.shape
            self.sep_convs[i].weight.data = current_weight.data.clone().detach()
            if module.bias is not None:
                assert self.sep_convs[i:].bias.shape == current_bias.shape
                current_bias = torch.cat(sep_bias[i], dim=0)
                self.sep_convs[i].bias.data = current_bias.data.clone().detach()

    def __setstate__(self, state):
        super().__setstate__(state)
        if not hasattr(self, 'padding_mode'):
            self.padding_mode = 'zeros'

    def forward(self, input, start):
        """
        Y1 = (X1 + X2 + X3) * K1
        Y2 = (X2 + X3) * K2
        Y3 = X3 * K3

        K1 in_channels = X1 + X2 + X3
           out_channels = Y1 ~ bias
        K2 in_channels = X2 + X3
           out_channels = Y2 ~ bias
        K3 in_channels = X3
           out_channels = Y3 ~ bias

        """
        if input.size(1) == sum(self.sep_in_channels):
            sep_input = torch.split(input, 
                                    split_size_or_sections = self.sep_in_channels,
                                    dim = 1)  # on channel dimension
            c = start
        else:
            sep_input = torch.split(input, 
                                    split_size_or_sections = self.sep_in_channels[start:],
                                    dim = 1)  # on channel dimension
            c = 0
        result = []
        sep_num = self.sep_num
        if start < 0 or start >= sep_num:
            raise SystemExit("start={}; sep_num={};".format(start, sep_num))
        for i in range(start, sep_num):
            sep_conv = self.sep_convs[i]
            scale = 1.0 * sep_num / (sep_num - i)
            current_input = torch.cat(sep_input[c:], dim=1)
            c += 1
            result.append(sep_conv(current_input) * scale)
        result = torch.cat(result, dim=1)
        return result
