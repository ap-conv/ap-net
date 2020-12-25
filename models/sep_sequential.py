import torch
import torch.nn as nn


class SepSequential(nn.Sequential):
    def __init__(self, *args):
        super(SepSequential, self).__init__(*args)
    
    def forward(self, input, start):
        for module in self:
            if any([isinstance(module, mod) 
                    for mod in [nn.AdaptiveAvgPool2d, nn.AvgPool2d, nn.MaxPool2d,
                                nn.ReLU, nn.BatchNorm2d, nn.Linear, nn.Conv2d]]):
                input = module(input)
            else:
                input = module(input, start)
                """
            if isinstance(module, SepConv2d):
                input = module(input, start)
            elif isinstance(module, SepSequential):
                input = module(input, start)
            elif isinstance(module, BasicBlock):
                input = module(input, start)
            else:
                try:
                    input = module(input)
                except TypeError:
                    print("debug success:", module)
                    raise SystemExit
                """
        return input

