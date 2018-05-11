import torch
import torch.nn as nn
from torch.nn import functional as F


def conv(nin, nout, kernel_size=3, stride=1, padding=1, bias=False, layer=nn.Conv2d,
         ws=False, BN=False, pn=False, activ=nn.LeakyReLU(0.2), gainWS=2):
    convlayer = layer(nin, nout, kernel_size, stride=stride, padding=padding, bias=bias)
    layers = []
    if ws:
        layers.append(WScaleLayer(convlayer, gain=gainWS))
        if bias:
            nn.init.constant_(convlayer.bias, 0)
    if BN:
        layers.append(nn.BatchNorm2d(nout))
    if activ is not None:
        if activ == nn.PReLU:
            # to avoid sharing the same parameter, activ must be set to nn.PReLU (without '()') and initialized here
            layers.append(activ(num_parameters=1))
        else:
            layers.append(activ)
    if pn:
        layers.append(PixelNormLayer())
    layers.insert(ws, convlayer)
    return nn.Sequential(*layers)


class PixelNormLayer(nn.Module):
    def __init__(self):
        super(PixelNormLayer, self).__init__()

    def forward(self, x):
        return F.normalize(x)

    def __repr__(self):
        return self.__class__.__name__


class WScaleLayer(nn.Module):
    def __init__(self, incoming, gain=2, init=nn.init.normal_):
        super(WScaleLayer, self).__init__()

        init(incoming.weight)
        self.scale = (gain / incoming.weight[0].numel()) ** 0.5

    def forward(self, input):
        return input * self.scale

    def __repr__(self):
        return f'{self.__class__.__name__}'
