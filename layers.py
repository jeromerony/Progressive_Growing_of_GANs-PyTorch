import torch
import torch.nn as nn


def conv(nin, nout, kernel_size=3, stride=1, padding=1, bias=False, layer=nn.Conv2d,
         ws=False, BN=False, pn=False, activ=nn.LeakyReLU(0.2), gainWS=2):
    convlayer = layer(nin, nout, kernel_size, stride=stride, padding=padding, bias=bias)
    layers = []
    if ws:
        layers.append(WScaleLayer(convlayer, gain=gainWS))
    if bias:
        layers.append(Bias(convlayer))
    if BN:
        layers.append(nn.BatchNorm2d(nout))
    if activ is not None:
        if type(activ) == nn.PReLU or type(activ) == nn.LeakyReLU or type(activ) == nn.ReLU:
            # if activ == nn.PReLU(), only one parameter will be shared for the whole network !
            layers.append(activ)
        elif activ == nn.PReLU:
            # to avoid sharing the same parameter, activ must be set to nn.PReLU (without '()') and initialized here
            layers.append(activ(num_parameters=1))
    if pn:
        layers.append(PixelNormLayer())
    layers.insert(ws, convlayer)
    return nn.Sequential(*layers)


def mean(tensor, axis, **kwargs):
    if isinstance(axis, int):
        axis = [axis]
    for ax in axis:
        tensor = torch.mean(tensor, dim=ax, **kwargs)
    return tensor


class PixelNormLayer(nn.Module):
    def __init__(self):
        super(PixelNormLayer, self).__init__()

    def forward(self, x):
        return x / torch.sqrt(torch.mean(x ** 2, dim=1, keepdim=True) + 1e-8)

    def __repr__(self):
        return self.__class__.__name__


class WScaleLayer(nn.Module):
    def __init__(self, incoming, gain=2, init=nn.init.normal):
        super(WScaleLayer, self).__init__()

        init(incoming.weight.data)
        self.scale = (gain / incoming.weight.data[0].numel()) ** 0.5

    def forward(self, input):
        return input * self.scale

    def __repr__(self):
        return f'{self.__class__.__name__}'


class Bias(nn.Module):
    """
    Applies bias out of the convolutions, i.e. after the WScale
    """

    def __init__(self, incoming, init='zero'):
        super(Bias, self).__init__()
        self.incomingname = incoming.__class__.__name__

        self.bias = incoming.bias
        incoming.bias = None
        if init == 'zero':
            nn.init.constant(self.bias, 0)
        elif init == 'normal':
            nn.init.normal(self.bias)

    def forward(self, input):
        return input + self.bias.view(1, -1, 1, 1)

    def __repr__(self):
        return f'{self.__class__.__name__}(incoming={self.incomingname})'
