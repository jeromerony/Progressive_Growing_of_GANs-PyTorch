from math import ceil

import torch.nn.functional as F

from layers import *


class Generator(nn.Module):
    def __init__(self, maxRes=8, nch=16, nc=3, bias=False, BN=False, ws=False, pn=False, activ=nn.LeakyReLU(0.2)):
        super(Generator, self).__init__()
        # resolution of output as 4 * 2^maxRes: 0 -> 4x4, 1 -> 8x8, ..., 8 -> 1024x1024
        self.maxRes = maxRes

        # output convolutions
        self.toRGBs = []
        for i in range(self.maxRes + 1):
            # max of nch * 32 feature maps as in the original article (with nch=16, 512 feature maps at max)
            self.toRGBs.append(conv(int(nch * 2 ** (8 - max(3, i))), nc, kernel_size=1, padding=0, bias=bias,
                               ws=ws, activ=None, gainWS=1))
        self.toRGBs = nn.ModuleList(self.toRGBs)

        # convolutional blocks
        self.blocks = []
        # first block, always present
        self.blocks.append(nn.Sequential(
            conv(nch * 32, nch * 32, kernel_size=4, padding=3, bias=bias, BN=BN, ws=ws, pn=pn, activ=activ),
            conv(nch * 32, nch * 32, bias=bias, BN=BN, ws=ws, pn=pn, activ=activ)))
        for i in range(self.maxRes):
            nin = int(nch * 2 ** (8 - max(3, i)))
            nout = int(nch * 2 ** (8 - max(3, i + 1)))
            self.blocks.append(nn.Sequential(
                conv(nin, nout, bias=bias, BN=BN, ws=ws, pn=pn, activ=activ),
                conv(nout, nout, bias=bias, BN=BN, ws=ws, pn=pn, activ=activ)))
        self.blocks = nn.ModuleList(self.blocks)

        self.pn = []
        if pn:
            self.pn.append(PixelNormLayer())
        self.pn = nn.Sequential(*self.pn)

    def forward(self, input, x=None):
        # value driving the number of layers used in generation
        if x is None:
            progress = self.maxRes
        else:
            progress = min(x, self.maxRes)

        alpha = progress - int(progress)

        pn_input = self.pn(input)

        # generating image of size corresponding to progress
        # Example : for progress going from 0 + epsilon to 1 excluded :
        # the output will be of size 8x8 as sum of 4x4 upsampled and output of convolution
        y1 = self.blocks[0](pn_input)
        y0 = y1

        for i in range(1, int(ceil(progress) + 1)):
            y1 = F.upsample(y1, scale_factor=2)
            y0 = y1
            y1 = self.blocks[i](y0)

        # converting to RGB
        y = self.toRGBs[int(ceil(progress))](y1)

        # adding upsampled image from previous layer if transitioning, i.e. when progress is not int
        if progress % 1 != 0:
            y0 = self.toRGBs[int(progress)](y0)
            y = alpha * y + (1 - alpha) * y0

        return y


class Discriminator(nn.Module):
    def __init__(self, maxRes=8, nch=16, nc=3, bias=False, BN=False, ws=False, activ=nn.LeakyReLU(0.2)):
        super(Discriminator, self).__init__()
        # resolution of output as 4 * 2^maxRes: 0 -> 4x4, 1 -> 8x8, ..., 8 -> 1024x1024
        self.maxRes = maxRes

        # input convolutions
        self.fromRGBs = []
        for i in range(self.maxRes + 1):
            self.fromRGBs.append(conv(nc, int(nch * 2 ** (8 - max(3, i))), kernel_size=1, padding=0, bias=bias,
                                 BN=BN, ws=ws, activ=activ))
        self.fromRGBs = nn.ModuleList(self.fromRGBs)

        # convolutional blocks
        self.blocks = []
        # last block, always present
        self.blocks.append(nn.Sequential(conv(nch * 32 + 1, nch * 32, bias=bias, BN=BN, ws=ws, activ=activ),
                                    conv(nch * 32, nch * 32, kernel_size=4, padding=0, bias=bias,
                                         BN=BN, ws=ws, activ=activ),
                                    conv(nch * 32, 1, kernel_size=1, padding=0, bias=bias,
                                         ws=ws, gainWS=1, activ=None)))
        for i in range(self.maxRes):
            nin = int(nch * 2 ** (8 - max(3, i + 1)))
            nout = int(nch * 2 ** (8 - max(3, i)))
            self.blocks.append(nn.Sequential(conv(nin, nin, bias=bias, BN=BN, ws=ws, activ=activ),
                                        conv(nin, nout, bias=bias, BN=BN, ws=ws, activ=activ)))
        self.blocks = nn.ModuleList(self.blocks)

    def minibatchstd(self, input):
        # must add 1e-8 in std for stability
        return mean(torch.sqrt(input.var(dim=0, keepdim=True) + 1e-8), axis=[1, 2, 3], keepdim=True)

    def forward(self, input, x=None):
        if x is None:
            progress = self.maxRes
        else:
            progress = min(x, self.maxRes)

        alpha = progress - int(progress)

        y0 = self.fromRGBs[int(ceil(progress))](input)

        if progress % 1 != 0:
            y1 = F.avg_pool2d(input, kernel_size=2, stride=2)
            y1 = self.fromRGBs[int(progress)](y1)
            y0 = self.blocks[int(ceil(progress))](y0)
            y0 = alpha * F.avg_pool2d(y0, kernel_size=2, stride=2) + (1 - alpha) * y1

        for i in range(int(progress), 0, -1):
            y0 = self.blocks[i](y0)
            y0 = F.avg_pool2d(y0, kernel_size=2, stride=2)

        y = self.blocks[0](torch.cat((y0, self.minibatchstd(y0).expand_as(y0[:, 0].unsqueeze(1))), dim=1))

        return y.squeeze()


if __name__ == '__main__':
    from utils import to_var


    def param_number(net):
        n = 0
        for par in net.parameters():
            n += par.numel()
        return n


    # test in original configuration
    nch = 16
    G = Generator(nch=nch, bias=True, ws=True, pn=True)
    print(G)
    D = Discriminator(nch=nch, bias=True, ws=True)
    print(D)
    G.cuda()
    D.cuda()

    z = to_var(torch.rand(4, nch * 32, 1, 1))
    z.volatile = True
    print('##### Testing Generator #####')
    print(f'Generator has {param_number(G)} parameters')
    for i in range((G.maxRes + 1) * 2):
        print(i / 2, ' -> ', G(z, i / 2).size())
    print('##### Testing Discriminator #####')
    print(f'Generator has {param_number(D)} parameters')
    for i in range((G.maxRes + 1) * 2):
        print(i / 2, ' -> ', D(G(z, i / 2), i / 2).size())
