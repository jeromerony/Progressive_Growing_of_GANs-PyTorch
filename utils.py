from math import ceil

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable, grad


def to_var(x, requires_grad=False):
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x, requires_grad=requires_grad)


def weights_init(m):
    if type(m) == nn.Conv2d or type(m) == nn.ConvTranspose2d:
        nn.init.xavier_normal(m.weight.data)
    elif type(m) == nn.BatchNorm2d:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


def hypersphere(z, radius=1):
    return z * radius / z.norm(p=2, dim=1, keepdim=True)


def exp_mov_avg(Gs, G, alpha=0.999):
    for state in G.state_dict().keys():
        Gs.state_dict()[state].copy_((1 - alpha) * G.state_dict()[state] + alpha * Gs.state_dict()[state])


def normal(*sizes):
    """Generates normal noise: faster than torch.randn(*sizes).cuda() when using a GPU"""
    if torch.cuda.is_available():
        return torch.cuda.FloatTensor(*sizes).normal_()
    else:
        return torch.randn(*sizes)


class Progress:
    """Determine the progress parameter of the training given the epoch and the progression in the epoch
    Args:
          n_iter (int): the number of epochs before changing the progress,
          pmax (int): the maximum progress of the training.
          batchSizeList (list): the list of the batchSize to adopt during the training
    """

    def __init__(self, n_iter, pmax, batchSizeList):
        assert n_iter > 0 and isinstance(n_iter, int), 'n_iter must be int >= 1'
        assert pmax >= 0 and isinstance(pmax, int), 'pmax must be int >= 0'
        assert isinstance(batchSizeList, list) and \
               all(isinstance(x, int) for x in batchSizeList) and \
               all(x > 0 for x in batchSizeList) and \
               len(batchSizeList) == pmax + 1, \
            'batchSizeList must be a list of int > 0 and of length pmax+1'

        self.n_iter = n_iter
        self.pmax = pmax
        self.p = 0
        self.batchSizeList = batchSizeList

    def progress(self, epoch, i, total):
        """Update the progress given the epoch and the iteration of the epoch
        Args:
            epoch (int): batch of images to resize
            i (int): iteration in the epoch
            total (int): total number of iterations in the epoch
        """
        x = (epoch + i / total) / self.n_iter
        self.p = min(max(int(x / 2), x - ceil(x / 2), 0), self.pmax)
        return self.p

    def resize(self, images):
        """Resize the images  w.r.t the current value of the progress.
        Args:
            images (Variable or Tensor): batch of images to resize
        """
        x = int(ceil(self.p))
        if x >= self.pmax:
            return images
        else:
            return F.adaptive_avg_pool2d(images, 4 * 2 ** x)

    @property
    def batchSize(self):
        """Returns the current batchSize w.r.t the current value of the progress"""
        x = int(ceil(self.p))
        return self.batchSizeList[x]


class GradientPenalty:
    """Computes the gradient penalty as defined in "Improved Training of Wasserstein GANs"
    (https://arxiv.org/abs/1704.00028)
    Args:
        batchSize (int): batch-size used in the training. Must be updated w.r.t the current batchsize
        lambdaGP (float): coefficient of the gradient penalty as defined in the article
        gamma (float): regularization term of the gradient penalty, augment to minimize "ghosts"
    """

    def __init__(self, batchSize, lambdaGP, gamma=1):
        self.batchSize = batchSize
        self.lambdaGP = lambdaGP
        self.gamma = gamma

    def __call__(self, netD, real_data, fake_data, progress):
        alpha = torch.rand(self.batchSize, 1, 1, 1).cuda()
        # randomly mix real and fake data
        interpolates = real_data + alpha * (fake_data - real_data)
        interpolates = to_var(interpolates, requires_grad=True)
        # compute output of D for interpolated input
        disc_interpolates = netD(interpolates, progress)
        # compute gradients w.r.t the interpolated outputs
        gradients = grad(outputs=disc_interpolates, inputs=interpolates,
                         grad_outputs=torch.ones(disc_interpolates.size()).cuda(),
                         create_graph=True, retain_graph=True, only_inputs=True)[0]
        gradient_penalty = (((gradients.norm(2, dim=1) - self.gamma) / self.gamma) ** 2).mean() * self.lambdaGP

        return gradient_penalty
