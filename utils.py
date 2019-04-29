from collections import OrderedDict
from math import ceil

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import grad


def hypersphere(z, radius=1):
    """Normalize the rows of z to lie on a hypersphere of given radius.

    Parameters
    ----------
    z : torch.Tensor
        Tensor containing vectors as rows.
    radius : float
        Radius of the hypersphere.

    Returns
    -------
    normalized_z : torch.Tensor
        Tensor containing the normalized vectors of z.

    """
    return z * radius / z.norm(p=2, dim=1, keepdim=True)


def exp_mov_avg(Gs, G, alpha=0.999, global_step=999):
    """Update the parameters (in-place for speed) of Gs as an exponential average of G.

    Parameters
    ----------
    Gs : nn.Module
        Exponential average of G.
    G : nn.Module
        Source network for the weights to average.
    alpha : float
        Coefficient for the averaging.
    global_step : int
        Current step of the training. In the early iterations, the source network (G) weights change a lot, so the
        exponential average will not properly follow. In these iterations, alpha is replaces by a smaller value to
        account for that faster change.
    """
    alpha = min(1 - 1 / (global_step + 1), alpha)
    for ema_param, param in zip(Gs.parameters(), G.parameters()):
        ema_param.data.mul_(alpha).add_(1 - alpha, param.data)


class Progress:
    """Determine the progress parameter of the training given the epoch and the progression in the epoch

    Parameters
    ----------
    n_iter : int
        Number of epochs before changing the progress.
    pmax : int
        Maximum progress of the training.
    batch_sizes : list
        List of batch_size to adopt during the training.
    """

    def __init__(self, n_iter, pmax, batch_sizes):

        assert n_iter > 0 and isinstance(n_iter, int), 'n_iter must be int >= 1'
        assert pmax >= 0 and isinstance(pmax, int), 'pmax must be int >= 0'
        batch_sizes = list(map(int, batch_sizes))
        assert isinstance(batch_sizes, list) and \
               all(isinstance(x, int) for x in batch_sizes) and \
               all(x > 0 for x in batch_sizes) and \
               len(batch_sizes) == pmax + 1, \
            'batch_sizes must be a list of int > 0 and of length pmax+1'

        self.n_iter = n_iter
        self.pmax = pmax
        self.p = 0
        self.batch_sizes = batch_sizes

    def progress(self, epoch, i, total):
        """Update the progress given the epoch and the iteration of the epoch.

        Parameters
        ----------
        epoch : int
            Current epoch.
        i : int
            Current iteration in the epoch.
        total : int
            Total number of iterations in the epoch.

        Returns
        -------
        progress : float
            Current progress of the training.
        """
        x = (epoch + i / total) / self.n_iter
        self.p = min(max(int(x / 2), x - ceil(x / 2), 0), self.pmax)
        return self.p

    def resize(self, images):
        """Resize the images  w.r.t the current value of the progress.

        Parameters
        ----------
        images : torch.Tensor
            Batch of images

        Returns
        -------
        resized_images: torch.Tensor
            Batch of resized images.
        """
        x = int(ceil(self.p))
        if x >= self.pmax:
            return images
        else:
            return F.interpolate(images, 4 * 2 ** x)

    @property
    def batch_size(self):
        """Returns the current batchSize w.r.t the current value of the progress"""
        x = int(ceil(self.p))
        return self.batch_sizes[x]


def gradient_penalty(discriminator: nn.Module,
                     real_data: torch.Tensor,
                     fake_data: torch.Tensor,
                     progress: float,
                     lambda_gp: float,
                     gamma: float = 1) -> torch.Tensor:
    """Computes the gradient penalty as defined in "Improved Training of Wasserstein GANs"
    (https://arxiv.org/abs/1704.00028)

    Parameters
    ----------
    discriminator : nn.Module
        Discriminator network.
    real_data : torch.Tensor
        Batch of real images.
    fake_data : torch.Tensor
        Batch of fake images produced by the generator.
    progress : float
        Progress parameterer condition the input size of the discriminator
    lambda_gp : float
        Gradient penalty coefficient.
    gamma : float
        Regularization term of the gradient penalty, augment to minimize "ghosts".
    Returns
    -------
    gradient_penalty : torch.Tensor
        Gradient penalty averaged on the batch.
    """
    alpha = torch.rand(real_data.shape[0], 1, 1, 1, requires_grad=True, device=real_data.device)
    # randomly mix real and fake data
    interpolates = real_data + alpha * (fake_data - real_data)
    # compute output of D for interpolated inpu
    disc_interpolates = discriminator(interpolates, progress)
    # compute gradients w.r.t the interpolated outputs
    gradients = grad(outputs=disc_interpolates, inputs=interpolates, grad_outputs=torch.ones_like(disc_interpolates),
                     create_graph=True, retain_graph=True, only_inputs=True)[0].flatten(1)
    gradient_penalty = (((gradients.norm(2, dim=1) - gamma) / gamma) ** 2).mean() * lambda_gp

    return gradient_penalty


def state_dict_to_cpu(state_dict: OrderedDict):
    """Moves a state_dict to cpu and removes the module. added by DataParallel.

    Parameters
    ----------
    state_dict : OrderedDict
        State_dict containing the tensors to move to cpu.

    Returns
    -------
    new_state_dict : OrderedDict
        State_dict on cpu.
    """
    new_state = OrderedDict()
    for k in state_dict.keys():
        newk = k.replace('module.', '')  # remove "module." if model was trained using DataParallel
        new_state[newk] = state_dict[k].cpu()
    return new_state


def requires_grad_(model: nn.Module, requires_grad: bool):
    """Change the requires_grad attribute of all parameters of a given model.

    Parameters
    ----------
    model : nn.Module
        Model on which to change the require_grad attribute.
    requires_grad : bool
        New value of requires_grad attribute,
    """
    for param in model.parameters():
        param.requires_grad_(requires_grad)
