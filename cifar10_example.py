import argparse
import os
import copy
import matplotlib
import tqdm

matplotlib.use('agg')
import matplotlib.pyplot as plt
import numpy as np

import torch
import torch.nn.functional as F
from torch.optim import Adam
from torch.utils.data import DataLoader
from torch.backends import cudnn
from torchvision import transforms
from torchvision.utils import save_image
from torchvision.datasets import CIFAR10

from model import Generator, Discriminator
from utils import gradient_penalty, Progress, hypersphere, exp_mov_avg, state_dict_to_cpu, requires_grad_

parser = argparse.ArgumentParser()
parser.add_argument('--data', type=str, default='data', help='directory containing the data')
parser.add_argument('--outd', default='Results', help='directory to save results')
parser.add_argument('--outf', default='Images', help='folder to save synthetic images')
parser.add_argument('--outl', default='Losses', help='folder to save Losses')
parser.add_argument('--outm', default='Models', help='folder to save models')

parser.add_argument('--epochs', type=int, default=200, help='number of training epochs')
parser.add_argument('--workers', type=int, default=8, help='number of data loading workers')
parser.add_argument('--batch-sizes', nargs='*', default=[16, 16, 16, 16],
                    help='list of batch sizes during the training')
parser.add_argument('--nch', type=int, default=16, help='base number of channel for networks')
parser.add_argument('--BN', action='store_true', help='use BatchNorm in G and D')
parser.add_argument('--WS', action='store_true', help='use WeightScale in G and D')
parser.add_argument('--PN', action='store_true', help='use PixelNorm in G')

parser.add_argument('--n_iter', type=int, default=20, help='number of epochs to train before changing the progress')
parser.add_argument('--lambda-gp', type=float, default=10, help='lambda for gradient penalty')
parser.add_argument('--gamma-gp', type=float, default=750, help='gamma for gradient penalty')
parser.add_argument('--e_drift', type=float, default=0.001, help='epsilon drift for discriminator loss')
parser.add_argument('--saveimages', type=int, default=1, help='number of epochs between saving image examples')
parser.add_argument('--savenum', type=int, default=64, help='number of examples images to save')
parser.add_argument('--savemodel', type=int, default=10, help='number of epochs between saves of model')
parser.add_argument('--savemaxsize', action='store_true',
                    help='save sample images at max resolution instead of real resolution')

opt = parser.parse_args()
print(opt)

DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
MAX_RES = 3  # for 32x32 output

dataset = CIFAR10(opt.data, download=True, train=True,
                  transform=transforms.Compose([transforms.RandomHorizontalFlip(),
                                                transforms.ToTensor(),
                                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]))

# creating output folders
os.makedirs(opt.outd)
for f in [opt.outf, opt.outl, opt.outm]:
    os.makedirs(os.path.join(opt.outd, f))

# Model creation and init
G = Generator(max_res=MAX_RES, nch=opt.nch, bn=opt.BN, ws=opt.WS, pn=opt.PN).to(DEVICE)
D = Discriminator(max_res=MAX_RES, nch=opt.nch, bn=opt.BN, ws=opt.WS).to(DEVICE)
Gs = copy.deepcopy(G)
Gs.eval()
requires_grad_(Gs, False)

optimizerG = Adam(G.parameters(), lr=1e-3, betas=(0, 0.99))
optimizerD = Adam(D.parameters(), lr=1e-3, betas=(0, 0.99))

global_step = 0
total = 2
d_losses = []
d_W_losses = []
g_losses = []
P = Progress(opt.n_iter, MAX_RES, opt.batch_sizes)

z_save = hypersphere(torch.randn(opt.savenum, opt.nch * 32, 1, 1, device=DEVICE))

P.progress(0, 1, total)
# Creation of DataLoader
data_loader = DataLoader(dataset,
                         batch_size=P.batch_size,
                         shuffle=True,
                         num_workers=opt.workers,
                         drop_last=True,
                         pin_memory=True)

for epoch in range(opt.epochs):

    g_loss_epoch = []
    d_loss_epoch = []
    d_W_loss_epoch = []

    cudnn.benchmark = True
    P.progress(epoch, 1, total)

    if P.batch_size != data_loader.batch_size:
        # modify DataLoader at each change in resolution to vary the batch-size as the resolution increases
        data_loader = DataLoader(dataset,
                                 batch_size=P.batch_size,
                                 shuffle=True,
                                 num_workers=opt.workers,
                                 drop_last=True,
                                 pin_memory=True)

    total = len(data_loader)
    pbar = tqdm.tqdm(enumerate(data_loader), dynamic_ncols=True, total=len(data_loader),
                     desc='Epoch {}/{}'.format(epoch, opt.epochs))
    for i, (images, _) in pbar:
        P.progress(epoch, i + 1, total + 1)
        global_step += 1
        images = images.to(DEVICE, non_blocking=True)  # Move images to GPU

        # ============= Train the discriminator ============= #

        # compute fake images with G
        z = hypersphere(torch.randn(P.batch_size, opt.nch * 32, 1, 1, device=DEVICE))
        with torch.no_grad():
            fake_images = G(z, P.p).clamp(-1, 1)

        # compute scores for real images
        images = P.resize(images)
        D_real = D(images, P.p)
        D_realm = D_real.mean()

        # compute scores for fake images
        D_fake = D(fake_images, P.p)
        D_fakem = D_fake.mean()

        # compute gradient penalty for WGAN-GP loss
        gp = gradient_penalty(D, images, fake_images, P.p, opt.lambda_gp, opt.gamma_gp)

        # prevent D_real from drifting too much from 0
        drift = (D_real ** 2).mean() * opt.e_drift

        # Total loss
        d_loss = D_fakem - D_realm
        d_W_loss = d_loss + gp + drift

        # Optimize
        D.zero_grad()
        d_W_loss.backward()
        optimizerD.step()

        d_loss_epoch.append(d_loss.item())
        d_W_loss_epoch.append(d_W_loss.item())

        # =============== Train the generator =============== #

        z = hypersphere(torch.randn(P.batch_size, opt.nch * 32, 1, 1, device=DEVICE))
        fake_images = G(z, P.p).clamp(-1, 1)
        # compute scores with new fake images
        G_fake = D(fake_images, P.p)
        G_fakem = G_fake.mean()
        # no need to compute D_real as it does not affect G
        g_loss = -G_fakem

        # Optimize
        G.zero_grad()
        g_loss.backward()
        optimizerG.step()

        g_loss_epoch.append(g_loss.item())

        # update Gs with exponential moving average
        exp_mov_avg(Gs, G, alpha=0.999, global_step=global_step)

        pbar.set_postfix_str('d_loss: {:.3f} - d_loss_W: {:.3f} - progress: {:.2f}'.format(
            d_loss.item(), d_W_loss.item(), P.p))

    pbar.set_postfix_str('d_loss: {:.3f} - d_loss_W: {:.3f} - progress: {:.2f}'.format(
        np.mean(d_loss_epoch), np.mean(d_W_loss_epoch), P.p))

    d_losses.extend(d_loss_epoch)
    d_W_losses.extend(d_W_loss_epoch)
    g_losses.extend(g_loss_epoch)

    np.save(os.path.join(opt.outd, opt.outl, 'd_losses.npy'), d_losses)
    np.save(os.path.join(opt.outd, opt.outl, 'd_losses_W.npy'), d_W_losses)
    np.save(os.path.join(opt.outd, opt.outl, 'g_losses.npy'), g_losses)

    cudnn.benchmark = False
    if not (epoch + 1) % opt.saveimages:
        # plotting loss values, g_losses is not plotted as it does not represent anything in the WGAN-GP
        ax = plt.subplot()
        ax.plot(np.linspace(0, epoch + 1, len(d_losses)), d_losses, '-b', label='d_loss', linewidth=0.1)
        ax.legend(loc=1)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.set_title('Progress: {:.2f}'.format(P.p))
        plt.savefig(os.path.join(opt.outd, opt.outl, 'D_losses_{}.png'.format(epoch)), dpi=200, bbox_inches='tight')
        plt.clf()

        with torch.no_grad():
            fake_images = Gs(z_save, P.p).clamp(-1, 1)
            if opt.savemaxsize:
                if fake_images.size(-1) != 4 * 2 ** MAX_RES:
                    fake_images = F.interpolate(fake_images, size=4 * 2 ** MAX_RES)
        save_image(fake_images,
                   os.path.join(opt.outd, opt.outf, 'fake_images-{:04d}-p{:.2f}.png'.format(epoch, P.p)),
                   nrow=8, pad_value=0, normalize=True, range=(-1, 1))

    if P.p >= P.pmax and not epoch % opt.savemodel:
        torch.save(state_dict_to_cpu(G.state_dict()),
                   os.path.join(opt.outd, opt.outm, 'G_nch-{}_epoch-{}.pth'.format(opt.nch, epoch)))
        torch.save(state_dict_to_cpu(D.state_dict()),
                   os.path.join(opt.outd, opt.outm, 'D_nch-{}_epoch-{}.pth'.format(opt.nch, epoch)))
        torch.save(state_dict_to_cpu(Gs.state_dict()),
                   os.path.join(opt.outd, opt.outm, 'Gs_nch-{}_epoch-{}.pth'.format(opt.nch, epoch)))
