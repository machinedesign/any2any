from __future__ import print_function
from collections import defaultdict
from datetime import datetime
from visdom import Visdom

import pandas as pd
import time
import sys
import numpy as np
import argparse
import os
import random
import torch
from torch.nn.init import xavier_uniform
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
from torchvision import datasets
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch.autograd import Variable

from torchvision.models import alexnet

from clize import run

class Discr(nn.Module):
    def __init__(self, nc=3, ndf=64, no=1, imagesize=32):
        super(Discr, self).__init__()

        if imagesize == 64:
            self.main = nn.Sequential(
                nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
                nn.LeakyReLU(0.2, inplace=True),

                nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
                nn.BatchNorm2d(ndf * 2),
                nn.LeakyReLU(0.2, inplace=True),

                nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
                nn.BatchNorm2d(ndf * 4),
                nn.LeakyReLU(0.2, inplace=True),

                nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
                nn.BatchNorm2d(ndf * 8),
                nn.LeakyReLU(0.2, inplace=True),

                nn.Conv2d(ndf * 8, no, 4, 1, 0, bias=False),
                nn.Sigmoid()
            )
        elif imagesize == 32:
            self.main = nn.Sequential(
                nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
                nn.LeakyReLU(0.2, inplace=True),

                nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
                nn.BatchNorm2d(ndf * 2),
                nn.LeakyReLU(0.2, inplace=True),

                nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
                nn.BatchNorm2d(ndf * 4),
                nn.LeakyReLU(0.2, inplace=True),

                nn.Conv2d(ndf * 4, no, 4, 1, 0, bias=False),
                nn.Sigmoid()
            )

    def forward(self, input):
        return self.main(input)

class Gen(nn.Module):
    def __init__(self, nz=3, ngf=64, no=3):
        super(Gen, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(nz, ngf, 3, 1, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(ngf, ngf * 2, 3, 1, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(ngf * 2, ngf * 4, 3, 1, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(ngf * 4, no, 3, 1, 1),
            #nn.BatchNorm2d(no),
            nn.Tanh()
        )
    def forward(self, input):
        return self.conv(input)


class GenDisco(nn.Module):
    def __init__(self, nz=3, no=3, extra_layers=False):
        super(GenDisco, self).__init__()
        if extra_layers == True:
            self.main = nn.Sequential(
                nn.Conv2d(nz, 64, 4, 2, 1, bias=False),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(64, 64 * 2, 4, 2, 1, bias=False),
                nn.BatchNorm2d(64 * 2),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(64 * 2, 64 * 4, 4, 2, 1, bias=False),
                nn.BatchNorm2d(64 * 4),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(64 * 4, 64 * 8, 4, 2, 1, bias=False),
                nn.BatchNorm2d(64 * 8),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(64 * 8, 100, 4, 1, 0, bias=False),
                nn.BatchNorm2d(100),
                nn.LeakyReLU(0.2, inplace=True),

                nn.ConvTranspose2d(100, 64 * 8, 4, 1, 0, bias=False),
                nn.BatchNorm2d(64 * 8),
                nn.ReLU(True),
                nn.ConvTranspose2d(64 * 8, 64 * 4, 4, 2, 1, bias=False),
                nn.BatchNorm2d(64 * 4),
                nn.ReLU(True),
                nn.ConvTranspose2d(64 * 4, 64 * 2, 4, 2, 1, bias=False),
                nn.BatchNorm2d(64 * 2),
                nn.ReLU(True),
                nn.ConvTranspose2d(64 * 2,     64, 4, 2, 1, bias=False),
                nn.BatchNorm2d(64),
                nn.ReLU(True),
                nn.ConvTranspose2d(    64,      no, 4, 2, 1, bias=False),
                nn.Tanh()
            )

        if extra_layers == False:
            self.main = nn.Sequential(
                nn.Conv2d(nz, 64, 4, 2, 1, bias=False),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(64, 64 * 2, 4, 2, 1, bias=False),
                nn.BatchNorm2d(64 * 2),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(64 * 2, 64 * 4, 4, 2, 1, bias=False),
                nn.BatchNorm2d(64 * 4),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(64 * 4, 64 * 8, 4, 2, 1, bias=False),
                nn.BatchNorm2d(64 * 8),
                nn.LeakyReLU(0.2, inplace=True),

                nn.ConvTranspose2d(64 * 8, 64 * 4, 4, 2, 1, bias=False),
                nn.BatchNorm2d(64 * 4),
                nn.ReLU(True),
                nn.ConvTranspose2d(64 * 4, 64 * 2, 4, 2, 1, bias=False),
                nn.BatchNorm2d(64 * 2),
                nn.ReLU(True),
                nn.ConvTranspose2d(64 * 2,     64, 4, 2, 1, bias=False),
                nn.BatchNorm2d(64),
                nn.ReLU(True),
                nn.ConvTranspose2d(    64,      no, 4, 2, 1, bias=False),
                nn.Tanh()
            )

    def forward(self, input):
        return self.main( input )


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)
    elif classname.find('Linear') != -1:
        xavier_uniform(m.weight.data)
        m.bias.data.zero_()

def _flatten(x):
    return x.view(x.size(0), -1)

def main(*, imageSize=32, dirs='.', batchSize=128, 
         nThreads=1, niter=100000, lr=0.0002, beta1=0.5, 
         ncols=3,
         outf='samples'):
    if not os.path.exists(outf):
        os.mkdir(outf)

    viz = Visdom('http://romeo163')
    win = viz.line(X=np.array([0]), Y=np.array([0]), opts=dict(title='any2any, started {}'.format(datetime.now())))

    viz.line(X=np.array([0]), Y=np.array([0]), update='append', win=win)

    transform = transforms.Compose([
        transforms.Scale(imageSize),
        transforms.CenterCrop(imageSize),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])
    
   
    dataloaders = []
    for dirname in dirs.split(','):
        dataset = datasets.ImageFolder(dirname, transform)
        dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=batchSize, shuffle=True, num_workers=nThreads)
        dataloaders.append(dataloader)
    nb_minibatches = min(map(len, dataloaders))

    code_size = 128

    netG = GenDisco(nz=ncols+code_size, no=ncols)
    netG.apply(weights_init)
 
    netD = Discr(nc=ncols+code_size, imagesize=imageSize)
    netD.apply(weights_init)

    netE_G = Discr(nc=ncols, no=code_size, imagesize=imageSize)
    netE_G.apply(weights_init)
    netE_D = Discr(nc=ncols, no=code_size, imagesize=imageSize)
    netE_D.apply(weights_init)

    input = torch.FloatTensor(batchSize, ncols, imageSize, imageSize)
    input = Variable(input)

    target = torch.FloatTensor(batchSize, ncols, imageSize, imageSize)
    target = Variable(target)

    label = torch.FloatTensor(batchSize)
    label = Variable(label)
 
    real_label = 1
    fake_label = 0

    criterion = nn.BCELoss()
    #criterion = nn.MSELoss()
    criterion = criterion.cuda()
    def wsgan(output, label):
        label = label * 2 - 1
        return torch.mean(output * label)
    #criterion = wsgan

    netE_G = netE_G.cuda()
    netE_D = netE_D.cuda()
    netG = netG.cuda()
    netD = netD.cuda()
    input = input.cuda()
    target = target.cuda()
    label = label.cuda()

    optimizerD = optim.Adam(netD.parameters(), lr=lr, betas = (beta1, 0.999))
    optimizerG = optim.Adam(netG.parameters(), lr=lr, betas = (beta1, 0.999))
    optimizerE_G = optim.Adam(netE_G.parameters(), lr=lr, betas = (beta1, 0.999))
    optimizerE_D = optim.Adam(netE_D.parameters(), lr=lr, betas = (beta1, 0.999))

    stats = defaultdict(list)
    j = 0
    for epoch in range(niter):
        dataiters = [iter(loader) for loader in dataloaders]
        for i in range(nb_minibatches):
            if len(dataloaders) == 1:
                c1 = 0
                c2 = 0
            else:
                c1 = np.random.randint(0, len(dataloaders) - 1)
                c2 = np.random.randint(c1 + 1, len(dataloaders))
            source_data = next(dataiters[c1])
            target_data = next(dataiters[c2])

            t = time.time()
            netD.zero_grad()
            netE_D.zero_grad()
            
            source_images, _ = source_data
            target_images, _ = target_data

            if source_images.size() != target_images.size():
                continue
            #source_images.uniform_(-1, 1)

            if ncols == 1:
                source_images = source_images[:, 0:1]
                target_images = target_images[:, 0:1]

            errD = 0.
            input.data.resize_(source_images.size()).copy_(source_images)
            target.data.resize_(target_images.size()).copy_(target_images)
            batch_size = source_images.size(0)
            errD_vals = []
            for X, Y in (input, target), (target, input): 
                X_ = X.detach()
                Y_ = Y.detach()

                label.data.resize_(batch_size).fill_(real_label)
                
                code_g = netE_G(Y_)
                code_g = code_g.repeat(1, 1, imageSize, imageSize)
                code_g = code_g.mean(0).repeat(input.size(0), 1, 1, 1)

                code_d = netE_D(Y_)
                code_d = code_d.repeat(1, 1, imageSize, imageSize)
                code_d = code_d.mean(0).repeat(input.size(0), 1, 1, 1)
                
                target_and_code = torch.cat((Y_, code_d), 1)
                output = netD(target_and_code)
                output = output.view(-1, 1)
                errD_real = criterion(output, label)
                errD_real.backward()
                # train with fake
                input_and_code = torch.cat((X_, code_g), 1)
                fake = netG(input_and_code)
                label.data.fill_(fake_label)
                fake_and_code = torch.cat((fake, code_d), 1)
                output = netD(fake_and_code.detach())
                output = output.view(-1, 1)
                errD_fake = criterion(output, label)
                errD_fake.backward()
                errD_vals += [errD_real.data[0], errD_fake.data[0]]
                errD += errD_real + errD_fake
            optimizerD.step()
            optimizerE_D.step()

            netG.zero_grad()
            netE_G.zero_grad()
            label.data.fill_(real_label) # fake labels are real for generator cost
            errG = 0.
            fakes = []
            for X, Y in (input, target), (target, input): 
                X_ = X.detach()
                Y_ = Y.detach()
                
                code_g = netE_G(Y_)
                code_g = code_g.repeat(1, 1, imageSize, imageSize)
                code_g = code_g.mean(0).repeat(input.size(0), 1, 1, 1)

                code_d = netE_D(Y_)
                code_d = code_d.repeat(1, 1, imageSize, imageSize)
                code_d = code_d.mean(0).repeat(input.size(0), 1, 1, 1)

                input_and_code = torch.cat((X_, code_g), 1) 
                fake = netG(input_and_code)
                fakes.append(fake.data)

                src_code_g = netE_G(X_)
                src_code_g = src_code_g.repeat(1, 1, imageSize, imageSize)
                src_code_g = src_code_g.mean(0).repeat(input.size(0), 1, 1, 1)
                fake_and_src_code = torch.cat((fake, src_code_g), 1)
                rec = netG(fake_and_src_code)

                fake_and_code = torch.cat((fake, code_d), 1)
                output = netD(fake_and_code)
                output = output.view(-1, 1)

                gan_loss = criterion(output, label)
                rec_loss = 0.01 * ((rec - X_)**2).sum() / rec.size(0)
                loss = gan_loss + rec_loss
                loss.backward()
                errG += loss + rec_loss
            optimizerG.step()
            optimizerE_G.step()

            delta_t = time.time() - t
            stats['iter'].append(j)
            stats['errG'].append(errG.data[0])
            stats['errD'].append(errD.data[0])
            stats['rec'].append(rec_loss.data[0])

            print('[%d/%d][%d/%d] Loss_D: %.4f Loss_G: %.4f, time : %.4f'
                  % (epoch, niter, i, nb_minibatches,
                     errD.data[0], errG.data[0], delta_t))
            viz.updateTrace(X=np.array([stats['iter'][-1]]), Y=np.array([stats['errG'][-1]]), win=win, name='errG')
            viz.updateTrace(X=np.array([stats['iter'][-1]]), Y=np.array([stats['rec'][-1]]), win=win, name='rec')
            viz.updateTrace(X=np.array([stats['iter'][-1]]), Y=np.array([stats['errD'][-1]]), win=win, name='errD')

            if j % 10 == 0:
                pd.DataFrame(stats).to_csv('{}/stats.csv'.format(outf), index=False)
                # the first 64 samples from the mini-batch are saved.
                folder = '{}/cl_{}_{}'.format(outf, c1, c2)
                if not os.path.exists(folder):
                    os.mkdir(folder)
                vutils.save_image((source_images[0:64,:,:,:]+1)/2., os.path.join(folder, 'source.png'), nrow=8)
                vutils.save_image((target_images[0:64,:,:,:]+1)/2., os.path.join(folder, 'target.png'), nrow=8)
                for name, f in (('source', fakes[0]), ('target', fakes[1])):
                    img_folder = os.path.join(folder, name)
                    if not os.path.exists(img_folder):
                        os.mkdir(img_folder)
                    vutils.save_image((f[0:64,:,:,:]+1)/2., '%s/epoch_%03d.png' % (img_folder, epoch), nrow=8)
            j += 1

        # do checkpointing
        torch.save(netG.state_dict(), '%s/netG_epoch_%d.pth' % (outf, epoch))
        torch.save(netD.state_dict(), '%s/netD_epoch_%d.pth' % (outf, epoch))

if __name__ == '__main__':
    run(main)
