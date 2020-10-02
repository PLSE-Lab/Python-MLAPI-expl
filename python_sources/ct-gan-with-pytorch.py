#!/usr/bin/env python
# coding: utf-8

# I tried CT-GAN.
# 
# This is the first time to open kenel.
# 
# If there are any problems, please comment!
# 
# 
# reference of code:
# 
# https://www.kaggle.com/speedwagon/ralsgan-dogs
# 
# https://github.com/ozanciga/gans-with-pytorch
# 
# 
# paper:
# 
# https://arxiv.org/abs/1803.01541

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))
import time

# Any results you write to the current directory are saved as output.


# In[ ]:


start_kernel_time = time.time()


# In[ ]:


from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch.autograd import Variable
import matplotlib.pyplot as plt
import numpy as np
from torch import nn, optim
import torch.nn.functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from tqdm import tqdm_notebook as tqdm


# In[ ]:


batch_size = 32
nz = 128
lr = 0.0007
beta1 = 0.5
epochs = 99999999

real_label = 0.90
fake_label = 0.10

n_critic = 5
lambda_1 = 10
lambda_2 = 2
param_M = 0.2

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# In[ ]:


random_transforms = [
    #transforms.ColorJitter(brightness=0.75, contrast=0.75, saturation=0.75, hue=0.51), 
    transforms.RandomRotation(degrees=5)]
transform = transforms.Compose([transforms.Resize(64),
                                transforms.CenterCrop(64),
                                transforms.RandomHorizontalFlip(p=0.5),
                                transforms.RandomApply(random_transforms, p=0.3),
                                transforms.ToTensor(),
                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

train_data = datasets.ImageFolder('../input/all-dogs/', transform=transform)
train_loader = torch.utils.data.DataLoader(train_data, shuffle=True,
                                           batch_size=batch_size, num_workers=4)
                                           
imgs, label = next(iter(train_loader))
imgs = imgs.numpy().transpose(0, 2, 3, 1)


# # Model Defenition

# In[ ]:


class Generator(nn.Module):
    def __init__(self, nz=128, channels=3):
        super(Generator, self).__init__()
        
        self.nz = nz
        self.channels = channels
        
        def convlayer(n_input, n_output, k_size=4, stride=2, padding=0):
            block = [
                nn.ConvTranspose2d(n_input, n_output, kernel_size=k_size, stride=stride, padding=padding, bias=False),
                nn.BatchNorm2d(n_output),
                nn.ReLU(inplace=True),
            ]
            return block

        self.model = nn.Sequential(
            *convlayer(self.nz, 1024, 4, 1, 0), # Fully connected layer via convolution.
            *convlayer(1024, 512, 4, 2, 1),
            *convlayer(512, 256, 4, 2, 1),
            *convlayer(256, 128, 4, 2, 1),
            *convlayer(128, 64, 4, 2, 1),
            nn.ConvTranspose2d(64, self.channels, 3, 1, 1),
            nn.Tanh()
        )


    def forward(self, z):
        z = z.view(-1, self.nz, 1, 1)
        img = self.model(z)
        return img

class Discriminator(nn.Module):
    def __init__(self, channels=3):
        super(Discriminator, self).__init__()
        
        self.channels = channels

        def convlayer(n_input, n_output, k_size=4, stride=2, padding=0, bn=False):
            block = [nn.Conv2d(n_input, n_output, kernel_size=k_size, stride=stride, padding=padding, bias=False)]
            if bn:
                block.append(nn.BatchNorm2d(n_output))
            block.append(nn.LeakyReLU(0.2, inplace=True))
            return block
    
        self.model = nn.Sequential(
            *convlayer(self.channels, 32, 4, 2, 1),
            *convlayer(32, 64, 4, 2, 1),
            *convlayer(64, 128, 4, 2, 1, bn=True),
            *convlayer(128, 256, 4, 2, 1, bn=True),
        )
        
        self.output_layer = nn.Conv2d(256, 1, 4, 1, 0, bias=False)


    def forward(self, imgs, dropout=0.0, intermediate_output=False):
        u1 = self.model(imgs)
        u2 = self.output_layer(u1)
        out = torch.sigmoid(u2)
        if intermediate_output:
            return out.view(-1, 1), (u1.view(imgs.size(0), 256, -1)).mean(dim=2) # u1 is the D_(.), intermediate layer given in paper.
    
        return out.view(-1, 1)


# # Training Settings

# In[ ]:


# TODO: Use some initialization in the future.
def init_weights(m):
    if type(m) == nn.ConvTranspose2d:
        torch.nn.init.kaiming_normal(m.weight, mode='fan_out', nonlinearity='relu')
    elif type(m) == nn.Conv2d:
        torch.nn.init.kaiming_uniform_(m.weight, mode='fan_in', nonlinearity='leaky_relu')


# In[ ]:


cuda = torch.cuda.is_available()
Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

generator = Generator().to(device)
discriminator = Discriminator().to(device)

optimizer_D = optim.Adam(discriminator.parameters(), lr=lr, betas=(beta1, 0.999))
optimizer_G = optim.Adam(generator.parameters(), lr=lr, betas=(beta1, 0.999))
criterion = nn.BCELoss()


# In[ ]:


for epoch in range(epochs):
    if ((time.time()-start_kernel_time)/3600>8.0):
        break
    print('Epoch {}'.format(epoch))
    for ii, (real_images, _) in tqdm(enumerate(train_loader), total=len(train_loader)-1):
        if ((time.time()-start_kernel_time)/3600>8.0):
            break
        # == Discriminator update == #
        for iter in range(n_critic):
            if ((time.time()-start_kernel_time)/3600>8.0):
                break
            # Sample real and fake images, using notation in paper.
            x = real_images.to(device)
            noise = torch.randn(real_images.shape[0], nz, 1, 1, device=device)

            x_tilde = Variable(generator(noise), requires_grad=True)
            epsilon = Variable(Tensor(real_images.shape[0], 1, 1, 1).uniform_(0, 1))

            x_hat = epsilon*x + (1 - epsilon)*x_tilde
            x_hat = torch.autograd.Variable(x_hat, requires_grad=True)

            # Put the interpolated data through critic.
            dw_x = discriminator(x_hat)
            grad_x = torch.autograd.grad(outputs=dw_x, inputs=x_hat,
                                         grad_outputs=Variable(Tensor(real_images.size(0), 1).fill_(real_label), requires_grad=False),
                                         create_graph=True, retain_graph=True, only_inputs=True)
            grad_x = grad_x[0].view(real_images.size(0), -1)
            grad_x = grad_x.norm(p=2, dim=1)

            # Update critic.
            optimizer_D.zero_grad()

            # Standard WGAN loss.
            d_wgan_loss = torch.mean(discriminator(x_tilde)) - torch.mean(discriminator(x))

            # WGAN-GP loss.
            d_wgp_loss = torch.mean((grad_x - 1)**2)

            ###### Consistency term. ######
            dw_x1, dw_x1_i = discriminator(x, dropout=0.5, intermediate_output=True) # Perturb the input by applying dropout to hidden layers.
            dw_x2, dw_x2_i = discriminator(x, dropout=0.5, intermediate_output=True)
            # Using l2 norm as the distance metric d, referring to the official code (paper ambiguous on d).
            second_to_last_reg = ((dw_x1_i-dw_x2_i) ** 2).mean(dim=1).unsqueeze_(1).unsqueeze_(2).unsqueeze_(3)
            d_wct_loss = (dw_x1-dw_x2) ** 2                          + 0.1 * second_to_last_reg                          - param_M
            d_wct_loss, _ = torch.max(d_wct_loss, 0) # torch.max returns max, and the index of max
            d_wct_loss = d_wct_loss.sum()

            # Combined loss.
            d_loss = d_wgan_loss + lambda_1*d_wgp_loss + lambda_2*d_wct_loss

            d_loss.backward()
            optimizer_D.step()

        # == Generator update == #
        noise = torch.randn(batch_size, nz, 1, 1, device=device)
        imgs_fake = generator(noise)

        optimizer_G.zero_grad()
        
        labels = torch.full((batch_size, 1), real_label, device=device)
        output = discriminator(imgs_fake)
        g_loss = criterion(output, labels)

        g_loss.backward()
        optimizer_G.step()


# In[ ]:


def show_generated_img():
    noise = torch.randn(1, nz, 1, 1, device=device)
    gen_image = generator(noise).to("cpu").clone().detach().squeeze(0)
    gen_image = gen_image.numpy().transpose(1, 2, 0)
    plt.imshow(gen_image)
    plt.show()


# In[ ]:


for _ in range(25):
    show_generated_img()


# In[ ]:


if not os.path.exists('../output_images'):
    os.mkdir('../output_images')
im_batch_size = 50
n_images=10000
for i_batch in range(0, n_images, im_batch_size):
    gen_z = torch.randn(im_batch_size, nz, 1, 1, device=device)
    gen_images = generator(gen_z)
    images = gen_images.to("cpu").clone().detach()
    images = images.numpy().transpose(0, 2, 3, 1)
    for i_image in range(gen_images.size(0)):
        save_image(gen_images[i_image, :, :, :], os.path.join('../output_images', f'image_{i_batch+i_image:05d}.png'))


import shutil
shutil.make_archive('images', 'zip', '../output_images')

