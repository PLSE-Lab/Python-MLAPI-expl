#!/usr/bin/env python
# coding: utf-8

# ## Loading Libraries

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
# Any results you write to the current directory are saved as output.
import argparse
import math
import sys
import matplotlib.pyplot as plt

import torchvision.transforms as transforms
from torchvision.utils import save_image

from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable

import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd
import torch
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn, optim
import torch.nn.functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm_notebook as tqdm
from time import time
from PIL import Image
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
import torchvision
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch.autograd import Variable
from torch import nn, optim
import torch.nn.functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image
import matplotlib.image as mpimg
import torchvision
import torchvision.datasets as dset
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch.autograd import Variable
import xml.etree.ElementTree as ET
import random


# In[ ]:


def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
seed_everything()


# In[ ]:


class opt:
    n_epochs = 1000
    batch_size = 64
    lr = 0.0002
    b1= 0.5
    b2 = 0.999
    latent_dim=100
    img_size = 64
    channels = 3
    n_critic = 60
    clip_value = 0.01
    sample_interval = 50


# In[ ]:


class DataGenerator(Dataset):
    def __init__(self, directory, transform=None, n_samples=np.inf):
        self.directory = directory
        self.transform = transform
        self.n_samples = n_samples

        self.samples = self._load_subfolders_images(directory)
        if len(self.samples) == 0:
            raise RuntimeError("Found 0 files in subfolders of: {}".format(directory))

    def _load_subfolders_images(self, root):
        IMG_EXTENSIONS = (
        '.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif', '.tiff', '.webp')

        def is_valid_file(x):
            return torchvision.datasets.folder.has_file_allowed_extension(x, IMG_EXTENSIONS)

        required_transforms = torchvision.transforms.Compose([
                torchvision.transforms.Resize(64),
                torchvision.transforms.CenterCrop(64),
        ])

        imgs = []
        paths = []
        for root, _, fnames in sorted(os.walk(root)):
            for fname in sorted(fnames)[:min(self.n_samples, 999999999999999)]:
                path = os.path.join(root, fname)
                paths.append(path)

        for path in paths:
            if is_valid_file(path):
                # Load image
                img = dset.folder.default_loader(path)

                # Get bounding boxes
                annotation_basename = os.path.splitext(os.path.basename(path))[0]
                annotation_dirname = next(
                        dirname for dirname in os.listdir('../input/annotation/Annotation/') if
                        dirname.startswith(annotation_basename.split('_')[0]))
                annotation_filename = os.path.join('../input/annotation/Annotation/',
                                                   annotation_dirname, annotation_basename)
                tree = ET.parse(annotation_filename)
                root = tree.getroot()
                objects = root.findall('object')
                for o in objects:
                    bndbox = o.find('bndbox')
                    xmin = int(bndbox.find('xmin').text)
                    ymin = int(bndbox.find('ymin').text)
                    xmax = int(bndbox.find('xmax').text)
                    ymax = int(bndbox.find('ymax').text)
                    
                    
                    w = np.min((xmax - xmin, ymax - ymin))
                    bbox = (xmin-5, ymin-5, xmin+w+10, ymin+w+10)
                    object_img = required_transforms(img.crop(bbox))
                    object_img = object_img.resize((64,64), Image.ANTIALIAS)
                    imgs.append(object_img)
        return imgs

    def __getitem__(self, index):
        sample = self.samples[index]
        
        if self.transform is not None:
            sample = self.transform(sample)
            
        return np.asarray(sample)
    
    def __len__(self):
        return len(self.samples)


# In[ ]:


get_ipython().run_cell_magic('time', '', "database = '../input/all-dogs/all-dogs/'\n\ntransform = transforms.Compose([transforms.RandomHorizontalFlip(p=0.1),\n                                transforms.ToTensor(),\n                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])\ntransform1 = transforms.Compose([transforms.RandomHorizontalFlip(p=0.9),\n                                transforms.ToTensor(),\n                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])\n\ntrain_data1 = DataGenerator(database, transform=transform,n_samples=20579)\n#train_data2 = DataGenerator(database, transform=transform,n_samples=20579)\n#train_data = train_data1+ train_data2\ndataloader = torch.utils.data.DataLoader(train_data1, shuffle=True,batch_size=opt.batch_size, num_workers = 4)")


# In[ ]:


x = next(iter(dataloader))
#imgs = imgs.numpy().transpose(0, 2, 3, 1)


fig = plt.figure(figsize=(25, 16))
for ii, img in enumerate(x[:32]):
    ax = fig.add_subplot(4, 8, ii + 1, xticks=[], yticks=[])
    
    img = img.numpy().transpose(1, 2, 0)
    plt.imshow((img +1.0)/2.0)
    

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ## Parameters for Parsing

# ## Configure for this problem

# In[ ]:


img_shape = (opt.channels, opt.img_size, opt.img_size)

cuda = True if torch.cuda.is_available() else False


# ## generator

# In[ ]:


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        def block(in_feat, out_feat, normalize=True):
            layers = [nn.Linear(in_feat, out_feat)]
            if normalize:
                layers.append(nn.BatchNorm1d(out_feat, 0.8))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *block(opt.latent_dim, 128, normalize=False),
            *block(128, 256),
            *block(256, 512),
            *block(512, 1024),
            nn.Linear(1024, int(np.prod(img_shape))),
            nn.Tanh()
        )

    def forward(self, z):
        img = self.model(z)
        img = img.view(img.shape[0], *img_shape)
        return img


# ## Discriminator

# In[ ]:


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(int(np.prod(img_shape)), 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1),
        )

    def forward(self, img):
        img_flat = img.view(img.shape[0], -1)
        validity = self.model(img_flat)
        return validity


# Co-efficient and power of regularization

# In[ ]:


k = 2
p = 6


# ## Initialize Generator and Discriminator

# In[ ]:


# Initialize generator and discriminator
generator = Generator()
discriminator = Discriminator()

if cuda:
    generator.cuda()
    discriminator.cuda()


# ## Configure data loader

# ## Optimizers

# In[ ]:


## This is to show the losses of the different loss functions
def show_losses():
    fig, ax = plt.subplots()
    plt.plot(d_losses, label='Discriminator', alpha=0.5)
    plt.plot(g_losses, label='Generator', alpha=0.5)
    #plt.title(title)
    plt.legend()
    #plt.savefig(filename)
    plt.show()
    plt.close()


# In[ ]:


optimizer_G = torch.optim.Adam(generator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))

Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor


# ## Training

# In[ ]:


if not os.path.exists('../output_images'):
    os.mkdir('../output_images')


# In[ ]:


batches_done = 0
g_losses = []
d_losses =[]
for epoch in range(opt.n_epochs):
    for i, imgs in tqdm(enumerate(dataloader), total=len(dataloader)):

        # Configure input
        real_imgs = Variable(imgs.type(Tensor), requires_grad=True)

        # ---------------------
        #  Train Discriminator
        # ---------------------

        optimizer_D.zero_grad()

        # Sample noise as generator input
        z = Variable(Tensor(np.random.normal(0, 1, (imgs.shape[0], opt.latent_dim))))

        # Generate a batch of images
        fake_imgs = generator(z)

        # Real images
        real_validity = discriminator(real_imgs)
        # Fake images
        fake_validity = discriminator(fake_imgs)

        # Compute W-div gradient penalty
        real_grad_out = Variable(Tensor(real_imgs.size(0), 1).fill_(1.0), requires_grad=False)
        real_grad = autograd.grad(
            real_validity, real_imgs, real_grad_out, create_graph=True, retain_graph=True, only_inputs=True
        )[0]
        real_grad_norm = real_grad.view(real_grad.size(0), -1).pow(2).sum(1) ** (p / 2)

        fake_grad_out = Variable(Tensor(fake_imgs.size(0), 1).fill_(1.0), requires_grad=False)
        fake_grad = autograd.grad(
            fake_validity, fake_imgs, fake_grad_out, create_graph=True, retain_graph=True, only_inputs=True
        )[0]
        fake_grad_norm = fake_grad.view(fake_grad.size(0), -1).pow(2).sum(1) ** (p / 2)

        div_gp = torch.mean(real_grad_norm + fake_grad_norm) * k / 2

        # Adversarial loss
        d_loss = -torch.mean(real_validity) + torch.mean(fake_validity) + div_gp
        #d_losses.append(d_loss)

        d_loss.backward()
        optimizer_D.step()

        optimizer_G.zero_grad()

        # Train the generator every n_critic steps
        if i % opt.n_critic == 0:

            # -----------------
            #  Train Generator
            # -----------------

            # Generate a batch of images
            fake_imgs = generator(z)
            # Loss measures generator's ability to fool the discriminator
            # Train on fake images
            fake_validity = discriminator(fake_imgs)
            g_loss = -torch.mean(fake_validity)
            #g_losses.append(g_loss)

            g_loss.backward()
            optimizer_G.step()
            
            print(
                "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]"
                % (epoch, opt.n_epochs, i, len(dataloader), d_loss.item(), g_loss.item())
            )

            if batches_done % opt.sample_interval == 0:
                g_losses.append(g_loss)
                d_losses.append(d_loss)
                save_image(fake_imgs.data[:25], "../output_images/%d.png" % batches_done, nrow=5, normalize=True)

            batches_done += opt.n_critic


# ## Show the images generated ?

# In[ ]:


def plot_fake_images(z):
    #gen_z = torch.randn(32, opt.latent_dim, 1, 1)
    gen_images = generator(z)
    images = gen_images.to("cpu").clone().detach()
    images = images.numpy().transpose(0, 2, 3, 1)
    images = (images +1.0)/2.0
    fig = plt.figure(figsize=(25, 16))
    for ii, img in enumerate(images[:32]):
        ax = fig.add_subplot(4, 8, ii+1 , xticks=[], yticks=[])
        plt.imshow(img)


# In[ ]:


show_losses()


# In[ ]:


ims_animation = []
if not os.path.exists('../output_images'):
    os.mkdir('../output_images')
im_batch_size = 64
n_images=200
for i_batch in range(0, n_images, im_batch_size):
    z = Variable(torch.cuda.FloatTensor((np.random.normal(0, 1, (im_batch_size, opt.latent_dim)))))
    #z = torch.randn(im_batch_size, latent_dim, device=device)
    plot_fake_images(z)
    gen_images = generator(z)
    images = gen_images.to("cpu").clone().detach()
    images = images.numpy().transpose(0, 2, 3, 1)
    ims_animation.append(images)
    for i_image in range(gen_images.size(0)):
        save_image((gen_images[i_image, :, :, :]+1.0)/2.0, os.path.join('../output_images', f'image_{i_batch+i_image:05d}.png'))


# In[ ]:





# In[ ]:


import matplotlib.animation as animation
#from matplotlib.animation import FuncAnimation

fig = plt.figure() 

ims = []
#fig, ax = plt.subplots()
#xdata, ydata = [], []
#ln, = plt.plot([], [], 'ro',animated=True)
for j in range(len(ims_animation)):
    im = plt.imshow(ims_animation[j][0],animated=True)
    ims.append([im])
    
anim  = animation.ArtistAnimation(fig, ims, interval=5, blit=True, repeat_delay=10,repeat = True)

anim.save('generate_dog.gif',writer='ffmpeg')

