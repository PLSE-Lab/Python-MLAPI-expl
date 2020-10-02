#!/usr/bin/env python
# coding: utf-8

# As far as we know, I/O operations are most expensive in training Neural Networks.<br> In this kernel I wrote pytorch Dataset loader that loads and preprocess all images once and stores them into RAM.

# In[ ]:


import numpy as np
import pandas as pd
import os
from os.path import join
import matplotlib.pyplot as plt

import torch
from torch import nn, optim
import torch.nn.functional as F
import torchvision
from torchvision import datasets, transforms
from torchvision.utils import save_image
from torch.utils.data import Dataset, DataLoader

from PIL import Image
import xml.etree.ElementTree as ET

from tqdm import tqdm_notebook as tqdm


# ## Parameters of GAN

# In[ ]:


batch_size = 16
lrG = 0.001
lrD = 0.001
beta1 = 0.5
epochs = 300

real_label = 0.5
fake_label = 0
nz = 128

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ## Pytorch Dataset and DataLoader

# In[ ]:


class DogDataset(Dataset):
    def __init__(self, img_dir, annotations_dir, transform1=None, transform2=None):

        self.img_dir = img_dir
        self.img_names = os.listdir(img_dir)
        self.transform1 = transform1
        self.transform2 = transform2
        
        self.imgs = []
        for img_name in self.img_names:
            path = join(img_dir, img_name)
            img = torchvision.datasets.folder.default_loader(path)
    
            # Crop image
            annotation_basename = os.path.splitext(os.path.basename(path))[0]
            annotation_dirname = next(dirname for dirname in os.listdir(annotations_dir) if dirname.startswith(annotation_basename.split('_')[0]))
            annotation_filename = os.path.join(annotations_dir, annotation_dirname, annotation_basename)
            tree = ET.parse(annotation_filename)
            root = tree.getroot()
            objects = root.findall('object')
            
            for o in objects:
                bndbox = o.find('bndbox')
                xmin = int(bndbox.find('xmin').text)
                ymin = int(bndbox.find('ymin').text)
                xmax = int(bndbox.find('xmax').text)
                ymax = int(bndbox.find('ymax').text)
                bbox = (xmin, ymin, xmax, ymax)
                img_ = img.crop(bbox)
                # Some crop's are black. if crop is black then don't crop
                if np.mean(img_) != 0:
                    img = img_

                if self.transform1 is not None:
                    img = self.transform1(img)

                self.imgs.append(img)

    def __getitem__(self, index):
        img = self.imgs[index]
        
        if self.transform2 is not None:
            img = self.transform2(img)
        
        return img

    def __len__(self):
        return len(self.imgs)


# # Benchmark

# ## Simple Dataloader

# In[ ]:


get_ipython().run_cell_magic('time', '', "random_transforms = [transforms.RandomRotation(degrees=5)]\ntransform = transforms.Compose([transforms.Resize(64),\n                                transforms.CenterCrop(64),\n                                transforms.RandomHorizontalFlip(p=0.5),\n                                transforms.RandomApply(random_transforms, p=0.2),\n                                transforms.ToTensor(),\n                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])\n\ntrain_data = datasets.ImageFolder('../input/all-dogs/', transform=transform)\ntrain_loader = torch.utils.data.DataLoader(train_data, shuffle=True,\n                                           batch_size=batch_size)")


# In[ ]:


get_ipython().run_cell_magic('time', '', 'for _ in train_loader:\n    continue')


# ## Simple Dataloader with num_workers=2

# In[ ]:


get_ipython().run_cell_magic('time', '', "random_transforms = [transforms.RandomRotation(degrees=5)]\ntransform = transforms.Compose([transforms.Resize(64),\n                                transforms.CenterCrop(64),\n                                transforms.RandomHorizontalFlip(p=0.5),\n                                transforms.RandomApply(random_transforms, p=0.2),\n                                transforms.ToTensor(),\n                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])\n\ntrain_data = datasets.ImageFolder('../input/all-dogs/', transform=transform)\ntrain_loader = torch.utils.data.DataLoader(train_data, shuffle=True,\n                                           batch_size=batch_size,\n                                          num_workers=2)")


# In[ ]:


get_ipython().run_cell_magic('time', '', 'for _ in train_loader:\n    continue')


# ## RAM Dataloader
# First step is **once** download data

# In[ ]:


get_ipython().run_cell_magic('time', '', "# First preprocessing of data\ntransform1 = transforms.Compose([transforms.Resize(64),\n                                transforms.CenterCrop(64)])\n\n# Data augmentation and converting to tensors\nrandom_transforms = [transforms.RandomRotation(degrees=5)]\ntransform2 = transforms.Compose([transforms.RandomHorizontalFlip(p=0.5),\n                                 transforms.RandomApply(random_transforms, p=0.3), \n                                 transforms.ToTensor(),\n                                 transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])\n                                 \ntrain_dataset = DogDataset(img_dir='../input/all-dogs/all-dogs/',\n                           annotations_dir='../input/annotation/Annotation/',\n                           transform1=transform1,\n                           transform2=transform2)\n\ntrain_loader = DataLoader(dataset=train_dataset,\n                          batch_size=batch_size,\n                          shuffle=True,\n                          num_workers=4)")


# Second step: iterate over downloaded images

# In[ ]:


get_ipython().run_cell_magic('time', '', 'for _ in train_loader:\n    continue')


# Let's calculate how much time you need to train 100 epochs your GAN (forward and backward pass of NNs doesn't counted)
#  - Simple Dataloader: ~90sec per epoch * 100 epochs = 9000sec = 2.5h
#  - Simple Dataloader with 2 workers: ~ 65sec per epoch * 100 epochs = 1h 50min
#  - RAM Dataloader: ~95sec download + 4.5sec per epoch * 100 epochs = 9 min<br>

# With this dataloader you can make much more experiments and epochs!

# ## Examples of dogs

# In[ ]:


x = next(iter(train_loader))

fig = plt.figure(figsize=(25, 16))
for ii, img in enumerate(x):
    ax = fig.add_subplot(4, 8, ii + 1, xticks=[], yticks=[])
    
    img = img.numpy().transpose(1, 2, 0)
    plt.imshow((img+1)/2)


# ### Model and training loop taken from https://www.kaggle.com/speedwagon/ralsgan-dogs

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
                nn.LeakyReLU(inplace=True),
            ]
            return block

        self.model = nn.Sequential(
            *convlayer(self.nz, 512, 4, 1, 0), # Fully connected layer via convolution.
            *convlayer(512, 256, 4, 2, 1),
            *convlayer(256, 128, 4, 2, 1),
            *convlayer(128, 64, 4, 2, 1),
            *convlayer(64, 32, 4, 2, 1),
            nn.ConvTranspose2d(32, self.channels, 3, 1, 1),
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
            nn.Conv2d(256, 1, 4, 1, 0, bias=False),  # FC with Conv.
        )

    def forward(self, imgs):
        out = self.model(imgs)
        return out.view(-1, 1)


# In[ ]:


netG = Generator(nz).to(device)
netD = Discriminator().to(device)

criterion = nn.BCELoss()

optimizerD = optim.Adam(netD.parameters(), lr=lrD, betas=(beta1, 0.999))
optimizerG = optim.Adam(netG.parameters(), lr=lrG, betas=(beta1, 0.999))


# In[ ]:


def show_generated_img():
    noise = torch.randn(1, nz, 1, 1, device=device)
    gen_image = (netG(noise).to("cpu").clone().detach().squeeze(0) + 1) / 2
    gen_image = gen_image.numpy().transpose(1, 2, 0)
    plt.imshow(gen_image)
    plt.show()


# In[ ]:


for epoch in range(epochs):
    
    for ii, real_images in tqdm(enumerate(train_loader), total=len(train_loader)):
        ############################
        # (1) Update D network
        ###########################
        netD.zero_grad()
        real_images = real_images.to(device)
        batch_size = real_images.size(0)
        labels = torch.full((batch_size, 1), real_label, device=device)
        outputR = netD(real_images)
        noise = torch.randn(batch_size, nz, 1, 1, device=device)
        fake = netG(noise)
        outputF = netD(fake.detach())
        errD = (torch.mean((outputR - torch.mean(outputF) - labels) ** 2) + 
                torch.mean((outputF - torch.mean(outputR) + labels) ** 2))/2
        errD.backward(retain_graph=True)
        optimizerD.step()
        ############################
        # (2) Update G network
        ###########################
        netG.zero_grad()
        outputF = netD(fake)   
        errG = (torch.mean((outputR - torch.mean(outputF) + labels) ** 2) +
                torch.mean((outputF - torch.mean(outputR) - labels) ** 2))/2
        errG.backward()
        optimizerG.step()
        
        if (ii+1) % (len(train_loader)//2) == 0:
            print('[%d/%d][%d/%d] Loss_D: %.4f Loss_G: %.4f'
                  % (epoch + 1, epochs, ii+1, len(train_loader),
                     errD.item(), errG.item()))

    show_generated_img()


# ## Let's visualise generated results

# In[ ]:


gen_z = torch.randn(32, nz, 1, 1, device=device)
gen_images = (netG(gen_z).to("cpu").clone().detach() + 1)/2
gen_images = gen_images.numpy().transpose(0, 2, 3, 1)

fig = plt.figure(figsize=(25, 16))
for ii, img in enumerate(gen_images):
    ax = fig.add_subplot(4, 8, ii + 1, xticks=[], yticks=[])
    plt.imshow(img)


# ## Make predictions and submit

# In[ ]:


if not os.path.exists('../output_images'):
    os.mkdir('../output_images')
im_batch_size = 50
n_images=10000
for i_batch in range(0, n_images, im_batch_size):
    gen_z = torch.randn(im_batch_size, nz, 1, 1, device=device)
    gen_images = (netG(gen_z)+1)/2
    images = gen_images.to("cpu").clone().detach()
    images = images.numpy().transpose(0, 2, 3, 1)
    for i_image in range(gen_images.size(0)):
        save_image(gen_images[i_image, :, :, :], os.path.join('../output_images', f'image_{i_batch+i_image:05d}.png'))


import shutil
shutil.make_archive('images', 'zip', '../output_images')

