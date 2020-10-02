#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
import PIL
import torchvision
import torchvision.datasets as dset
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch.autograd import Variable
import xml.etree.ElementTree as ET
import numpy as np

import matplotlib.pyplot as plt


import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
from torch.nn.init import xavier_uniform_


import time
import torch
import torch.nn as nn

import torch.nn.parallel
import torch.optim as optim
from torch.nn.utils import spectral_norm
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

import torch.nn.functional as F
from torch.nn import Parameter


import numpy as np
import os
import gzip, pickle
import tensorflow as tf
from scipy import linalg
import pathlib
import urllib
import warnings
from tqdm import tqdm
from PIL import Image
import zipfile

from tqdm import tqdm_notebook as tqdm


# In[ ]:


kernel_start_time = time.perf_counter()


# In[ ]:


class PixelwiseNorm(nn.Module):
    def __init__(self):
        super(PixelwiseNorm, self).__init__()

    def forward(self, x, alpha=1e-8):
        """
        forward pass of the module
        :param x: input activations volume
        :param alpha: small number for numerical stability
        :return: y => pixel normalized activations
        """
        y = x.pow(2.).mean(dim=1, keepdim=True).add(alpha).sqrt()  # [N1HW]
        y = x / y  # normalize the input x volume
        return y
    
class MinibatchStdDev(nn.Module):
    def __init__(self):
        super(MinibatchStdDev, self).__init__()

    def forward(self, x, alpha=1e-8):
        """
        forward pass of the layer
        :param x: input activation volume
        :param alpha: small number for numerical stability
        :return: y => x appended with standard deviation constant map
        """
        batch_size, _, height, width = x.shape
        # [B x C x H x W] Subtract mean over batch.
        y = x - x.mean(dim=0, keepdim=True)
        # [1 x C x H x W]  Calc standard deviation over batch
        y = torch.sqrt(y.pow(2.0).mean(dim=0, keepdim=False) + alpha)

        # [1]  Take average over feature_maps and pixels.
        y = y.mean().view(1, 1, 1, 1)

        # [B x 1 x H x W]  Replicate over group and pixels.
        y = y.repeat(batch_size, 1, height, width)

        # [B x C x H x W]  Append as new feature_map.
        y = torch.cat([x, y], 1)
        # return the computed values:
        return y


# ## GAN
# ### G
# convlution transpose + spectral norm + pixel wise norm
# ### D
# convlution + spectral norm + batch norm + pixel wise norm

# In[ ]:


class Generator(nn.Module):
    def __init__(self, nz=128, num_classes=120, channels=3, nfeats=32):
        super(Generator, self).__init__()
        self.nz = nz
        self.num_classes = num_classes
        self.channels = channels
        
        self.label_emb = nn.Embedding(num_classes, nz)
        self.pixnorm = PixelwiseNorm()
        self.conv1 = spectral_norm(nn.ConvTranspose2d(2*nz, nfeats * 8, 4, 1, 0, bias=False))
        self.conv2 = spectral_norm(nn.ConvTranspose2d(nfeats * 8, nfeats * 8, 4, 2, 1, bias=False))
        self.conv3 = spectral_norm(nn.ConvTranspose2d(nfeats * 8, nfeats * 4, 4, 2, 1, bias=False))
        self.conv4 = spectral_norm(nn.ConvTranspose2d(nfeats * 4, nfeats * 2, 4, 2, 1, bias=False))
        self.conv5 = spectral_norm(nn.ConvTranspose2d(nfeats * 2, nfeats, 4, 2, 1, bias=False))
        self.conv6 = spectral_norm(nn.ConvTranspose2d(nfeats, channels, 3, 1, 1, bias=False))

    def forward(self, inputs):
        z, labels = inputs
        enc = self.label_emb(labels).view((-1, self.nz, 1, 1))
        enc = F.normalize(enc, p=2, dim=1)
        x = torch.cat((z, enc), 1)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.pixnorm(x)
        x = F.relu(self.conv3(x))
        x = self.pixnorm(x)
        x = F.relu(self.conv4(x))
        x = self.pixnorm(x)
        x = F.relu(self.conv5(x))
        x = self.pixnorm(x)
        x = torch.tanh(self.conv6(x))
        return x

    
class Discriminator(nn.Module):
    def __init__(self, num_classes=120, channels=3, nfeats=64):
        super(Discriminator, self).__init__()
        self.channels = channels
        self.num_classes = num_classes
        self.label_emb = nn.Embedding(num_classes, 64*64)
        self.conv1 = nn.Conv2d(channels+1, nfeats, 5, 2, 2, bias=False)
        self.conv2 = spectral_norm(nn.Conv2d(nfeats, nfeats * 2, 5, 2, 2, bias=False))
        self.bn2 = nn.BatchNorm2d(nfeats * 2)
        self.conv3 = spectral_norm(nn.Conv2d(nfeats * 2, nfeats * 4, 5, 2, 2, bias=False))
        self.bn3 = nn.BatchNorm2d(nfeats * 4)
        self.conv4 = spectral_norm(nn.Conv2d(nfeats * 4, nfeats * 8, 5, 2, 2, bias=False))
        self.bn4 = nn.MaxPool2d(2)
        self.batch_discriminator = MinibatchStdDev()
        self.pixnorm = PixelwiseNorm()
        self.conv5 = spectral_norm(nn.Conv2d(nfeats * 8 +1, 1, 2, 1, 0, bias=False))

    def forward(self, inputs):
        imgs, labels = inputs
        enc = self.label_emb(labels).view((-1, 1, 64, 64))
        enc = F.normalize(enc, p=2, dim=1)
        x = torch.cat((imgs, enc), 1)   # 4 input feature maps(3rgb + 1label)
        x = F.relu(self.conv1(x), 0.2)
        x = F.relu(self.bn2(self.conv2(x)), 0.2)
        x = F.relu(self.bn3(self.conv3(x)), 0.2)
        x = F.relu(self.bn4(self.conv4(x)), 0.2)
        x = self.batch_discriminator(x)
        x = torch.sigmoid(self.conv5(x))
        return x.view(-1, 1)


# # Data Loader

# In[ ]:


class DataGenerator(Dataset):
    def __init__(self, directory, transform=None, n_samples=np.inf, crop_dogs=True):
        self.directory = directory
        self.transform = transform
        self.n_samples = n_samples        
        self.samples, self.labels = self.load_dogs_data(directory, crop_dogs)

    def load_dogs_data(self, directory, crop_dogs):
        required_transforms = torchvision.transforms.Compose([
                torchvision.transforms.Resize(64),
                torchvision.transforms.CenterCrop(64),
        ])

        imgs = []
        labels = []
        paths = []
        for root, _, fnames in sorted(os.walk(directory)):
            for fname in sorted(fnames)[:min(self.n_samples, 999999999999999)]:
                path = os.path.join(root, fname)
                paths.append(path)

        for path in paths:
            # Load image
            try: img = dset.folder.default_loader(path)
            except: continue
            
            # Get bounding boxes
            annotation_basename = os.path.splitext(os.path.basename(path))[0]
            annotation_dirname = next(
                    dirname for dirname in os.listdir('../input/annotation/Annotation/') if
                    dirname.startswith(annotation_basename.split('_')[0]))
                
            if crop_dogs:
                tree = ET.parse(os.path.join('../input/annotation/Annotation/',
                                             annotation_dirname, annotation_basename))
                root = tree.getroot()
                objects = root.findall('object')
                for o in objects:
                    bndbox = o.find('bndbox')
                    xmin = int(bndbox.find('xmin').text)
                    ymin = int(bndbox.find('ymin').text)
                    xmax = int(bndbox.find('xmax').text)
                    ymax = int(bndbox.find('ymax').text)
                    object_img = required_transforms(img.crop((xmin, ymin, xmax, ymax)))
                    imgs.append(object_img)
                    labels.append(annotation_dirname.split('-')[1].lower())

            else:
                object_img = required_transforms(img)
                imgs.append(object_img)
                labels.append(annotation_dirname.split('-')[1].lower())
            
        return imgs, labels
    
    
    def __getitem__(self, index):
        sample = self.samples[index]
        label = self.labels[index]
        
        if self.transform is not None: 
            sample = self.transform(sample)
        return np.asarray(sample), label

    
    def __len__(self):
        return len(self.samples)


# ## Load data

# In[ ]:


database = '../input/all-dogs/all-dogs/'
crop_dogs = True
n_samples = np.inf
BATCH_SIZE = 32

epochs = 1600
criterion = nn.BCELoss()

# use_soft_noisy_labels=True

nz = 128
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


transform = transforms.Compose([transforms.RandomHorizontalFlip(p=0.5),
                                transforms.ToTensor(),
                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

train_data = DataGenerator(database, transform=transform, n_samples=n_samples, crop_dogs=crop_dogs)

decoded_dog_labels = {i:breed for i, breed in enumerate(sorted(set(train_data.labels)))}
encoded_dog_labels = {breed:i for i, breed in enumerate(sorted(set(train_data.labels)))}
train_data.labels = [encoded_dog_labels[l] for l in train_data.labels] # encode dog labels in the data generator


train_loader = torch.utils.data.DataLoader(train_data, shuffle=True,
                                           batch_size=BATCH_SIZE, num_workers=4)


print("Dog breeds loaded:  ", len(encoded_dog_labels))
print("Data samples loaded:", len(train_data))


# # Hyperparm

# In[ ]:


netG = Generator(nz, num_classes=len(encoded_dog_labels), nfeats=32).to(device)
netD = Discriminator(num_classes=len(encoded_dog_labels), nfeats=32).to(device)
print("Generator parameters:    ", sum(p.numel() for p in netG.parameters() if p.requires_grad))
print("Discriminator parameters:", sum(p.numel() for p in netD.parameters() if p.requires_grad))

optimizerG = optim.Adam(netG.parameters(), lr=0.0002, betas=(0.5, 0.999))
optimizerD = optim.Adam(netD.parameters(), lr=0.0002, betas=(0.5, 0.999))

lr_schedulerG = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizerG, T_0=epochs//200, eta_min=0.00005)
lr_schedulerD = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizerD, T_0=epochs//200, eta_min=0.00005)


# # Some show function

# In[ ]:


### This is to show one sample image for iteration of chosing
def show_generated_img():
    noise = torch.randn(1, nz, 1, 1, device=device)
    dog_label = torch.randint(0, len(encoded_dog_labels), (1, ), device=device)
    gen_image = netG((noise, dog_label)).to("cpu").clone().detach().squeeze(0)
    #gen_image = netG(noise).to("cpu").clone().detach().squeeze(0)
    gen_image = gen_image.numpy().transpose(1, 2, 0)
    gen_image = ((gen_image+1.0)/2.0)
    plt.imshow(gen_image)
    plt.show()


# ## GAN training (SGAN)
# 1. Basic SGAN training
# 2. 1600 epochs
# 3. BCEloss
# 4. True label is 0.7 + random(-0.1~0.1) and Fake label is 0.0 + random(0.0~0.2)

# In[ ]:


for epoch in range(epochs):
    epoch_time = time.perf_counter()
    if time.perf_counter() - kernel_start_time > 30000:
            print("Time limit reached! Stopping kernel!"); break

    for ii, (real_images, dog_labels) in tqdm(enumerate(train_loader),total=len(train_loader)):
        if real_images.shape[0]!= BATCH_SIZE: continue
            
        # smooth label
        real_labels = torch.squeeze(torch.empty((BATCH_SIZE, 1), device=device).uniform_(0.60, 0.8))
        fake_labels = torch.squeeze(torch.empty((BATCH_SIZE, 1), device=device).uniform_(0.00, 0.2))
#         for p in np.random.choice(BATCH_SIZE, size=np.random.randint((BATCH_SIZE//8)), replace=False):
#             real_labels[p], fake_labels[p] = fake_labels[p], real_labels[p] # swap labels
        
        
        ############################
        # (1) Update D network
        ###########################
        # Update real images to D
        netD.zero_grad()
        dog_labels = torch.tensor(dog_labels, device=device)
        real_images = real_images.to(device)
        output = netD((real_images, dog_labels))
        errD_real = criterion(output, real_labels)
        errD_real.backward()
        D_x = output.mean().item()
        
        # Update fake images to D
        noise = torch.randn(BATCH_SIZE, nz, 1, 1, device=device)
        fake_images = netG((noise, dog_labels))
        output = netD((fake_images.detach(), dog_labels)) 
        errD_fake = criterion(output, fake_labels)
        errD_fake.backward()
        D_G_z1 = output.mean().item()
        errD = errD_real + errD_fake
        optimizerD.step()
        
        ############################
        # (2) Update G network
        ###########################
        netG.zero_grad()
        output = netD((fake_images, dog_labels))
        errG = criterion(output, real_labels)
        errG.backward()
        D_G_z2 = output.mean().item()
        optimizerG.step()
        
        lr_schedulerG.step(epoch)
        lr_schedulerD.step(epoch)

    
    print('%.2fs [%d/%d] Loss_D: %.4f Loss_G: %.4f' % (
          time.perf_counter()-epoch_time, epoch+1, epochs, errD.item(), errG.item()))
    print('D(x): ',D_x,', D(G(z)): ',D_G_z1)
    show_generated_img()
    


# ## Submit the best breed

# In[ ]:


def mse(imageA, imageB):
        err = np.sum((imageA.astype("float") - imageB.astype("float")) ** 2)
        err /= float(imageA.shape[0] * imageA.shape[1])
        return err

def analyse_generated_by_class(n_images=5):
    good_breeds = []
    for l in range(len(decoded_dog_labels)):
        sample = []
        for _ in range(n_images):
            noise = torch.randn(1, nz, 1, 1, device=device)
            dog_label = torch.full((1,) , l, device=device, dtype=torch.long)
            gen_image = netG((noise, dog_label)).to("cpu").clone().detach().squeeze(0)
            gen_image = gen_image.numpy().transpose(1, 2, 0)
            sample.append(gen_image)
        
        # if mse for sample k to k-1 is too different , discard the sample
        d = np.round(np.sum([mse(sample[k], sample[k+1]) for k in range(len(sample)-1)])/n_images, 1)
        
        if d < 1.0: continue  # had mode colapse(discard)
            
        print(f"Generated breed({d}): ", decoded_dog_labels[l])
        figure, axes = plt.subplots(1, len(sample), figsize=(64, 64))
        for index, axis in enumerate(axes):
            axis.axis('off')
            image_array = (sample[index] + 1.) / 2.
            axis.imshow(image_array)
        plt.show()
        
        good_breeds.append(l)
    return good_breeds


# In[ ]:


def create_submit(good_breeds):
    print("Creating submit")
    os.makedirs('../output_images', exist_ok=True)
    im_batch_size = 100
    n_images = 10000
    
    all_dog_labels = np.random.choice(good_breeds, size=n_images, replace=True)
    for i_batch in range(0, n_images, im_batch_size):
        noise = torch.randn(im_batch_size, nz, 1, 1, device=device)
        dog_labels = torch.from_numpy(all_dog_labels[i_batch: (i_batch+im_batch_size)]).to(device)
        gen_images = netG((noise, dog_labels))
        gen_images = (gen_images.to("cpu").clone().detach() + 1) / 2
        for ii, img in enumerate(gen_images):
            save_image(gen_images[ii, :, :, :], os.path.join('../output_images', f'image_{i_batch + ii:05d}.png'))
            
    import shutil
    shutil.make_archive('images', 'zip', '../output_images')


# In[ ]:


good_breeds = analyse_generated_by_class(6)
create_submit(good_breeds)


# In[ ]:




