#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# install pytorch lightning if unable to import
try:
    import pytorch_lightning as pl
except:
    get_ipython().system('pip install pytorch-lightning')


# In[ ]:


# import libraries
from os import makedirs
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, utils

import pytorch_lightning as pl

import warnings
warnings.filterwarnings('ignore')

path = '/kaggle/working/output'

# create output folder if does'nt exist
makedirs(path, exist_ok=True)

# shape of the image (channel, height, width)
img_shape = (1, 28, 28)


# In[ ]:


# Generator Model
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.fc1 = nn.Linear(100, 128)
        self.fc2 = nn.Linear(128, 512)
        self.fc3 = nn.Linear(512, 1024)
        self.fc4 = nn.Linear(1024, 28*28)
        self.bn1 = nn.BatchNorm1d(128)
        self.bn2 = nn.BatchNorm1d(512)
        self.bn3 = nn.BatchNorm1d(1024)

    def forward(self, x):
        x = self.fc1(x)
        x = self.bn1(x)
        x = F.leaky_relu(x, 0.2)
        # mixing above 3 steps in single line
        x = F.leaky_relu(self.bn2(self.fc2(x)), 0.2)
        x = F.leaky_relu(self.bn3(self.fc3(x)), 0.2)
        x = torch.tanh(self.fc4(x))
        return x.view(x.shape[0], *img_shape)


# Discriminator Model
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.fc1 = nn.Linear(28*28, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, 1)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = F.leaky_relu(self.fc1(x), 0.2)
        x = F.leaky_relu(self.fc2(x), 0.2)
        x = F.leaky_relu(self.fc3(x), 0.2)
        x = torch.sigmoid(self.fc4(x))
        return x


# In[ ]:


# PyTorch Lightning Class (where all the magic(automation) will happen)
class GAN(pl.LightningModule):
    
    # Model Initialization/Creation
    def __init__(self, hparams):
        super(GAN, self).__init__()

        self.hparams = hparams
        self.generator = Generator()
        self.discriminator = Discriminator()
    
    # Forward Pass of Model
    def forward(self, x):
        return self.discriminator(x)
    
    # Loss Function
    def loss_function(self, y_hat, y):
        return nn.BCELoss()(y_hat, y)
    
    # Optimizers
    def configure_optimizers(self):
        optimizer_G = torch.optim.Adam(self.generator.parameters(), lr=self.hparams.lr, betas=(0.4, 0.999))
        optimizer_D = torch.optim.Adam(self.discriminator.parameters(), lr=self.hparams.lr, betas=(0.4, 0.999))
        
        # return the list of optimizers and second empty list is for schedulers (if any)
        return [optimizer_G, optimizer_D], []

    # Data preparation (Download/Preprocessing)
    def prepare_data(self):
        transform = transforms.Compose([transforms.ToTensor(),
                                        transforms.Normalize([0.5], [0.5])])
        
        train_data = datasets.MNIST('data/', train=True, download=True, transform=transform)
        return train_data

    # Calls after prepare_data for DataLoader
    def train_dataloader(self):
        train_loader = DataLoader(self.prepare_data(), batch_size=self.hparams.batch_size, shuffle=True)
        return train_loader
    
    # Training Loop
    def training_step(self, batch, batch_idx, optimizer_idx):
        # batch returns x and y tensors
        real_images, _ = batch
        
        # ground truth (tensors of ones and zeros) same shape as images
        valid = torch.ones(real_images.size(0), 1)
        fake = torch.zeros(real_images.size(0), 1)
        
        # svaing loss_function as local variable
        criterion = self.loss_function
        
        # As there are 2 optimizers we have to train for both using 'optimizer_idx'
        ## Generator
        if optimizer_idx == 0:
            # Generating Noise (input for the generator)
            gen_input = torch.randn(real_images.shape[0], 100)
            
            # Converting noise to images
            self.gen_images = self.generator(gen_input)
            
            # Calculating generator loss
            # How well the generator can create real images
            g_loss = criterion(self(self.gen_images), valid)
            
            # for output and logging purposes (return as dictionaries)
            tqdm_dict = {'g_loss': g_loss}
            output = OrderedDict({
                'loss': g_loss,
                'progress_bar': tqdm_dict,
                'log': tqdm_dict,
                'g_loss': g_loss
            })
            return output

        ## Discriminator
        if optimizer_idx == 1:
            # Calculating disciminator loss
            # How well discriminator identifies the real and fake images
            real_loss = criterion(self(real_images), valid)
            fake_loss = criterion(self(self.gen_images.detach()), fake)
            d_loss = (real_loss + fake_loss)/2.0
            
            # for output and logging purposes (return as dictionaries)
            tqdm_dict = {'d_loss': d_loss}
            output = OrderedDict({
                'loss': d_loss,
                'progress_bar': tqdm_dict,
                'log': tqdm_dict,
                'd_loss': d_loss
            })
            return output
    
    # calls after every epoch ends
    def on_epoch_end(self):
        # Saving 5x5 grid
        utils.save_image(self.gen_images.data[:25], path+'/%d.png' % self.current_epoch, nrow=5, padding=0, normalize=True)


# In[ ]:


import argparse

# Helper Function to replicate command line inputs
def dict_to_args(d):

    args = argparse.Namespace()

    def dict_to_args_recursive(args, d, prefix=''):
        for k, v in d.items():
            if type(v) == dict:
                dict_to_args_recursive(args, v, prefix=k)
            elif type(v) in [tuple, list]:
                continue
            else:
                if prefix:
                    args.__setattr__(prefix + '_' + k, v)
                else:
                    args.__setattr__(k, v)

    dict_to_args_recursive(args, d)
    
    return args


# In[ ]:


# Hyperparameters
hparams = dict_to_args({'batch_size': 32,
                        'lr': 2e-4,
                        'epochs': 20
                      })

# Model Initialization with hyperparameters
gan = GAN(hparams=hparams)

# PyTorch Lightning Trainer (where loss backward, optimizer grading, gpu/tpu code automates)
trainer = pl.Trainer(max_epochs=hparams.epochs, fast_dev_run=False)

# Fitting the model to trainer
trainer.fit(gan)

