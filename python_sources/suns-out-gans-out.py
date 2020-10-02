#!/usr/bin/env python
# coding: utf-8

#  <h1><center><font size="30">Suns out GANs out</font></center></h1>
# ![](https://www.dailydot.com/wp-content/uploads/2020/02/elon-bezos-star-trek-deepfake.jpg.webp)

# # Introduction to GANs
#    * Generative Adversarial Networks
#    * How GANs work
#    * Process of training
#    * Starting literature

# ## Generative Adversarial Networks
# GANs are type of unsupervised machine learning method where they try to generate new, synthetic instances of data that mimics the real data. They are extremely popular in image, video (Deepfake) and voice generation. GANs were invented by Ian Goodfellow and his colleagues in 2014. and quickly rose to fame. They are constructed of two neural networks: Generator and Discriminator.

# ## How GANs work
# Let's take an example dataset and explain the theory of GANs on that specific dataset. For our example we will use dataset <b>MNIST</b> (dataset of handwritten digits). Neural network Generator generates new instances of handwritten digits and Neural network Discriminator tries to evaluate if those images are real or fake. We can look at those two NNs as enemies trying to win against each other (that's where the name adversarial comes from).The goal of Generator is to produce the best possible hand-written digits that Discriminator will classify as real and the goal of Discriminator is identify those synthetic images coming from a Generator as fake. What that means is that we will have two losses (Generator loss and Discriminator loss). It is interesting how good generator sometimes performs and in some instances can also trick a human eye. Example (generating new faces):
# 
# <img src="https://miro.medium.com/max/800/1*mdoXOnJmAgvMzfs7W9fnmA.jpeg" alt="Drawing" style="width: 400px;"/>

# In short:
#    * <b>Generator</b> - produces new images as best as it cans
#    * <b>Discriminator</b> - Classifies if images are real or fake

#  ## Process of training
#    *  <b>Generator</b> takes an input (in most cases random noise) and returns an image
#    *  Generators output (image) is handed to discriminator together with some instances of the real dataset
#    *  <b>Discriminator</b> takes images from both, Generator and real dataset and returns probabilites (0 being fake, 1 being authentic)
#    
#    
#    <img src="https://pathmind.com/images/wiki/GANs.png" alt="Drawing" style="width: 600px;"/>
#    
#    
#    <b>But how exactly do Generator and Discriminator learn?</b> Here we come to the very important part. Both, Discriminator and Generator are trained separtly. What this means is that when we train one neural network, other stays constant.
#    
#    <b>DISCRIMINATOR TRAINING</b>
#    
#    * Discriminator classifies data (both real and fake from the generator)
#    * Discriminator loss penalizes the Discriminator (for classifying authentic images as generated or generated as authentic)
#    * Discriminator updates its weights with regards to its loss (backpropagation)
#    
#    
#    <b>GENERATOR TRAINING</b>
#    
#    * Discriminator classifies data
#    * Generator loss penalizes the Generator (for classifying synthetic images as synthetic)
#    * We backpropagate through both Discriminant and Generator but update only Generator weights
#    
# <b>Let's make a summary.</b> GANs consist of two neural networks: Generator and Discriminator. Generator produces new images from random noise and Discriminator is a classifier that tries to classify those produced images as false. We train those NNs separately through backpropagation with regards to their loss. When we train one neural network the other stays constant. Generator learns to generate new images more realistically only from loss that has been given by the Discriminator when it classified it as fake. We play this game of cat and mouse until we come to the point where Discriminator can't tell if the image is real or fake.

# ## Starting literature
# * [Generative Adversarial Networks, paper by Ian Goodfellow](http://papers.nips.cc/paper/5423-generative-adversarial-nets.pdf)
# * [Generative Adversarial Networks, Google's guide](https://developers.google.com/machine-learning/gan)
# * [A Beginner's Guide to Generative Adversarial Networks (GANs)](https://pathmind.com/wiki/generative-adversarial-network-gan)
# 

# # Data
# I have chosen CIFAR-10 dataset, which is a dataset consisting of 10 different classes. Each class has 5000 images in training set and 1000 in test set. We will use DCGANs, which is just one version of GANs. You can read more [<b>here</b>](https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html).
# 

# In[ ]:


from __future__ import print_function
#%matplotlib inline
import argparse
import os
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from IPython.display import HTML


# In[ ]:


transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True,transform=transform)

#choosing only one dataset (cars)
idx = []
for i,k in enumerate(trainset.targets):
    if k==1:
        idx.append(i)

trainset.targets=list( trainset.targets[i] for i in idx )
trainset.data = trainset.data[idx]

trainloader = torch.utils.data.DataLoader(trainset, batch_size=32,
                                          shuffle=True)


imgs,label = next(iter(trainloader))

#decide which device you'll be using
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

plt.figure(figsize=(12,12))
plt.axis("off")
plt.title("Training Images")
plt.imshow(np.transpose(vutils.make_grid(imgs.to(device)[:32], padding=2, normalize=True).cpu(),(1,2,0)))


# # Implementation

# ## Weight Initialization
# 

# In[ ]:


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


# ## Generator

# In[ ]:


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.main = nn.Sequential(
            nn.ConvTranspose2d( 100, 128, 4, 1, 0, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            
            nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            
            nn.ConvTranspose2d( 64, 32, 4, 2, 1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            
            nn.ConvTranspose2d( 32, 3, 4, 2, 1, bias=False),
            nn.Tanh()
        
        )

    def forward(self, input):
        return self.main(input)

netG = Generator().to(device)
netG.apply(weights_init)


# ## Discriminator

# In[ ]:



class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(3, 32, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(32, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(64, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(128, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, input):
        return self.main(input)

netD = Discriminator().to(device)
netD.apply(weights_init)


# ## Loss function and optimizers
# 
# We will be using Binary Cross Entropy loss function as our loss function.

# In[ ]:


# Binary Cross Entropy loss function as our criterion
criterion = nn.BCELoss()

#  the progression of the generator
fixed_noise = torch.randn(32, 100, 1, 1, device=device)

real_label = 1
fake_label = 0

# Using Adam optimizer as our optimizer
optimizerD = optim.Adam(netD.parameters(), lr=0.0002, betas=(0.5, 0.999))
optimizerG = optim.Adam(netG.parameters(), lr=0.0002, betas=(0.5, 0.999))


# ## Plot

# In[ ]:


def display_images(n=5,m=5):
    sample = []
    figure, axes = plt.subplots(n, m)
    k=0
    for i in range(n):
        for j in range(m):
            noise = torch.randn(1, 100, 1, 1, device=device)
            gen_image = netG(noise).to("cpu").clone().detach().squeeze(0)
            gen_image = gen_image.numpy().transpose(1, 2, 0)
            sample.append(gen_image)
            axes[i,j].imshow(sample[k])
            axes[i,j].axis('off')
            k+=1
    plt.show()
    plt.close()


# ## Training

# In[ ]:



# Training Loop
G_losses = []
D_losses = []
iters = 0
num_epochs=1000

print("Starting Training Loop...")
# For each epoch
for epoch in range(num_epochs):
    # For each batch in the dataloader
    for i, data in enumerate(trainloader, 0):

        ############################
        # maximizing log(D(x)) + log(1 - D(G(z)))
        ###########################
        ## Train with all-real batch
        netD.zero_grad()
        real = data[0].to(device)
        b_size = real.size(0)
        label = torch.full((b_size,), real_label, device=device)
        output = netD(real).view(-1)
        errD_real = criterion(output, label)
        errD_real.backward()
        D_x = output.mean().item()

        ## Train with all-fake batch
        noise = torch.randn(b_size, 100, 1, 1, device=device)
        fake = netG(noise)
        label.fill_(fake_label)
        output = netD(fake.detach()).view(-1)
        errD_fake = criterion(output, label)
        errD_fake.backward()
        D_G_z1 = output.mean().item()
        errD = errD_real + errD_fake
        optimizerD.step()

        ############################
        # maximizing log(D(G(z)))
        ###########################
        netG.zero_grad()
        label.fill_(real_label) 
        output = netD(fake).view(-1)
        errG = criterion(output, label)
        errG.backward()
        D_G_z2 = output.mean().item()
        optimizerG.step()

        # Output training stats
        if i % 500 == 0:
            print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f'
                  % (epoch, num_epochs, i, len(trainloader),
                     errD.item(), errG.item()))

        G_losses.append(errG.item())
        D_losses.append(errD.item())

        # Displaying generated images
        if iters % 1000 == 0:
            display_images()

        iters += 1


# In[ ]:


plt.figure(figsize=(10,5))
plt.title("Generator and Discriminator Loss")
plt.plot(G_losses,label="G")
plt.plot(D_losses,label="D")
plt.xlabel("iterations")
plt.ylabel("Loss")
plt.legend()
plt.show()


# In[ ]:




