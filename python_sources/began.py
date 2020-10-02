#!/usr/bin/env python
# coding: utf-8

# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')


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
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from IPython.display import HTML

# Set random seem for reproducibility
manualSeed = 520
#manualSeed = random.randint(1, 10000) # use if you want new results
print("Random Seed: ", manualSeed)
random.seed(manualSeed)
torch.manual_seed(manualSeed)


# In[ ]:


# Root directory for dataset
dataroot = "../input/celeba-dataset/img_align_celeba"

# Number of workers for dataloader
workers = 1

# Batch size during training
batch_size = 16

# Spatial size of training images. All images will be resized to this
#   size using a transformer.
image_size = 64

# Number of channels in the training images. For color images this is 3
nc = 3

# Size of z latent vector (i.e. size of generator input)
nz = 64

# Size of feature maps in generator
ngf = 64

# Size of feature maps in discriminator
ndf = 64

# Number of training epochs
num_epochs = 5

# Learning rate for optimizers
lr = 0.0001

# Beta1 hyperparam for Adam optimizers
beta1 = 0.2

# Number of GPUs available. Use 0 for CPU mode.
ngpu = 1

# Decide which device we want to run on
device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")

# Faster
torch.backends.cudnn.benchmark = True

print(device)
print(ngpu)


# In[ ]:


# We can use an image folder dataset the way we have it setup.
# Create the dataset
dataset = dset.ImageFolder(root=dataroot,
                           transform=transforms.Compose([
                               transforms.Resize(image_size),
                               transforms.CenterCrop(image_size),
                               transforms.ToTensor(),
                               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                           ]))
# Create the dataloader
dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                         shuffle=True, num_workers=workers)

# Plot some training images
real_batch = next(iter(dataloader))
plt.figure(figsize=(8,8))
plt.axis("off")
plt.title("Training Images")
plt.imshow(np.transpose(vutils.make_grid(real_batch[0].to(device)[:64], padding=2, normalize=True).cpu(),(1,2,0)))


# In[ ]:


def noise():
  return 2*torch.rand(nz, ngf, device = device) - 1


# In[ ]:


# Generator Code

class Generator(nn.Module):
    def __init__(self, ngpu):
        super(Generator, self).__init__()
        self.ngpu = ngpu
        self.fromLatent = nn.Linear(ngf, (image_size*image_size//16)*ndf)
        self.dlayer1 = nn.Sequential(
            nn.Conv2d(ndf, ndf, 3, padding=1),
            nn.BatchNorm2d(ndf),
            nn.ELU(inplace=True),
            nn.Conv2d(ndf, ndf, 3, padding=1),
            nn.BatchNorm2d(ndf),
            nn.ELU(inplace=True)
        )
        self.dlayer2 = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(2 * ndf, ndf, 3, padding=1),
            nn.BatchNorm2d(ndf),
            nn.ELU(inplace=True),
            nn.Conv2d(ndf, ndf, 3, padding=1),
            nn.BatchNorm2d(ndf),
            nn.ELU(inplace=True)
        )
        self.dlayer3 = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(2 * ndf, ndf, 3, padding=1),
            nn.BatchNorm2d(ndf),
            nn.ELU(inplace=True),
            nn.Conv2d(ndf, ndf, 3, padding=1),
            nn.BatchNorm2d(ndf),
            nn.ELU(inplace=True)
        )
        self.toImage = nn.Conv2d(ndf, 3, 3, padding = 1) 

    def forward(self, x):
      x = self.fromLatent(x)
      x = x.view(x.size(0), ndf, image_size//4, image_size//4)
      h0 = x
      x = self.dlayer1(x)
      x = torch.cat([x, h0], dim=1)
      x = self.dlayer2(x)
      h0 = nn.functional.interpolate(h0, scale_factor=2, mode='nearest') 
      x = torch.cat([x, h0], dim=1)
      x = self.dlayer3(x)
      x = self.toImage(x)
      return x


# In[ ]:


class Discriminator(nn.Module):
    def __init__(self, ngpu):
        super(Discriminator, self).__init__()
        self.ngpu = ngpu
        self.elayer1 = nn.Sequential(
            nn.Conv2d(nc, ndf, 3, padding = 1),
            nn.BatchNorm2d(ndf),
            nn.ELU(inplace=True),
            nn.Conv2d(ndf, ndf, 3, padding = 1),
            nn.BatchNorm2d(ndf),
            nn.ELU(inplace=True),
            nn.Conv2d(ndf, 2 * ndf, 3, padding = 1),
            nn.BatchNorm2d(2 * ndf),
            nn.ELU(inplace=True),
        )
        self.elayer2 = nn.Sequential(
            nn.MaxPool2d(2),
            nn.Conv2d(2 * ndf, 2 * ndf, 3, padding = 1),
            nn.BatchNorm2d(2 * ndf),
            nn.ELU(inplace=True),
            nn.Conv2d(2 * ndf, 3 * ndf, 3, padding = 1),
            nn.BatchNorm2d(3 * ndf),
            nn.ELU(inplace=True)
        )
        self.elayer3 = nn.Sequential(
            nn.MaxPool2d(2),
            nn.Conv2d(3 * ndf, 3 * ndf, 3, padding = 1),
            nn.BatchNorm2d(3 * ndf),
            nn.ELU(inplace=True),
            nn.Conv2d(3 * ndf, 3 * ndf, 3, padding = 1),
            nn.BatchNorm2d(3 * ndf),
            nn.ELU(inplace=True)
        )
        self.toLatent = nn.Linear((image_size*image_size//16)*3*ndf, ngf)

        self.fromLatent = nn.Linear(ngf, (image_size*image_size//16)*ndf)
        self.dlayer1 = nn.Sequential(
            nn.Conv2d(ndf, ndf, 3, padding=1),
            nn.BatchNorm2d(ndf),
            nn.ELU(inplace=True),
            nn.Conv2d(ndf, ndf, 3, padding=1),
            nn.BatchNorm2d(ndf),
            nn.ELU(inplace=True)
        )
        self.dlayer2 = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(2*ndf, ndf, 3, padding=1),
            nn.BatchNorm2d(ndf),
            nn.ELU(inplace=True),
            nn.Conv2d(ndf, ndf, 3, padding=1),
            nn.BatchNorm2d(ndf),
            nn.ELU(inplace=True)
        )
        self.dlayer3 = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(2*ndf, ndf, 3, padding=1),
            nn.BatchNorm2d(ndf),
            nn.ELU(inplace=True),
            nn.Conv2d(ndf, ndf, 3, padding=1),
            nn.BatchNorm2d(ndf),
            nn.ELU(inplace=True)
        )
        self.toImage = nn.Conv2d(ndf, 3, 3, padding = 1)

    def forward(self, x):
      x = self.elayer1(x)
      x = self.elayer2(x)
      x = self.elayer3(x)
      x = x.view(x.size(0), -1)
      x = self.toLatent(x)
      x = self.fromLatent(x)
      x = x.view(x.size(0), ndf, image_size//4, image_size//4)
      h0 = x
      x = self.dlayer1(x)
      x = torch.cat([x, h0], dim=1)
      x = self.dlayer2(x)
      h0 = torch.nn.functional.interpolate(h0, scale_factor=2, mode='nearest')
      x = torch.cat([x, h0], dim=1)
      x = self.dlayer3(x)
      x = self.toImage(x)
      return x


# In[ ]:


# Create the generator
netG = Generator(ngpu).to(device)

# Handle multi-gpu if desired
if (device.type == 'cuda') and (ngpu > 1):
    netG = nn.DataParallel(netG, list(range(ngpu)))

# Apply the weights_init function to randomly initialize all weights
#  to mean=0, stdev=0.2.
# netG.apply(weights_init)

# Print the model
print(netG)


# In[ ]:


# Create the Discriminator
netD = Discriminator(ngpu).to(device)

# Handle multi-gpu if desired
if (device.type == 'cuda') and (ngpu > 1):
    netD = nn.DataParallel(netD, list(range(ngpu)))
    
# Apply the weights_init function to randomly initialize all weights
#  to mean=0, stdev=0.2.
#netD.apply(weights_init)

# Print the model
print(netD)


# In[ ]:


# Initialize BCELoss function
criterion = nn.L1Loss()

# Create batch of latent vectors that we will use to visualize
#  the progression of the generator
fixed_noise = noise()

# Setup Adam optimizers for both G and D
optimizerD = optim.Adam(netD.parameters(), lr=lr, betas=(beta1, 0.999))
optimizerG = optim.Adam(netG.parameters(), lr=lr, betas=(beta1, 0.999))


# In[ ]:


import time

class Timer():
  def __init__(self):
    self.startTime = time.time()
    self.lastTime = time.time()
  
  def timeElapsed(self):
    auxTime = self.lastTime
    self.lastTime = time.time()
    return self.lastTime - auxTime

  def timeSinceStart(self):
    self.lastTime = time.time()
    return self.lastTime - self.startTime


# In[ ]:


# Training Loop
k = 0
gamma = 0.4
lambda_k = 0.005
# Lists to keep track of progress
img_list = []
G_losses = []
D_losses = []
iters = 0

print("Starting Training Loop...")
# For each epoch
timer = Timer()
for epoch in range(num_epochs):
    # For each batch in the dataloader
    for i, data in enumerate(dataloader, 0):
        ############################
        # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
        ###########################
        ## Train with all-real batch
        netD.zero_grad()
        # Format batch
        real_cpu = data[0].to(device)
        b_size = real_cpu.size(0)
        # Forward pass real batch through D
        output = netD(real_cpu).view(-1)
        # Calculate loss on all-real batch
        errD_real = criterion(output, real_cpu.view(-1))
        # Calculate gradients for D in backward pass
        D_x = output.mean().item()

        ## Train with all-fake batch
        # Generate batch of latent vectors
        # Generate fake image batch with G
        fake = netG(noise())
        # Classify all fake batch with D
        output = netD(fake.detach()).view(-1)
        # Calculate D's loss on the all-fake batch
        errD_fake = criterion(output, fake.view(-1))
        # Calculate the gradients for this batch
        D_loss = errD_real - k * errD_fake
        D_loss.backward()
        D_G_z1 = output.mean().item()
        # Add the gradients from the all-real and all-fake batches
        optimizerD.step()

        ############################
        # (2) Update G network: maximize log(D(G(z)))
        ###########################
        netG.zero_grad()
        fake = netG(noise())
        # Since we just updated D, perform another forward pass of all-fake batch through D
        output = netD(fake).view(-1)
        # Calculate G's loss based on this output
        errG = criterion(output, fake.view(-1))
        # Calculate gradients for G
        errG.backward()
        D_G_z2 = output.mean().item()
        # Update G
        optimizerG.step()

        delta = (gamma*errD_real - errG).data
        k = max(min(k + lambda_k*delta, 1.0), 0.0)
        
        # Output training stats
        if i % 50 == 0:
            print('[%.4f] [%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
                  % (timer.timeElapsed(), epoch, num_epochs, i, len(dataloader),
                     D_loss.item(), errG.item(), D_x, D_G_z1, D_G_z2))
        
        # Save Losses for plotting later
        G_losses.append(errG.item())
        D_losses.append(D_loss.item())
        
        # Check how the generator is doing by saving G's output on fixed_noise
        if (iters % 1000 == 0) or ((epoch == num_epochs-1) and (i == len(dataloader)-1)):
            with torch.no_grad():
                fake = netG(fixed_noise).detach().cpu()
            img_list.append(vutils.make_grid(fake, padding=2, normalize=True))
            
        iters += 1


# In[ ]:


plt.figure(figsize=(10,5))
plt.title("Generator and Discriminator Loss During Training")
plt.plot(G_losses,label="G")
plt.plot(D_losses,label="D")
plt.xlabel("iterations")
plt.ylabel("Loss")
plt.legend()
plt.show()   


# In[ ]:


#%%capture
fig = plt.figure(figsize=(8,8))
plt.axis("off")
ims = [[plt.imshow(np.transpose(i,(1,2,0)), animated=True)] for i in img_list]
ani = animation.ArtistAnimation(fig, ims, interval=1000, repeat_delay=1000, blit=True)

HTML(ani.to_jshtml())


# In[ ]:


# Grab a batch of real images from the dataloader
real_batch = next(iter(dataloader))

# Plot the real images
plt.figure(figsize=(15,15))
plt.subplot(1,2,1)
plt.axis("off")
plt.title("Real Images")
plt.imshow(np.transpose(vutils.make_grid(real_batch[0].to(device)[:64], padding=5, normalize=True).cpu(),(1,2,0)))

# Plot the fake images from the last epoch
plt.subplot(1,2,2)
plt.axis("off")
plt.title("Fake Images")
plt.imshow(np.transpose(img_list[-1],(1,2,0)))
plt.savefig('fake_image.png')
plt.show()


# In[ ]:




