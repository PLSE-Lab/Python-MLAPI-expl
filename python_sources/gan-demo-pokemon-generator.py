#!/usr/bin/env python
# coding: utf-8

# # Generative Adversarial Network for Pokemon Generation
# This is a simple implementation of the DCGAN (Deep Convolutional Generative Adversarial Network) architecture, as described in [Generative Adversarial Nets](https://arxiv.org/pdf/1406.2661.pdf) by Ian Goodfellow, et. al.
# <br><br>
# The GAN was applied to the Complete Pokemon Image Dataset (24,647 .jpg images of size 160x160px, comprising all existing Pokemon designs) in order to generate new designs.

# In[ ]:


import numpy as np
import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torchvision.utils as tvutils
import torch.utils.data as datautils
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

# use gpu if available
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


# # Data Utility Functions

# In[ ]:


# return iterable over dataset that will load minibatches (size = 128) for training
# preprocessing: images are resized to 64x64, each channel is normalized with mean = 0.5, s = 0.5
def get_loader(datapath):
    dataset = datasets.ImageFolder(root=datapath,transform=transforms.Compose([
                               transforms.Resize(64),
                               transforms.CenterCrop(64),
                               transforms.ToTensor(),
                               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                           ]))
    loader = datautils.DataLoader(dataset, batch_size=128, shuffle=True, num_workers=2)
    return loader


# In[ ]:


# view training data sample images as 8x8 grid
def view_images(loader):
    sample = next(iter(dataloader))
    plt.figure(figsize=(12,12))
    plt.axis('off')
    plt.title('Dataset Sample Images')
    plt.imshow(np.transpose(tvutils.make_grid(sample[0].to(device)[:64], padding=5, normalize=True).cpu(),(1,2,0)))


# # Generator and Discriminator Models

# In[ ]:


class Generator(nn.Module):
    def __init__(self, z_size=100, num_channels=3, gfm_size=64):
        super(Generator, self).__init__()
        self.main = nn.Sequential(
            
            # input layer (latent vector z taken as input)
            nn.ConvTranspose2d(z_size, gfm_size * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(gfm_size * 8),
            nn.ReLU(True),
            
            # state size: (gfm_size * 8) x 4 x 4
            nn.ConvTranspose2d(gfm_size * 8, gfm_size * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(gfm_size * 4),
            nn.ReLU(True),
            
            # state size: (gfm_size * 4) x 8 x 8
            nn.ConvTranspose2d( gfm_size * 4, gfm_size * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(gfm_size * 2),
            nn.ReLU(True),
            
            # state size: (gfm_size * 2) x 16 x 16
            nn.ConvTranspose2d( gfm_size * 2, gfm_size, 4, 2, 1, bias=False),
            nn.BatchNorm2d(gfm_size),
            nn.ReLU(True),
            
            # state size: (gfm_size) x 32 x 32
            nn.ConvTranspose2d(gfm_size, num_channels, 4, 2, 1, bias=False),
            nn.Tanh()
            # output: num_channels x 64 x 64
        )
        
    def forward(self, input):
        return self.main(input)


# In[ ]:


class Discriminator(nn.Module):
    def __init__(self, num_channels=3, dfm_size=64):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            
            # input: num_channels x 64 x 64 
            nn.Conv2d(num_channels, dfm_size, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            
            # state size: dfm_size x 32 x 32
            nn.Conv2d(dfm_size, dfm_size * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(dfm_size * 2),
            nn.LeakyReLU(0.2, inplace=True),
            
            # state size: (dfm_size * 2) x 16 x 16
            nn.Conv2d(dfm_size * 2, dfm_size * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(dfm_size * 4),
            nn.LeakyReLU(0.2, inplace=True),
            
            # state size: (dfm_size * 4) x 8 x 8
            nn.Conv2d(dfm_size * 4, dfm_size * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(dfm_size * 8),
            nn.LeakyReLU(0.2, inplace=True),
            
            # state size: (dfm_size * 8) x 4 x 4
            nn.Conv2d(dfm_size * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, input):
        # view(-1) flattens the output from 2D to 1D 
        return self.main(input).view(-1)


# In[ ]:


# initialize all model weights
def init_weights(model):
    classname = model.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(model.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(model.weight.data, 1.0, 0.02)
        nn.init.constant_(model.bias.data, 0)


# # Importing Data

# In[ ]:


path = '/kaggle/input/complete-pokemon-image-dataset/pokemon/'
dataloader = get_loader(path)
view_images(dataloader)


# In[ ]:


num_epochs = 50
z_size = 100
real_label = 1
fake_label = 0

# creating generator and discriminator networks
gen_net = Generator().to(device)
disc_net = Discriminator().to(device)

# init_weights is applied to self and every submodule recursively
gen_net.apply(init_weights)
disc_net.apply(init_weights)

print(gen_net, disc_net)

# create criterion to measure error (binary cross entropy between target and output)
bce_loss = nn.BCELoss()

# create Adam SGD optimizers with learning rate 0.0002 and beta1 = 0.5
gen_opt = optim.Adam(gen_net.parameters(), lr=0.0002, betas=(0.5, 0.999))
disc_opt = optim.Adam(disc_net.parameters(), lr=0.0002, betas=(0.5, 0.999))

# noise used for visualizing generator's learning curve 
test_noise = torch.randn(64, z_size, 1, 1, device=device)

image_lst = []
gen_losses = []
disc_losses = []
num_iter = 0


# # Training Loop
# Note: In order to obtain images from the generator that looked recognizably Pokemon-like, the number of epochs was increased from 5 (as described in the GAN paper) to 50.

# In[ ]:


for epoch in range(num_epochs):
    for idx,data in enumerate(dataloader):
        
        # DISCRIMINATOR UPDATE

        # set gradients of all model parameters to zero
        disc_net.zero_grad()
        
        # get batch of all real images
        real_batch = data[0].to(device)
        batch_size = real_batch.size(0)
        labels = torch.full((batch_size,), real_label, device=device)
        
        # forward pass real batch through discriminator
        output = disc_net(real_batch)
        disc_loss_real = bce_loss(output, labels)
        
        # calculate gradients for discriminator in backward pass
        disc_loss_real.backward()
        Dx_value = output.mean().item()
        
        # get batch of all fake images
        latent_z = torch.randn(batch_size, z_size, 1, 1, device=device)
        fake_batch = gen_net(latent_z)
        labels.fill_(fake_label)
        
        # classify fake images using discriminator
        output = disc_net(fake_batch.detach())
        disc_loss_fake = bce_loss(output, labels)
        
        # calculate gradients for discriminator in backward pass
        disc_loss_fake.backward()
        DGz_value_1 = output.mean().item()
        
        # add gradients from both batches, save total loss
        disc_loss = disc_loss_real + disc_loss_fake
        disc_losses.append(disc_loss.item())
        
        # update discriminator
        disc_opt.step()
        
        # GENERATOR UPDATE
        
        # set gradients of all model parameters to zero
        gen_net.zero_grad()
        
        # fake labels are "real" in terms of generator cost
        labels.fill_(real_label)
        
        # after updating discriminator, perform another forward pass of fake batch
        output = disc_net(fake_batch)
        
        # calculate and save generator loss on discriminator's new output 
        gen_loss = bce_loss(output, labels)
        gen_losses.append(gen_loss.item())
        
        # calculate gradients for generator in backward pass
        gen_loss.backward()
        DGz_value_2 = output.mean().item()
        
        # update generator
        gen_opt.step()
        
        # print update every 100th batch
        if idx % 100 == 0:
            print('Epoch: %d/%d Idx: %d/%d\nLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
                % (epoch, num_epochs, idx, len(dataloader),
                disc_loss.item(), gen_loss.item(), Dx_value, DGz_value_1, DGz_value_2))
        
        # periodically test and save generator performance on test_noise
        # do not calculate gradients
        if (num_iter % 500 == 0) or ((epoch == num_epochs-1) and (idx == len(dataloader)-1)):           
            with torch.no_grad():
                fake_image = gen_net(test_noise).detach().cpu()
            image_lst.append(tvutils.make_grid(fake_image, padding=2, normalize=True))
        
        num_iter += 1


# # Generator and Discriminator Loss Graph

# In[ ]:


plt.figure(figsize=(10,5))
plt.title("Generator and Discriminator Loss During Training")
plt.plot(gen_losses,label="Generator")
plt.plot(disc_losses,label="Discriminator")
plt.xlabel("Iterations")
plt.ylabel("Loss")
plt.legend()
plt.show()


# # Generated Images

# In[ ]:


# generate 12x12" figure, axes off, showing progression of generated images
fig = plt.figure(figsize=(12,12))
plt.axis("off")
ims = [[plt.imshow(np.transpose(i,(1,2,0)))] for i in image_lst]


# Final set of images.
