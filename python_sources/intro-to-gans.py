#!/usr/bin/env python
# coding: utf-8

# # Generative Adversarial Networks
# 
# Welcome to this tutorial on generative adversarial networks, typically called GANs. The GAN architecture has proven to produce state-of-the-art performance on image generation tasks. For example, the GAN in this tutorial produced these images of hand-written digits.
# 
# <center><img src="https://i.imgur.com/q9z0Vcc.png"></center>
# <br>
# 
# In this kernel, I'll show you the basic concepts behind GANs and how to implement a simple GAN with PyTorch.

# ## Networks in competition
# 
# The main idea behind GANs is there are two networks competing with each other, a generator and a discriminator. The generator creates fake data while the discriminator does its best to tell if the data is fake or real. Theoretically, the generator could be used to create any type of data, but so far it's worked best with images. The networks are trained by showing the discriminator real images from some dataset and fake images from the generator. As the discriminator gets better at detecting fake images, the generator gets better at creating realistic images.
# 
# <center><img src="https://i.imgur.com/oXfBdjf.png"></center>

# The generator takes a latent noise vector that it uses to generate an image. Conceptually, the generator maps the distribution of the latent vectors to the distribution of the real data. This means as you adjust the latent vector you are producing different images from the distribution of real images.
# 
# For this GAN, I'll be using the MNIST dataset of hand-written digits. The goal then is to train the generator to create images of hand-written digits. I'll be implementing the GAN in PyTorch, my preferred deep learning framework. First up, importing the modules we'll be using. Then I'll define the generator and discriminator networks, and finally show you how the networks are trained.

# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn, optim
import torch.nn.functional as F
from torchvision import datasets, transforms


# Here I'm using `torchvision` to load the MNIST images. I'm also normalizing the images so they have values between -1 and 1. This is important as we'll make the generator output tensors with values between -1 and 1. I'm setting the batch size here too, this is how many images we'll pass to the networks in each update step.

# In[ ]:


batch_size = 64

transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize((0.5,), (0.5,))])

# You need to have internet access on to get this data
train_data = datasets.MNIST('~/pytorch/MNIST_data', download=True, transform=transform)
train_loader = torch.utils.data.DataLoader(train_data, shuffle=True,
                                           batch_size=batch_size)


# Here we can see one of the images.

# In[ ]:


imgs, label = next(iter(train_loader))
imgs = imgs.numpy().transpose(0, 2, 3, 1)
plt.imshow(imgs[1].squeeze())


# # Models
# 
# Next I'll define the generator and discriminator models for the GAN architecture. These will be simple dense networks with one hidden layer.
# 
# <center><img src="https://imgur.com/M8Ev03g.png" width=550px></center>

# First I'll define the generator then the discriminator.
# 
# ### The Generator
# 
# The goal of the generator is to take a noise vector and convert it into a 28x28 image. Important things here:
# 
# - Leaky ReLU activations on the dense hidden layers. The output of normal ReLUs have a lot of zeros which leads to a lot of zeros in the gradients as well. This tends to make training GANs difficult, so we'll use leaky ReLUs to avoid these sparse gradients
# - Tanh on the output layer. The images we generate should have values between -1 and 1.
# - Reshape the final output to be the same shape as the real images
# 

# In[ ]:


class Generator(nn.Module):
    def __init__(self, nz, nhidden):
        super(Generator, self).__init__()

        # input is Z, noise vector, going into a dense layer
        self.fc1 = nn.Linear(nz, nhidden)
        self.fc2 = nn.Linear(nhidden, nhidden)
        # Output layer. I'll reshape in forward()
        self.fc3 = nn.Linear(nhidden, 784)

    def forward(self, x):
        x = F.leaky_relu(self.fc1(x), 0.2)
        x = F.leaky_relu(self.fc2(x), 0.2)
        x = torch.tanh(self.fc3(x))
        
        # Reshape to (batch-size, color channels, width, height)
        return x.view(-1, 1, 28, 28)


# ### Discriminator
# 
# The discriminator will be a common binary classification network. Again, using leaky ReLUs, but otherwise a normal classifier. The output will be the probability that the input is real.

# In[ ]:


class Discriminator(nn.Module):
    def __init__(self, nhidden):
        super(Discriminator, self).__init__()

        self.fc1 = nn.Linear(784, nhidden)
        self.fc2 = nn.Linear(nhidden, nhidden)
        self.fc3 = nn.Linear(nhidden, 1)
        
    def forward(self, x):
        x = x.view(-1, 28 * 28)
        x = F.leaky_relu(self.fc1(x))
        x = F.leaky_relu(self.fc2(x))
        x = torch.sigmoid(self.fc3(x))
        
        return x.view(-1, 1)


# # Creating functions for displaying images
# 
# It's useful to watch the generated images as the networks train so I'm creating a couple functions here to  generate images from the tensors.

# In[ ]:


def imshow(tensor):
    """ Display a tensor as an image. """
    image = tensor.to("cpu").clone().detach()
    image = image.numpy().transpose(1, 2, 0)
    plt.imshow(image.squeeze()*2 + 0.5)
    plt.tight_layout()
    plt.axis('off')
    
def image_grid(tensor, figsize = (6,6)):
    """ Display batched images as a grid. Batch size must be a perfect square. """
    bs, c, w, h = tensor.shape
    image_grid = torch.zeros(c, int(w*np.sqrt(bs)), int(h*np.sqrt(bs)))
    for ii, img in enumerate(tensor):
        x = (ii % int(np.sqrt(bs))) * w
        y = (ii // int(np.sqrt(bs))) * h
        image_grid[:, x: x + w, y: y + h] = img
    plt.figure(figsize=figsize)
    plt.tight_layout()
    imshow(image_grid)


# # Training the networks
# 
# Writing the code for training the networks is surprisingly straightforward. We'll update the networks in three stages:
# 
# 1. Pass a batch of real images to the discriminator, set the labels to "REAL", calculate the loss, and get the gradients. This improves the discriminator on real images.
# 2. Pass a batch of fake images to the discriminator, set the labels to "FAKE", calculate the loss, get the gradients, and update the discriminator. This improves the discriminator on fake images.
# 3. Pass a batch of fake images to the discriminator, set the labels to "REAL", calculate the loss, and update the generator. We want the discriminator to think the fake images are real, so using "REAL" labels will train the generator to make images that the discriminator observes as real.
# 
# However, finding the correct hyperparameters can be difficult. Since the generator and discriminator are competing, they need to be balanced so one doesn't dominate the other. The most effective way I've found to balance the models is adjusting the size of the generator and discriminator networks. You can add more hidden units to make wider networks, or add more layers.
# 
# ### Label smoothing
# Stability is improved if the labels for real and fake images are slightly different than 1 and 0, respectively. From what I've seen, most implementations have the label for real images as 0.9 and fake images as 0.
# 
# ### Setting up things for training
# 
# Here I'm going to define some parameters for training, create the models, define the loss, and create the optimizers.

# In[ ]:


# Use a GPU if one is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

lr = 0.0003
batch_size = train_loader.batch_size
beta1 = 0.5   # parameter for Adam optimizer

# Fixed latent noise vectors for observing training progress
nz = 100
fixed_noise = torch.randn(25, nz, device=device)

netG = Generator(nz, 300).to(device)
netD = Discriminator(50).to(device)

criterion = nn.BCELoss()

optimizerD = optim.Adam(netD.parameters(), lr=lr, betas=(beta1, 0.999))
optimizerG = optim.Adam(netG.parameters(), lr=lr, betas=(beta1, 0.999))

# Labels we'll use for training, smoothing for the real labels
real_label = 0.9
fake_label = 0


# # Training loop
# 
# Finally we're at the training loop. You'll want to monitor the training progress to see if the networks are stable and learning together. To do this, I have it printing out a few metrics.
# 
# - The discriminator loss, `Loss_D`. If this drops to 0, something is wrong.
# - The average prediction probability on real images `D(x)`. Around 0.5 - 0.7 is good.
# - The average prediction probability on fake images `D(G(z))`, before and after training the discriminator on fake images. Around 0.3 - 0.5 is good.
# 
# I've found it most informative to watch `D(x)` and `D(G(z))`. If the discriminator is really good at detecting real images, `D(x)` will be close to 1 since it's predicting nearly all the real images as real. This means though that it's also really good at detecting fakes and `D(G(z))` will be near 0. In this case the discriminator is too strong. Try making the generator stronger or the discriminator weaker.
# 
# If the generator is successfully fooling the discriminator, `D(x)` and `D(G(z))` should be around 0.5 since the discriminator can't tell between the real and fake images. You'll usually find that `D(x)` is a bit larger that 0.5, but as long as the two networks are roughly balanced, the generator will continue to improve.
# 
# I've also set it up to generate images after each epoch. This way you can watch the generator get better over time. If the images continue to look like noise, something is wrong. However, sometimes it can take a while before the images look like anything real, so I prefer to watch `D(x)` and `D(G(z))`.

# In[ ]:


# You can get pretty good results with 80 epochs. More is better.
epochs = 80

step = 0
for epoch in range(epochs):
    for ii, (real_images, _) in enumerate(train_loader):
        ############################
        # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
        ###########################
        # Sometimes you'll get not full batches, so need to be flexible
        batch_size = real_images.size(0)
        
        # Train discriminator on real images
        real_images = real_images.to(device)
        labels = torch.full((batch_size, 1), real_label, device=device)
        netD.zero_grad()
        output = netD(real_images)
        errD_real = criterion(output, labels)
        errD_real.backward()
        
        # For monitoring training progress
        D_x = output.mean().item()

        # Train with fake
        noise = torch.randn(batch_size, nz, device=device)
        fake = netG(noise)
        # Changing labels in-place because gotta go fast
        labels.fill_(fake_label)
        # Detach here so we don't backprop through to the generator
        output = netD(fake.detach())
        errD_fake = criterion(output, labels)
        errD_fake.backward()
        optimizerD.step()
        
        # For monitoring training progress
        D_G_z1 = output.mean().item()
        errD = errD_real + errD_fake

        ############################
        # (2) Update G network: maximize log(D(G(z)))
        ###########################
        netG.zero_grad()
        labels.fill_(real_label)  # Real labels for fake images
        output = netD(fake)
        errG = criterion(output, labels)
        errG.backward()
        optimizerG.step()
        
        # For monitoring training progress
        D_G_z2 = output.mean().item()
        
        if step % 200 == 0:
            print('[%d/%d][%d/%d] Loss_D: %.4f Loss_G: %.4f D(x): %.4f D(G(z)): %.4f / %.4f'
                  % (epoch + 1, epochs, ii, len(train_loader),
                     errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))
            
        step += 1
    else:
        valid_image = netG(fixed_noise)
        image_grid(valid_image)
        plt.show()


# After training, I'm displaying generate images from the final model and real images, for comparison.

# In[ ]:


image_grid(valid_image)


# In[ ]:


real_images, _ = next(iter(train_loader))
image_grid(real_images[:25])


# # Save networks
# 
# Here we can save the networks if we want to load them again.

# In[ ]:


torch.save(netG.state_dict(), 'generator.pth')
torch.save(netD.state_dict(), 'discriminator.pth')


# # Save generated images
# 
# Finally we want to generate a bunch of images and save them. For the kernel to work properly, we'll need to make sure the images are archived in a zip file and the individual images are deleted. Also, torchvision provides a utility function that speeds up saving the images.

# In[ ]:


get_ipython().system('mkdir images')


# In[ ]:


from torchvision.utils import save_image


# In[ ]:


noise = torch.randn(10000, nz, device=device)
image_tensors = netG(noise)
# Move values back to be between 0 and 1
image_tensors = (image_tensors * 0.5 + 0.5)
for ix, image in enumerate(image_tensors):
    save_image(image, f'images/image_{ix:05d}.png')


# Now I'll zip up the image directory into an archive and delete the image folder itself.

# In[ ]:


import shutil
shutil.make_archive('images', 'zip', 'images')


# In[ ]:


get_ipython().system('rm -r images')


# In the next tutorial, I'll show you how build a more powerful GAN with convolutional networks, called DCGANs.
