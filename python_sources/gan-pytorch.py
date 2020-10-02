# Importing Libraries
import os
from tqdm import tqdm

import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, utils

path = '/kaggle/working/output'

# create output folder if doesn't exist
os.makedirs(path, exist_ok=True)

# shape of the image (channel, height, width)
img_shape = (1, 28, 28)

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


# Loss Function
loss_func = nn.BCELoss()

# Model's initialization
generator = Generator()
discriminator = Discriminator()

# Dataset and DataLoader
dataset = torch.utils.data.DataLoader(
    datasets.MNIST('data/', 
                   train=True, 
                   download=True,
                   transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize(mean=(0.5,), std=(0.5,))
                   ])), 
    batch_size=64, 
    shuffle=True
)

# Send to GPU if available
if torch.cuda.is_available():
    generator.cuda()
    discriminator.cuda()
    loss_func.cuda()

# Optimizer for generator and discriminator
optimizer_G = torch.optim.Adam(generator.parameters(), lr=0.0002, betas=(0.4, 0.999))
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.4, 0.999))


# Trainig Loop
for epoch in tqdm(range(20), desc="Epochs"):
    for i, (real_images, _) in tqdm(enumerate(dataset), desc="Batches", total=940):

        # ground truth (tensors of ones and zeros) same shape as images
        valid = torch.ones(real_images.size(0), 1)
        fake = torch.zeros(real_images.size(0), 1)

        # seng images to GPU if available
        if torch.cuda.is_available():
            real_images.cuda()

        # zero grading the generator optimizer
        optimizer_G.zero_grad()

        # Generating Noise (input for the generator)
        gen_input = torch.randn(real_images.shape[0], 100)

        # Converting noise to images
        gen_images = generator(gen_input)

        # Calculating loss and backward propogation, updating optimizer
        # How well the generator can create real images
        g_loss = loss_func(discriminator(gen_images), valid)
        g_loss.backward()
        optimizer_G.step()

        # zero grading the discriminator optimizer
        optimizer_D.zero_grad()
        
        # Calculating loss and backward propogation, updating optimizer
        # How well discriminator identifies the real and fake images
        real_loss = loss_func(discriminator(real_images), valid)
        fake_loss = loss_func(discriminator(gen_images.detach()), fake)
        d_loss = (real_loss + fake_loss) / 2.0
        d_loss.backward()
        optimizer_D.step()

        # Saving 5x5 grid every 400 batches
        total_batch = epoch * len(dataset) + i
        if total_batch % 400 == 0:
            utils.save_image(gen_images.data[:25], path+'/%d.png' % total_batch, nrow=5, padding=0, normalize=True)

    # printing the model losses at each epoch
    tqdm.write(f"[Epoch {epoch}/{20}] [D loss: {d_loss.item()}] [G loss: {g_loss.item()}]")