#!/usr/bin/env python
# coding: utf-8

# # Imports

# In[1]:


import numpy as np
import pandas as pd
import torch


# In[2]:


np.random.seed(2)
torch.manual_seed(2)


# In[3]:


from torchvision import datasets
import torchvision.transforms as transforms
import torch.nn.functional as F
from torch.optim import lr_scheduler
from torch.utils.data import Dataset, DataLoader
from PIL import Image, ImageOps, ImageEnhance
import math
import random
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# # Creating a Dataset to get images and labels from the csv to get images and labels with some transforms

# In[4]:


class dataset(Dataset):
    def __init__(self, file_path, transform=transforms.Compose([transforms.ToPILImage(), 
                                                                transforms.ToTensor(), 
                                                                transforms.Normalize(mean=(0.5,), 
                                                                                     std=(0.5,))])):
        df = pd.read_csv(file_path)
        if len(df.columns)==n_pixels:
            self.X = df.values.reshape((-1, 28, 28)).astype(np.uint8)[:, :, :, None]
            self.y = None
        else:
            self.X = df.iloc[:, 1:].values.reshape((-1, 28, 28)).astype(np.uint8)[:, :, :, None]
            self.y = torch.from_numpy(df.iloc[:, 0].values)
        self.transform = transform
        
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        if self.y is not None:
            return self.transform(self.X[idx]), self.y[idx]
        return self.transform(self.X[idx])


# In[5]:


test_df = pd.read_csv('../input/test.csv')
test_df.head()


# In[6]:


n_pixels = len(test_df.columns)
n_pixels


# For training, we use the data augmentations of random rotation. For testing, we just create normal tensors

# In[7]:


num_workers = 0
batch_size = 64
transform = transforms.Compose([transforms.ToPILImage(), transforms.RandomRotation(degrees=20),
                                transforms.ToTensor(), transforms.Normalize(mean=(0.5,), std=(0.5,))])

train_dataset = dataset('../input/train.csv', transform=transform)
test_dataset = dataset('../input/test.csv')
train_dl = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_dl = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


# Plot one batch of images

# In[8]:


dataiter = iter(train_dl)
images, labels = dataiter.next()
images = images.numpy()

fig = plt.figure(figsize=(25, 4))
for idx in np.arange(batch_size):
    ax = fig.add_subplot(2, batch_size/2, idx+1, xticks=[], yticks=[])
    ax.imshow(np.squeeze(images[idx]), cmap='gray')
    ax.set_title(str(labels[idx].item()))


# In[9]:


import torch.nn as nn


# # A Capsule net has 3 main parts - 
# 1. A conv layer
# 2. A primary capsule
# 3. A digit capsule

# The conv layer applies a normal convolution with kernel size of 9 and output channels of 256. The output thus produced is of size 20x20x256

# In[11]:


class ConvLayer(nn.Module):
    def __init__(self, in_channels=1, out_channels=256):
        super(ConvLayer, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=9, stride=1, padding=0)
    def forward(self, x):
        x = F.relu(self.conv(x))
        return x


# The primary capsule is just 8 stacked convolutions, whose output is then squashed.

# In[14]:


class PrimaryCaps(nn.Module):
    def __init__(self, num_capsules=8, in_channels=256, out_channels=32):
        super(PrimaryCaps, self).__init__()
        self.capsules = nn.ModuleList([
            nn.Conv2d(in_channels, out_channels, kernel_size=9, stride=2, padding=0)
            for _ in range(num_capsules)
        ])
    def forward(self, x):
        batch_size = x.size(0)
        u = [capsule(x).view(batch_size, 32*6*6, 1) for capsule in self.capsules]
        u = torch.cat(u, dim=-1)
        u_squashed = self.squash(u)
        return u_squashed
    
    def squash(self, x):
        squared_norm = (x**2).sum(dim=-1, keepdim=True)
        scale = squared_norm/(1+squared_norm)
        output = scale * x/torch.sqrt(squared_norm)
        return output


# In[15]:


def softmax(x, dim=1):
    transposed_inp = x.transpose(dim, len(x.size())-1)
    softmaxed = F.softmax(transposed_inp.contiguous().view(-1, transposed_inp.size(-1)), dim=-1)
    return softmaxed.view(*transposed_inp.size()).transpose(dim, len(x.size())-1)


# In[16]:


def dynamic_routing(b_ij, u_hat, squash, routing_iterations=3):
    for iterations in range(routing_iterations):
        c_ij = softmax(b_ij, dim=2)
        s_j = (c_ij*u_hat).sum(dim=2, keepdim=True)
        v_j = squash(s_j)
        if iterations < routing_iterations-1:
            a_ij = (u_hat * v_j).sum(dim=-1, keepdim=True)
            b_ij = b_ij + a_ij
    return v_j


# In[17]:


TRAIN_ON_GPU = torch.cuda.is_available()
if TRAIN_ON_GPU: print('training on gpu')


# The digit capusle takes the 8 primary capsules and produces 10 capsules as output. These 10 capsules correspond to the 10 classes of MNIST. The digit capsules apply dynamic routing on the primary capsules to select the children which corresponds the maximum to each capsule.

# In[18]:


class DigitCaps(nn.Module):
    def __init__(self, num_caps=10, previous_layer_nodes=32*6*6,
                 in_channels=8, out_channels=16):
        super(DigitCaps, self).__init__()
        self.num_caps = num_caps
        self.previous_layer_nodes = previous_layer_nodes
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.W = nn.Parameter(torch.randn(num_caps, previous_layer_nodes,
                                          in_channels, out_channels))
    
    def forward(self, x):
        x = x[None, :, :, None, :]
        W = self.W[:, None, :, :, :]
        x_hat = torch.matmul(x, W)
        b_ij = torch.zeros(*x_hat.size())
        if TRAIN_ON_GPU: b_ij = b_ij.cuda()
        v_j = dynamic_routing(b_ij, x_hat, self.squash, routing_iterations=3)
        return v_j
    
    def squash(self, x):
        squared_norm = (x**2).sum(dim=-1, keepdim=True)
        scale = squared_norm/(1+squared_norm)
        out = scale * x/torch.sqrt(squared_norm)
        return out


# For getting reconstructed images, we also create a decoder, which takes the 10 digit capsules and after a series of linear and relu operations, convert the capsules again into a 28x28 image

# In[19]:


class Decoder(nn.Module):
    def __init__(self, input_vector_length=16, input_capsules=10, hidden_dim=512):
        super(Decoder, self).__init__()
        input_dim = input_vector_length*input_capsules
        self.lin_layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim*2),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim*2, 28*28),
            nn.Sigmoid()
        )
    def forward(self, x):
        classes = (x**2).sum(dim=-1)**0.5
        classes = F.softmax(classes, dim=-1)
        _, max_length_indices = classes.max(dim=1)
        sparse_matrix = torch.eye(10)
        if TRAIN_ON_GPU: sparse_matrix = sparse_matrix.cuda()
        y = sparse_matrix.index_select(dim=0, index=max_length_indices.data)
        x = x*y[:, :, None]
        flattened_x = x.view(x.size(0), -1)
        reconstructed = self.lin_layers(flattened_x)
        return reconstructed, y


# We apply all these layers to create capsule network

# In[20]:


class CapsuleNetwork(nn.Module):
    def __init__(self):
        super(CapsuleNetwork, self).__init__()
        self.conv_layer = ConvLayer()
        self.primary_capsule = PrimaryCaps()
        self.digit_capsule = DigitCaps()
        self.decoder = Decoder()
    def forward(self, x):
        primary_caps_out = self.primary_capsule(self.conv_layer(x))
        caps_out = self.digit_capsule(primary_caps_out).squeeze().transpose(0, 1)
        reconstructed, y = self.decoder(caps_out)
        return caps_out, reconstructed, y


# In[21]:


capsule_net = CapsuleNetwork()

print(capsule_net)

if TRAIN_ON_GPU: capsule_net = capsule_net.cuda()


# For the loss function of our network, we use the CapsuleLoss module defined below. 

# In[22]:


class CapsuleLoss(nn.Module):
    def __init__(self):
        super(CapsuleLoss, self).__init__()
        self.reconstruction_loss = nn.MSELoss(size_average=False)
    
    def forward(self, x, labels, images, reconstructions):
        batch_size = x.size(0)
        v_c = torch.sqrt((x**2).sum(dim=2, keepdim=True))
        left = F.relu(0.9-v_c).view(batch_size, -1)
        right = F.relu(v_c-0.1).view(batch_size, -1)
        margin_loss = labels * left + 0.5 * (1.-labels) * right
        margin_loss = margin_loss.sum()
        images = images.view(reconstructions.size()[0], -1)
        reconstruction_loss = self.reconstruction_loss(reconstructions, images)
        return (margin_loss + 0.0005 * reconstruction_loss)/images.size(0)


# In[23]:


import torch.optim as optim
criterion = CapsuleLoss()
optimizer = optim.Adam(capsule_net.parameters())


# In[24]:


def train(capsule_net, criterion, optimizer, n_epochs, print_every=300):
    losses = []
    for epoch in range(1, n_epochs+1):
        train_loss = 0.0
        capsule_net.train() 
        for batch_i, (images, target) in enumerate(train_dl):
            target = torch.eye(10).index_select(dim=0, index=target)
            if TRAIN_ON_GPU: images, target = images.cuda(), target.cuda()
            optimizer.zero_grad()
            caps_output, reconstructions, y = capsule_net(images)
            loss = criterion(caps_output, target, images, reconstructions)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            if batch_i != 0 and batch_i % print_every == 0:
                avg_train_loss = train_loss/print_every
                losses.append(avg_train_loss)
                print('Epoch: {} \tTraining Loss: {:.8f}'.format(epoch, avg_train_loss))
                train_loss = 0 
    return losses


# In[25]:


n_epochs = 10


# # Training the model and plotting the losses

# In[26]:


losses = train(capsule_net, criterion, optimizer, n_epochs=n_epochs)


# In[27]:


plt.plot(losses)
plt.title('Training Loss')
plt.show()


# In[60]:


out = []


# In[61]:


capsule_net.eval()
for image in test_dl:
    if TRAIN_ON_GPU: image = image.cuda()
    caps_out, reconstructed, y = capsule_net(image)
    _, pred = torch.max(y.data.cpu(), 1)
    out.extend(pred.numpy().tolist())


# In[62]:


len(out)


# In[63]:


sub = pd.read_csv('../input/sample_submission.csv')


# In[64]:


sub['Label'] = out


# In[65]:


sub.to_csv('capsule.csv', index=False)


# This model gives 98.8% accuracy on public LB. This can further be improved by training more and adding more data augmentation

# In[ ]:




