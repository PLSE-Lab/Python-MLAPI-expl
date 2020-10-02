#!/usr/bin/env python
# coding: utf-8

# # Autoencoder step by step

# # Importing the essencial libriries

# In[ ]:


import matplotlib.pyplot as plt
import torch
from torch import nn
import torchvision
from torchvision import datasets
import pandas as pd
import torchvision.transforms as transforms
import torch.nn.functional as F
import numpy as  np
import seaborn as sns


# # Load the Fashion-MNIST dataset and transform into a Tensor

# ### Transform a  PIL image or a numpy.ndarray to a tensor
# ```Python
#     transforms.ToTensor()
# ```
# will transform our Fashion - MNIST dataset into a tensor
# 
# 
# ref: https://pytorch.org/docs/stable/torchvision/transforms.html

# In[ ]:


transform = transforms.ToTensor()


# ### downloading the MNIST dataset and applying the transform

# ```Python
# datasets.FashionMNIST(root='data', train=True,
#                            download=True, transform=transform)
# ```
# This is a piece of code that download a data set and apply some transformation.
# Here is all datasets you can load using Pytorch torchvision.datasets: https://pytorch.org/docs/stable/torchvision/datasets.html

# In[ ]:


train_data = datasets.FashionMNIST(root='data', train=True,
                                   download=True, transform=transform)
test_data = datasets.FashionMNIST(root='data', train=False,
                                  download=True, transform=transform)


# # Creating Train and Test Loaders

# ### Now we're going to create the DataLoader to iterate over the data.
# 

# ```Python
#     num_workers = 0
# ```
# Creat workers so we can load the data in parallel

# ```Python
#     batch_size = 20
# ```
# The number of training examples in one forward/backward pass. The higher the batch size, the more memory space you'll need.

# In[ ]:


num_workers = 0 
batch_size = 20


# ```Python
#     train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, num_workers=num_workers)
# ```
# > PyTorch supplies another utility function called the DataLoader which acts as a data feeder for a Dataset object
# 
# 
# More about how to create a Dataset object here: [Link](https://towardsdatascience.com/building-efficient-custom-datasets-in-pytorch-2563b946fd9f)

# In[ ]:


train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, num_workers=num_workers)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, num_workers=num_workers)


# # Taking one image and plotting

# ```Python
#     dataiter = iter(train_loader)
#     images, labels = dataiter.next()
# ```
# Those two line of code will take the next batch. Our batch size is 20 so it will take the first 20 images.

# In[ ]:


dataiter = iter(train_loader)
images, labels = dataiter.next()


# In[ ]:


print(f"The number of images in each batch is equals: {len(images)}")


# ```Python
#     images = images.numpy()
#     images[0].shape
# ```
# In those two line we'll convert the tensor into a numpy array and display the shape this way we can see its behavior.

# In[ ]:


images = images.numpy()
images[0].shape


# The first thing we want to do when working with a dataset is to visualize the data in a meaningful way. In our case the the image is has (28 * 28 * 1) dimentions and we cannot plot that this way. So we have to squeeze the single-dimensional entries to be allowed to plot.

# ```Python
#     img = np.squeeze(images[0])
#     img.shape
# ```
# This code will remove the single dimensional entries. Take a look below

# In[ ]:


img = np.squeeze(images[0])
img.shape


# ### Here we'll plot the one of the figures to understand better our data

# In[ ]:


fig = plt.figure(figsize = (5,5)) 
ax = fig.add_subplot(111)
ax.imshow(img, cmap='gray')


# # Creating a simple Autoencoder model

# ### Image representation of a Autoencoder

# <center> <img src='https://hackernoon.com/hn-images/1*-5D-CBTusUnmsbA6VYdY3A.png' /> </center>

# **Autoencoders** (AE) are a family of neural networks for which the output try to represent the input. The **encoder** is the part of the NN that brings the data to a smaller dimension. This small representation is called **latent space**. The **Decoder** part take this encoded part and try to coverts it back to its original shape.

# In[ ]:


class Autoencoder(nn.Module):
    def __init__(self, dim):
        super(Autoencoder, self).__init__()
        self.encoder1 = nn.Linear(dim, 128)
        self.encoder2 = nn.Linear(128, 64)
        self.encoder3 = nn.Linear(64, 32)
        
        self.decoder1 = nn.Linear(32, 64)
        self.decoder2 = nn.Linear(64, 128)
        self.decoder3 = nn.Linear(128, dim)
        
    def forward(self, x):
        x = F.relu(self.encoder1(x))
        x = F.relu(self.encoder2(x))
        x = F.relu(self.encoder3(x))
        x = F.relu(self.decoder1(x))
        x = F.relu(self.decoder2(x))
        x = torch.sigmoid(self.decoder3(x))
        return x
    
dim = 28*28
model = Autoencoder(dim)
print(model)


# # Selecting a Criterion and Optimizer

# In[ ]:


criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)


# # Running the model

# In[ ]:



n_epochs = 2

for epoch in range(1, n_epochs+1):
    train_loss = 0.0
    
  
    for data in train_loader:
        images, _ = data
        images = images.view(images.size(0), -1)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, images)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()*images.size(0)
            
    train_loss = train_loss/len(train_loader)
    print('Epoch: {} \tTraining Loss: {:.6f}'.format(
        epoch, 
        train_loss
        ))


# # Plotting the ground-truth and Genereted numbers

# In[ ]:


dataiter = iter(test_loader)
images, labels = dataiter.next()

images_flatten = images.view(images.size(0), -1)
output = model(images_flatten)
images = images.numpy()

output = output.view(batch_size, 1, 28, 28)
output = output.detach().numpy()
fig, axes = plt.subplots(nrows=2, ncols=10, sharex=True, sharey=True, figsize=(25,4))

for images, row in zip([images, output], axes):
    for img, ax in zip(images, row):
        ax.imshow(np.squeeze(img), cmap='gray')
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)


# *This code is based on Udacity Autoencoder lesson.*

# In[ ]:




