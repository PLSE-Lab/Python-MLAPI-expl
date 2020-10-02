#!/usr/bin/env python
# coding: utf-8

# In[42]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# ## Import Modules

# In[43]:


from torch.utils.data.dataset import Dataset
import torch
from torchvision import transforms
from PIL import Image


# ## Load Data

# In[44]:


class CustomDatasetFromCSV(Dataset):
    def __init__(self, csv_path, height, width, transforms=None,train=False):
        self.data = pd.read_csv(csv_path)
        self.train = train
        if train:
            self.labels = np.asarray(self.data.iloc[:, 0])
        self.height = height
        self.width = width
        self.transforms = transforms

    def __getitem__(self, index):
        
        # Read each 784 pixels and reshape the 1D array ([784]) to 2D array ([28,28]) 
        if self.train:
            single_image_label = self.labels[index]
            img_as_np = np.asarray(self.data.iloc[index][1:]).reshape(28,28).astype('uint8')
        else:
            img_as_np = np.asarray(self.data.iloc[index][0:]).reshape(28,28).astype('uint8')
        # Convert image from numpy array to PIL image, mode 'L' is for grayscale
        img_as_img = Image.fromarray(img_as_np)
        img_as_img = img_as_img.convert('L')
        # Transform image to tensor
        if self.transforms is not None:
            img_as_tensor = self.transforms(img_as_img)
        # Return image and the label
        if self.train:
            return (img_as_tensor, single_image_label)
        else:
            return img_as_tensor
        

    def __len__(self):
        return len(self.data.index)
        


transformations = transforms.Compose([transforms.ToTensor()])
train_data = CustomDatasetFromCSV('../input/train.csv', 28, 28, transformations, True)
test_data  = CustomDatasetFromCSV('../input/test.csv', 28, 28, transformations, False)


# ## Data Loaders

# In[45]:


# number of subprocesses to use for data loading
num_workers = 0
# how many samples per batch to load
batch_size = 20
# prepare data loaders
train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size,
    num_workers=num_workers)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, 
    num_workers=num_workers)


# ## Visualize a Batch of Training Data

# In[46]:


import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
    
# obtain one batch of training images
dataiter = iter(train_loader)
images, labels = dataiter.next()
images = images.numpy()

# plot the images in the batch, along with the corresponding labels
fig = plt.figure(figsize=(25, 4))
for idx in np.arange(20):
    ax = fig.add_subplot(2, 20/2, idx+1, xticks=[], yticks=[])
    ax.imshow(np.squeeze(images[idx]), cmap='gray')
    # print out the correct label for each image
    # .item() gets the value contained in a Tensor
    ax.set_title(str(labels[idx].item()))


# ## View an Image in More Detail

# In[47]:


img = np.squeeze(images[1])

fig = plt.figure(figsize = (12,12)) 
ax = fig.add_subplot(111)
ax.imshow(img, cmap='gray')
width, height = img.shape
thresh = img.max()/2.5
for x in range(width):
    for y in range(height):
        val = round(img[x][y],2) if img[x][y] !=0 else 0
        ax.annotate(str(val), xy=(y,x),
                    horizontalalignment='center',
                    verticalalignment='center',
                    color='white' if img[x][y]<thresh else 'black')


# ## Define the Network Architecture
# The architecture will be responsible for seeing as input a 784-dim Tensor of pixel values for each image, and producing a Tensor of length 10 (our number of classes) that indicates the class scores for an input image. This particular example uses two hidden layers and dropout to avoid overfitting.

# In[48]:


import torch.nn as nn
import torch.nn.functional as F
#import torch.optim as optim

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # linear layer (784 -> 1 hidden node)
        hidden_1 = 512
        hidden_2 = 512
        
        self.fc1 = nn.Linear(28 * 28, hidden_1)
        self.fc2 = nn.Linear(hidden_1, hidden_2)
        self.fc3 = nn.Linear(hidden_2,10)
        
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        # flatten image input
        x = x.view(-1, 28 * 28)
        # add hidden layer, with relu activation function
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x

# initialize the NN
model = Net()
print(model)


# 
# ## Specify Loss Function and Optimizer
# 
# It's recommended to use cross-entropy loss for classification. PyTorch's cross entropy function applies a softmax funtion to the output layer and then calculates the log loss.
# 

# In[49]:


criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr = 0.01)


# ## Train the Network
# 
# The steps for training/learning from a batch of data are described in the comments below:
# 
#     - Clear the gradients of all optimized variables
#     - Forward pass: compute predicted outputs by passing inputs to the model
#     - Calculate the loss
#     - Backward pass: compute gradient of the loss with respect to model parameters
#     - Perform a single optimization step (parameter update)
#     - Update average training loss
# 

# In[50]:


n_epochs = 10  # suggest training between 20-50 epochs

model.train() # prep model for training

for epoch in range(n_epochs):
    # monitor training loss
    train_loss = 0.0
    
    ###################
    # train the model #
    ###################
    for data, target in train_loader:
        # clear the gradients of all optimized variables
        optimizer.zero_grad()
        # forward pass: compute predicted outputs by passing inputs to the model
        output = model(data)
        # calculate the loss
        loss = criterion(output, target)
        # backward pass: compute gradient of the loss with respect to model parameters
        loss.backward()
        # perform a single optimization step (parameter update)
        optimizer.step()
        # update running training loss
        train_loss += loss.item()*data.size(0)
        
    # print training statistics 
    # calculate average loss over an epoch
    train_loss = train_loss/len(train_loader.dataset)

    print('Epoch: {} \tTraining Loss: {:.6f}'.format(
        epoch+1, 
        train_loss
        ))


# 
# ## Test the Trained Network
# 
# Finally, we test our best model on previously unseen test data and evaluate it's performance. Testing on unseen data is a good way to check that our model generalizes well. It may also be useful to be granular in this analysis and take a look at how this model performs on each class as well as looking at its overall loss and accuracy.
# 
# **model.eval()**
# 
# model.eval() will set all the layers in your model to evaluation mode. This affects layers like dropout layers that turn "off" nodes during training with some probability, but should allow every node to be "on" for evaluation!
# 

# In[51]:




model.eval() # prep model for *evaluation*


# 
# ## Visualize Sample Test Results
# 
# This cell displays test images and their labels in this format: predicted (ground-truth). The text will be green for accurately classified examples and red for incorrect predictions.
# 

# In[54]:


# obtain one batch of test images
dataiter = iter(test_loader)
images = dataiter.next()

# get sample outputs
output = model(images)
# convert output probabilities to predicted class
_, preds = torch.max(output, 1)
# prep images for display
images = images.numpy()

# plot the images in the batch, along with predicted and true labels
fig = plt.figure(figsize=(25, 4))
for idx in np.arange(20):
    ax = fig.add_subplot(2, 20/2, idx+1, xticks=[], yticks=[])
    ax.imshow(np.squeeze(images[idx]), cmap='gray')
    ax.set_title("Pred:"+str(preds[idx].item()))


# In[ ]:




