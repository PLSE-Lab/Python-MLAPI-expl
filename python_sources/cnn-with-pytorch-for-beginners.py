#!/usr/bin/env python
# coding: utf-8

# # CNN with PyTorch for Beginners

# 1. Load and normalizing the MNIST datasets
# 2. Define a Convolution Neural Network
# 3. Define a loss function
# 4. Train the network on the training data
# 5. Evaluate the model
# 6. Prediction and submition

# In[ ]:


# data loading and presentation
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

# PyTorch
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader


# ### 1. Load data

# In[ ]:


train_df = pd.read_csv('../input/train.csv')


# In[ ]:


train_df.head()


# In[ ]:


train_df.info()


# Our training dataset has 42000 rows(i.e. 42000 samples), each row has a 'label' column and 784 'pixel' columns(represent a $28 \times 28$ pixel picture). Let's see a few of them.

# In[ ]:


for i in range(8):
    plt.subplot(1, 8, i+1)
    plt.imshow(train_df.iloc[i, 1:].values.reshape(28, 28), cmap='gray')
    print(train_df.iloc[i, 0])


# In[ ]:


train_df.iloc[i, 1:].plot(kind='hist')


# The value of each pixel is 0~255, we'd better normalize it(devided by 255). `torch.utils.data.Dataset` is a convenient tools for data loading provided by PyTorch.
# 
# It is an abstract class representing a dataset. Your custom dataset should inherit Dataset and override the following methods:
# 
# - `__len__` so that len(dataset) returns the size of the dataset.
# - `__getitem__` to support the indexing such that dataset[i] can be used to get ith sample

# In[ ]:


class MNIST_dataset(Dataset):
    def __init__(self, df, rows=42000):
        self.imgnp = df.iloc[:rows, 1:].values
        self.labels = df.iloc[:rows, 0].values
        self.rows = rows
    
    def __len__(self):
        return self.rows
    
    def __getitem__(self, idx):
        image = torch.tensor(self.imgnp[idx], dtype=torch.float) / 255  # Normalize
        image = image.view(1, 28, 28)  # (channel, height, width)
        label = self.labels[idx]
        return (image, label)


# In[ ]:


trainloader = DataLoader(MNIST_dataset(train_df, 42000), batch_size=4, shuffle=True)


# In[ ]:


dataiter = iter(trainloader)


# In[ ]:


images, labels = dataiter.next()


# In[ ]:


images.size(), labels.size()


# In[ ]:


for i in range(4):
    plt.subplot(1, 4, i+1)
    plt.imshow(images[i, 0], cmap='gray')


# In[ ]:


labels


# ## 2. Define a Convolution Neural Network

# Our CNN model is as follow
# ![CNN model](https://pytorch.org/tutorials/_images/mnist.png)

# In[ ]:


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 4 * 4, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
        
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 4 * 4)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


# In[ ]:


net = Net()


# ## 3. Loss Function

# In[ ]:


criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.9)


# lr(learning rate) is a very important hyperparameter. Small lr will make the traning process very slow, but increase lr may also cause overshooting the global optimum and cause divergent. A good method is plot the Learning rate curve.

# ## 4. Train the network on the training data

# In[ ]:


running_loss_list = []
for epoch in range(2):
    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        inputs, labels = data
        optimizer.zero_grad()
        
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        # print statistics
        running_loss += loss.item()
        if i % 800 == 799:
            print('[%d, %5d] loss: %.3f' %
                 (epoch + 1, i + 1, running_loss / 800)
                 )
            running_loss_list.append(running_loss)
            running_loss = 0.0
print('Finished Training')


# In[ ]:


plt.plot(running_loss_list)


# The Learning rate curve is divergent, which means the learning rate is to large, we should decrease it.

# In[ ]:


net = Net()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

running_loss_list = []
for epoch in range(2):
    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        inputs, labels = data
        optimizer.zero_grad()
        
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        # print statistics
        running_loss += loss.item()
        if i % 200 == 199:
            print('[%d, %5d] loss: %.3f' %
                 (epoch + 1, i + 1, running_loss / 200)
                 )
            running_loss_list.append(running_loss)
            running_loss = 0.0
print('Finished Training')


# In[ ]:


plt.plot(running_loss_list)


# I't better, we will take this model as our final one.

# ## 5. Evaluate the model

# In[ ]:


correct = 0
total = 0
with torch.no_grad():
    for data in trainloader:
        images, labels = data
        outputs = net(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
print('Accuracy of the network on train images: ', correct/total)


# We achieve 98% accuracy on training data. It's not bad!

# ## 6. Prediction and Submition

# In[ ]:


test_df = pd.read_csv('../input/test.csv')


# In[ ]:


test_df.values.shape


# In[ ]:


test_tensor = torch.tensor(test_df.values, dtype=torch.float) / 255
test_tensor = test_tensor.view(-1, 1, 28, 28)


# In[ ]:


outputs = net(test_tensor)


# In[ ]:


_, predicted = torch.max(outputs, 1)


# In[ ]:


submit_df = pd.DataFrame({'ImageId': np.arange(1, 28001), 'Label': predicted.numpy()})


# In[ ]:


submit_df.to_csv('cnn.csv', index=False)


# ### That's all, thank's for advice!

# In[ ]:




