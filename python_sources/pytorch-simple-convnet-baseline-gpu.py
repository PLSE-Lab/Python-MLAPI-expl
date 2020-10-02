#!/usr/bin/env python
# coding: utf-8

# # <center> PyTorch baseline for CIFAR-10
# **Based on [this](http://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html) tutorial.**

# In[2]:


get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
from tqdm import tqdm_notebook
import numpy as np
import pandas as pd
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F


# Currently GPU's are supported in Kernels for this version of PyTorch:

# In[2]:


torch.__version__


# Thus, we also need Variable. See changes [here](https://pytorch.org/2018/04/22/0_4_0-migration-guide.html).

# In[3]:


from torch.autograd import Variable


# In[4]:


use_gpu = torch.cuda.is_available()
use_gpu


# **Untar and load data**
# 
# There is a bug with Kaggle kernels: freshly added data is not seen, so I've added a new "Data Source" CIFAR-10-Python.[](http://)

# In[5]:


get_ipython().system('ls ../input/cifar10-python/')


# In[6]:


get_ipython().system('tar -zxvf ../input/cifar10-python/cifar-10-python.tar.gz')


# In[7]:


get_ipython().system('ls .')


# In[8]:


transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

trainset = torchvision.datasets.CIFAR10(root='.', train=True,
                                        download=False, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=32,
                                          shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='.', train=False,
                                       download=False, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=32,
                                         shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


# In[9]:


def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))


# get some random training images
dataiter = iter(trainloader)
images, labels = dataiter.next()

# show images
imshow(torchvision.utils.make_grid(images))
# print labels
print(' '.join('%5s' % classes[labels[j]] for j in range(32)))


# **Define net architecture.**

# In[10]:


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


net = Net()
if torch.cuda.is_available():
    net.cuda()


# **Define loss and optimizer. With GPU available, loss will also be caclulated on GPU.**

# In[11]:


import torch.optim as optim

criterion = nn.CrossEntropyLoss()
if use_gpu:
    criterion = criterion.cuda()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)


# **Train with mini-batches, 10 epochs.**

# In[12]:


# loop over the dataset multiple times
for epoch in tqdm_notebook(range(10)):  

    running_loss = 0.0
    for i, data in tqdm_notebook(enumerate(trainloader, 0)):
        # get the inputs
        inputs, labels = data

        if torch.cuda.is_available():
            # in versions of Torch < 0.4.0 we have to wrap these into torch.autograd.Variable as well
            inputs, labels = Variable(inputs).cuda(), Variable(labels).cuda()

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.data[0]
        if i % 2000 == 1999:    # print every 2000 mini-batches
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 2000))
            running_loss = 0.0

print('Finished Training')


# **Prediction for first mini-batch**

# In[13]:


dataiter = iter(testloader)
images, labels = dataiter.next()

# print images
imshow(torchvision.utils.make_grid(images))
print('GroundTruth: ', ' '.join('%5s' % classes[labels[j]] for j in range(32)))


# In[14]:


# in PyTorch 0.4.0 you won't need the Variable wrapper
outputs = net(Variable(images).cuda()) if use_gpu else net(Variable(images))


# In[15]:


_, predicted = torch.max(outputs.data, 1)

print('Predicted: ', ' '.join('%5s' % classes[predicted[j]]
                              for j in range(32)))


# **Now prediction for the whole test set.**

# In[16]:


all_pred = np.empty((0, 10), float)


# In[17]:


for data in tqdm_notebook(testloader):
    images, _ = data
    if use_gpu:
        images = images.cuda()
    outputs = net(Variable(images))
    curr_pred = F.softmax(outputs).data.cpu().numpy()
    all_pred = np.vstack([all_pred, curr_pred])


# In[18]:


all_pred.shape


# In[20]:


pd.DataFrame(all_pred, columns=classes).to_csv('baseline.csv', index_label='id')

