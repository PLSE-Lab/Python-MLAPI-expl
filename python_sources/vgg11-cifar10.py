#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


import seaborn as sns

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision.utils import make_grid

from PIL import Image

import os
import sys
import time
import math


# In[ ]:


transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
#     transforms.RandomAffine(0, shear=10, scale=(0.8,1.2)),
#     transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])


# In[ ]:


trainset = torchvision.datasets.CIFAR10(
    root='./data', train=True, download=True, transform=transform_train)

testset = torchvision.datasets.CIFAR10(
    root='./data', train=False, download=True, transform=transform_test)

classes = ('plane', 'car', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck')


# In[ ]:


trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=64, shuffle=True, num_workers=4)
testloader = torch.utils.data.DataLoader(
    testset, batch_size=64, shuffle=False, num_workers=4)


# In[ ]:


'''LeNet in PyTorch.'''
import torch.nn as nn
import torch.nn.functional as F

class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1   = nn.Linear(16*5*5, 120)
        self.fc2   = nn.Linear(120, 84)
        self.fc3   = nn.Linear(84, 10)

    def forward(self, x):
        out = F.relu(self.conv1(x))
        out = F.max_pool2d(out, 2)
        out = F.relu(self.conv2(out))
        out = F.max_pool2d(out, 2)
        out = out.view(out.size(0), -1)
        out = F.relu(self.fc1(out))
        out = F.relu(self.fc2(out))
        out = self.fc3(out)
        return out


# In[ ]:


class VGG11(nn.Module):
    def __init__(self):
        super(VGG11, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(256)
        self.conv4 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(256)
        self.conv5 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.bn5 = nn.BatchNorm2d(512)
        self.conv6 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.bn6 = nn.BatchNorm2d(512)
        self.conv7 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.bn7 = nn.BatchNorm2d(512)
        self.conv8 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.bn8 = nn.BatchNorm2d(512)
        self.avgPool1 = nn.AvgPool2d(kernel_size=1, stride=1)
        self.fc1 = nn.Linear(512, 10)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.max_pool2d(out,2)
        out = F.relu(self.bn2(self.conv2(out)))
        out = F.max_pool2d(out,2)
        out = F.relu(self.bn3(self.conv3(out)))
        out = F.relu(self.bn4(self.conv4(out)))
        out = F.max_pool2d(out,2)
        out = F.relu(self.bn5(self.conv5(out)))
        out = F.relu(self.bn6(self.conv6(out)))
        out = F.max_pool2d(out,2)
        out = F.relu(self.bn7(self.conv7(out)))
        out = F.relu(self.bn8(self.conv8(out)))
        out = F.max_pool2d(out,2)
        out = self.avgPool1(out)
        out = out.view(out.size(0), -1)
        out = self.fc1(out)
        return out


# In[ ]:


device = 'cuda' if torch.cuda.is_available() else 'cpu'
device


# In[ ]:


best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch


# In[ ]:


# net = LeNet()
net = VGG11()
net = net.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.1,
                      momentum=0.9, weight_decay=5e-4)


# In[ ]:


# get only one training image and pass it to the network
# im = transforms.ToPILImage()(torch.squeeze(next(iter(trainloader))[0], 0))
# im

sample = next(iter(trainloader))
smoketest_image = sample[0]
smoketest_label = sample[1]
print(smoketest_label)
net.train()
smoketest_image = smoketest_image.to(device)
smoketest_label = smoketest_label.to(device)
optimizer.zero_grad()
output = net(smoketest_image)
loss = criterion(output, smoketest_label)
loss.backward()
optimizer.step()


net.eval()
with torch.no_grad():
    output = net(smoketest_image)
    loss = criterion(output, smoketest_label)
    print("test loss", loss.item())
    _, predicted = output.max(1) # get the predicted label
    print("Print predicted label", predicted)
    


# In[ ]:


num_epochs = 30
train_running_loss= []
train_running_acc= []
test_running_loss=[]
test_running_acc=[]


# In[ ]:


# Training
def train(epoch):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0.0
    running_corrects = 0.0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
#         total += targets.size(0)
#         correct += predicted.eq(targets).sum().item()
        running_corrects += torch.sum(predicted == targets.data)
    #plot train loss
    train_running_loss.append(train_loss/len(trainset))
    train_running_acc.append(running_corrects.float()/len(trainset))
#     print("loss ", train_loss)


# In[ ]:


def test(epoch):
    global best_acc
    net.eval()
    test_loss = 0.0
    running_corrects = 0.0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
#             correct += predicted.eq(targets).sum().item()
            running_corrects += torch.sum(predicted == targets.data)
        # plot test loss
#         test_loss_plot.append(test_loss/len(testset))
        test_running_loss.append(test_loss/len(testset))
        test_running_acc.append(running_corrects/len(testset))
#         print("test loss", test_loss)


# In[ ]:


for epoch in range(start_epoch, start_epoch+num_epochs):
    train(epoch)
    test(epoch)


# In[ ]:


sns.set()
fig = plt.figure(1, figsize = (25,8))
ax1 = plt.subplot(1,2,1)
ax1.plot(range(num_epochs), train_running_loss, label="train")
ax1.plot(range(num_epochs), test_running_loss, label="test")
ax1.set_title("Loss curve")
plt.legend()
ax2 = plt.subplot(1,2,2)

ax2.plot(range(num_epochs), train_running_acc, label="train")
ax2.plot(range(num_epochs), test_running_acc, label="test")
ax2.set_title("accuracy curve")
plt.legend()
plt.show()

# plt.plot(range(num_epochs), train_loss_plot, label="train")
# plt.plot(range(num_epochs), test_loss_plot, label="test")


# In[ ]:


correct_count, all_count = 0, 0
for images,labels in testloader:
    for i in range(len(labels)):
        img = images[i].view(1, 3, 32, 32).to(device)
        with torch.no_grad():
            logps = net(img)

        ps = torch.exp(logps)
        probab = list(ps.cpu().numpy()[0])
        pred_label = probab.index(max(probab))
        true_label = labels.numpy()[i]
        if(true_label == pred_label):
          correct_count += 1
        all_count += 1

print("Number Of Images Tested =", all_count)
print("\nModel Accuracy =", (correct_count/all_count))


# # Development Area

# In[ ]:


raise ValueError ("check the code before running the following cells")


# In[ ]:


fc1_output = []
def hook(module, input, output):
    fc1_output.append(output)
net.fc1.register_forward_hook(hook)


# In[ ]:


a = torch.ones(5)
b = a.numpy()
type(b)


# In[ ]:


features = fc1_output[0]
# features = features.numpy()
features = features.cpu()

import numpy as np
from sklearn.manifold import TSNE
tsne = TSNE().fit_transform(features)
tx, ty = tsne[:,0], tsne[:,1]
tx = (tx-np.min(tx)) / (np.max(tx) - np.min(tx))
ty = (ty-np.min(ty)) / (np.max(ty) - np.min(ty))


# In[ ]:


import matplotlib.pyplot as plt
from PIL import Image
width = 4000
height = 3000
max_dim = 100
full_image = Image.new('RGB', (width, height))
for idx, x in enumerate(testloader):
    tile = Image.fromarray(np.uint8(x * 255))
    rs = max(1, tile.width / max_dim, tile.height / max_dim)
    tile = tile.resize((int(tile.width / rs),
                        int(tile.height / rs)),
                       Image.ANTIALIAS)
    full_image.paste(tile, (int((width-max_dim) * tx[idx]),
                            int((height-max_dim) * ty[idx])))


# In[ ]:


test = []
t1 = [[2,5],[1,2,3,4,6]]
test.append(t1)
test


# In[ ]:


test.

