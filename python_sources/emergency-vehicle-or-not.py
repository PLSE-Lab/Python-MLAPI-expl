#!/usr/bin/env python
# coding: utf-8

# In[ ]:


get_ipython().system('pip install torch_summary')


# In[ ]:


import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torchvision.models as models
import torch.optim as optim
from tqdm import tqdm_notebook
import torch.nn.functional as F
from tqdm import tqdm,tqdm_notebook
from torchsummary import summary
import torchvision
import random
from imageio import imread
import matplotlib.pyplot as plt
import warnings
import os
warnings.simplefilter('ignore')


# In[ ]:


device = torch.device("cuda" if torch.cuda.is_available() 
                                  else "cpu")
print(device)


# In[ ]:


DATA_PATH = '/kaggle/input/emergency-vehicle/datasets/'
train = os.path.join(DATA_PATH,'train')
test = os.path.join(DATA_PATH,'test')
images = os.path.join(DATA_PATH,'images')


# In[ ]:


train_dataloader = torchvision.datasets.ImageFolder(train,transform=torchvision.transforms.Compose([
    torchvision.transforms.ToTensor()
]))


# In[ ]:


test_dataloader = torchvision.datasets.ImageFolder(test,transform=torchvision.transforms.Compose([
    torchvision.transforms.ToTensor()
]))


# In[ ]:


fig=plt.figure(figsize=(12, 6))
for i in range(10):
    plt.subplot(2,5,i+1)
    file = random.choice([x for x in os.listdir(images)
               if os.path.isfile(os.path.join(images, x))])
    img = os.path.join(images,file)
    image = imread(img)
    plt.imshow(image)
plt.tight_layout()
plt.show()


# In[ ]:


train_imgs = torch.utils.data.DataLoader(train_dataloader,
                                         shuffle=True,
                                         batch_size=16)

test_loader = torch.utils.data.DataLoader(test_dataloader,
                                         shuffle=True,
                                         batch_size=1)


# In[ ]:


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3,16,5)
        self.conv2 = nn.Conv2d(16,16,5)
        self.drop = nn.Dropout2d(0.4)
        self.fc1 = nn.Linear(44944,128)
        self.fc2 = nn.Linear(128,10)
    
    def forward(self,x):
        x = F.relu(F.max_pool2d(self.conv1(x),2))
        x = F.relu(F.max_pool2d(self.conv2(x),2))
        x = x.view(-1,44944)
        x = F.relu(self.drop(self.fc1(x)))
        x = self.fc2(x)
        return F.sigmoid(x)


# In[ ]:


network = Net()
print(summary(network,(3,224,224)))


# In[ ]:


criterian = nn.CrossEntropyLoss()
optimizer = optim.Adam(network.parameters())
network.to(device)


# In[ ]:


train_loss = []
def train_model(epoch):
    network.train()
    for batch_idx,(data,target) in tqdm_notebook(enumerate(train_imgs)):
        optimizer.zero_grad()
        output = network.forward(data.cuda())
        loss = criterian(output,target.cuda())
        loss.backward()
        optimizer.step()
        train_loss.append(loss.item())
        if batch_idx%10 == 0:
            print(f'epoch: {epoch} , loss: {loss.item()}')


# In[ ]:


test_losses = []
def test():
    network.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            output = network.forward(data.to(device))
            test_loss += criterian(output, target.to(device)).item()
            pred = output.data.max(1, keepdim=True)[1]
            correct += pred.eq(target.to(device).data.view_as(pred)).sum()
        test_loss /= len(test_loader.dataset)
        print(f'Test loss for this iteration is: {test_loss}')
        test_losses.append(test_loss)


# In[ ]:


test()
epochs = 10
print('<------------------training starts--------------->')
for i in tqdm_notebook(range(epochs)):
    train_model(i)
    test()


# In[ ]:


y = range(len(train_loss))
tlr = [i*100 for i in range(epochs+1)]


# In[ ]:


plt.plot(y,train_loss,label= 'train_loss')
plt.plot(tlr,test_losses,'.r',label='test_loss')
plt.title('train loss and test loss')
plt.legend()
plt.show()

