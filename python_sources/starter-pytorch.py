#!/usr/bin/env python
# coding: utf-8

# Easy example of using CNN in PyTorch

# In[ ]:


import numpy as np
import pandas as pd
import os
import matplotlib.image as mpimg

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms

import warnings
warnings.filterwarnings("ignore")


# In[ ]:


data_dir = '../input'
train_dir = data_dir + '/train/train/'
test_dir = data_dir + '/test/test/'


# In[ ]:


labels = pd.read_csv("../input/train.csv")
labels.head()


# In[ ]:


class ImageData(Dataset):
    def __init__(self, df, data_dir, transform):
        super().__init__()
        self.df = df
        self.data_dir = data_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, index):       
        img_name = self.df.id[index]
        label = self.df.has_cactus[index]
        
        img_path = os.path.join(self.data_dir, img_name)
        image = mpimg.imread(img_path)
        image = self.transform(image)
        return image, label


# In[ ]:


epochs = 25
batch_size = 20
device = torch.device('cuda:0')


# In[ ]:


data_transf = transforms.Compose([transforms.ToPILImage(), transforms.ToTensor()])
train_data = ImageData(df = labels, data_dir = train_dir, transform = data_transf)
train_loader = DataLoader(dataset = train_data, batch_size = batch_size)


# In[ ]:


class Network(nn.Module): 
    def __init__(self):
        super(Network, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=10, kernel_size=3)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=3)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(720, 1024)
        self.fc2 = nn.Linear(1024, 2)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(x.shape[0],-1)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return x


# In[ ]:


net = Network().to(device)
optimizer = optim.Adam(net.parameters(), lr=0.001)
loss_func = nn.CrossEntropyLoss()


# In[ ]:


get_ipython().run_cell_magic('time', '', "# Train model\nfor epoch in range(epochs):\n    for i, (images, labels) in enumerate(train_loader):\n        images = images.to(device)\n        labels = labels.to(device)\n        \n        # Forward\n        outputs = net(images)\n        loss = loss_func(outputs, labels)\n        \n        # Backward and optimize\n        optimizer.zero_grad()\n        loss.backward()\n        optimizer.step()\n        \n        if (i+1) % 500 == 0:\n            print ('Epoch [{}/{}], Loss: {:.4f}'.format(epoch+1, epochs, loss.item()))")


# In[ ]:


submit = pd.read_csv('../input/sample_submission.csv')
test_data = ImageData(df = submit, data_dir = test_dir, transform = data_transf)
test_loader = DataLoader(dataset = test_data, shuffle=False)


# In[ ]:


predict = []
for batch_i, (data, target) in enumerate(test_loader):
    data, target = data.to(device), target.to(device)
    output = net(data)
    
    _, pred = torch.max(output.data, 1)
    predict.append(int(pred))
    
submit['has_cactus'] = predict
submit.to_csv('submission.csv', index=False)


# In[ ]:


submit.head()

