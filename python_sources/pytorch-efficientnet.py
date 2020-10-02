#!/usr/bin/env python
# coding: utf-8

# Example of using EfficientNet model in PyTorch.

# In[ ]:


import numpy as np
import pandas as pd
import os
import matplotlib.image as mpimg

import torch
import torch.nn as nn
import torch.optim as optim 

import torchvision
from torch.utils.data import DataLoader, Dataset
import torch.utils.data as utils
from torchvision import transforms

import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

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


data_transf = transforms.Compose([transforms.ToPILImage(), transforms.ToTensor()])
train_data = ImageData(df = labels, data_dir = train_dir, transform = data_transf)
train_loader = DataLoader(dataset = train_data, batch_size = 64)


# In[ ]:


get_ipython().system('pip install efficientnet_pytorch')


# In[ ]:


from efficientnet_pytorch import EfficientNet
model = EfficientNet.from_name('efficientnet-b1')


# In[ ]:


# Unfreeze model weights
for param in model.parameters():
    param.requires_grad = True


# In[ ]:


num_ftrs = model._fc.in_features
model._fc = nn.Linear(num_ftrs, 1)


# In[ ]:


model = model.to('cuda')


# In[ ]:


optimizer = optim.Adam(model.parameters())
loss_func = nn.BCELoss()


# In[ ]:


get_ipython().run_cell_magic('time', '', "# Train model\nloss_log = []\n\nfor epoch in range(5):    \n    model.train()    \n    for ii, (data, target) in enumerate(train_loader):\n        data, target = data.cuda(), target.cuda()\n        target = target.float()                \n\n        optimizer.zero_grad()\n        output = model(data)                \n    \n        m = nn.Sigmoid()\n        loss = loss_func(m(output), target)\n        loss.backward()\n\n        optimizer.step()  \n        \n        if ii % 1000 == 0:\n            loss_log.append(loss.item())\n       \n    print('Epoch: {} - Loss: {:.6f}'.format(epoch + 1, loss.item()))")


# In[ ]:


# plt.figure(figsize=(10,8))
# plt.plot(loss_log)


# In[ ]:


submit = pd.read_csv('../input/sample_submission.csv')
test_data = ImageData(df = submit, data_dir = test_dir, transform = data_transf)
test_loader = DataLoader(dataset = test_data, shuffle=False)


# In[ ]:


get_ipython().run_cell_magic('time', '', "predict = []\nmodel.eval()\nfor i, (data, _) in enumerate(test_loader):\n    data = data.cuda()\n    output = model(data)    \n\n    pred = torch.sigmoid(output)\n    predicted_vals = pred > 0.5\n    predict.append(int(predicted_vals))\n    \nsubmit['has_cactus'] = predict\nsubmit.to_csv('submission.csv', index=False)")


# In[ ]:


submit.head()

