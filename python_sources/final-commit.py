#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from __future__ import print_function

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from skimage import data
from skimage.transform import resize

import torch
from torch import utils
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
import torchvision
from torchvision import datasets, transforms

import math
import random

import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


batch_size = 64

test_batch_size = 512

image_size = 40

learning_rate = 0.003

epoches_num = 40


# In[ ]:


def dataset_(path, id_dict):
    print("\nStarted dataset ", path, "\n", sep = '')
    
    data = np.load(path)
    
    # resize
    for i in range(len(data)):
        data[i][0] = resize(data[i][0], (image_size, image_size))
        
    # dealing with labels   
    for i in range(len(data)):
        data[i][1] = id_dict[data[i][1]]
        
    # dataset      
    x = [data[i][0] for i in range(len(data))]
    data_x = torch.stack([torch.from_numpy(i).type(torch.FloatTensor) for i in x])

    y = [data[i][1] for i in range(len(data))]
    data_y = torch.stack([torch.tensor(i) for i in y])
        
    dataset_data = utils.data.TensorDataset(data_x,data_y)
    
    # free some space
    del x
    del y
    del data
    del data_x
    del data_y
    
    print("\nFinished dataset ", path, "\n", sep = '')
    
    return dataset_data


# In[ ]:


data_id = np.load("../input/train-1.npy")

id_dict = {}

id_curr = 0
for i in range(len(data_id)):
    if (not(data_id[i][1] in id_dict)):
        id_dict[data_id[i][1]] = id_curr
        id_curr += 1
        
del data_id


# In[ ]:


data1 = dataset_("../input/train-1.npy", id_dict)
data2 = dataset_("../input/train-2.npy", id_dict)
data3 = dataset_("../input/train-3.npy", id_dict)
data4 = dataset_("../input/train-4.npy", id_dict)


# In[ ]:


whole_data = torch.utils.data.ConcatDataset((data1, data2, data3, data4))

print(len(whole_data))


# In[ ]:


train_dataset, test_dataset = torch.utils.data.random_split(whole_data, (330000, 2987))

print(len(train_dataset))
print(len(test_dataset))


# In[ ]:


train_loader = torch.utils.data.DataLoader(train_dataset, batch_size, shuffle = True)

test_loader = torch.utils.data.DataLoader(test_dataset, test_batch_size, shuffle = True)


# In[ ]:


class Net(nn.Module):    
    def __init__(self):
        super(Net, self).__init__()
          
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
          
        self.classifier = nn.Sequential(
            nn.Dropout(p = 0.5),
            nn.Linear(image_size * image_size * 4, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(p = 0.4),
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(p = 0.4),
            nn.Linear(512, 1000)
        )
          
        for m in self.features.children():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
        
        for m in self.classifier.children():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
                

    def forward(self, x):
        x = x.view(x.shape[0], 1, x.shape[1], x.shape[2])
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        
        return x  


# In[ ]:


torch.manual_seed(1200)

model = Net()

optimizer = optim.Adam(model.parameters(), lr=learning_rate)

criterion = nn.CrossEntropyLoss()

exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.7)

if torch.cuda.is_available():
    model = model.cuda()
    criterion = criterion.cuda()


# In[ ]:


def train(epoch):
    model.train()
    exp_lr_scheduler.step()
    
    batches = len(train_loader)
    percent = {int(batches * 1 / 5) : 20,
               int(batches * 2 / 5) : 40, 
               int(batches * 3 / 5) : 60, 
               int(batches * 4 / 5) : 80,
               batches - 1 : 100}
    for batch_idx, (data, target) in enumerate(train_loader):
        if (batch_idx in percent):
            print("{}% ready".format(percent[batch_idx]))
        
        if torch.cuda.is_available():
            data = data.cuda()
            target = target.cuda()
        
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        
        loss.backward()
        optimizer.step()
    
    print("Training finished\n")


# In[ ]:


def test_model(data_loader, title):
    print("Testing", title)
    model.eval()
    with torch.no_grad():
        correct = 0
    
        for data, target in data_loader:
            if torch.cuda.is_available():
                data = data.cuda()
                target = target.cuda()
        
            output = model(data)
            
            pred = output.data.max(1, keepdim=True)[1]
            correct += pred.eq(target.data.view_as(pred)).cpu().sum()
    
    print('Accuracy: {}/{} ({:.3f}%)\n'.format(correct, len(data_loader.dataset),
        100. * correct / len(data_loader.dataset)))


# In[ ]:


for epoch in range(epoches_num):
    print("Epoch number", epoch + 1)
    train(epoch)
    test_model(train_loader, "train set")
    test_model(test_loader, "test set")


# In[ ]:


def prediciton(data_loader):
    model.eval()
    test_pred = torch.LongTensor()
    
    for data, target in data_loader:
        if torch.cuda.is_available():
            data = data.cuda()
            
        output = model(data)
        
        pred = output.cpu().data.max(1, keepdim=True)[1]
        test_pred = torch.cat((test_pred, pred), dim=0)
        
    return test_pred


# In[ ]:


test = np.load("../input/test.npy")

for i in range(len(test)):
        test[i] = resize(test[i], (image_size, image_size))
        
x = [test[i] for i in range(len(test))]
data_x = torch.stack([torch.from_numpy(i).type(torch.FloatTensor) for i in x])

y = [0 for i in range(len(test))]
data_y = torch.stack([torch.tensor(i) for i in y])

dataset_test = utils.data.TensorDataset(data_x,data_y)

final_test_loader = torch.utils.data.DataLoader(dataset_test, test_batch_size, shuffle = False)


# In[ ]:


test_pred = prediciton(final_test_loader)


# In[ ]:


reverse = {}

for key in id_dict:
    reverse[id_dict[key]] = key


# In[ ]:


for i in range(len(test_pred)):
    test_pred[i] = reverse[test_pred[i].item()]


# In[ ]:


out_df = pd.DataFrame(np.c_[np.arange(1, len(test_pred)+1)[:,None], test_pred.numpy()], 
                      columns=['Id', 'Category'])


# In[ ]:


out_df.to_csv('submission.csv', index=False)

