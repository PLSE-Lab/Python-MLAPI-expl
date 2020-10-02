#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[ ]:



import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from IPython.display import Image as IM
import os
import zipfile
import pandas as pd
import numpy as np
import os
import time
import torch
from torchvision import datasets,transforms
from torch.utils.data import DataLoader,Dataset
from PIL import Image


# In[ ]:


train_csv = pd.read_csv('/kaggle/input/aerial-cactus-identification/train.csv')
train_csv


# In[ ]:


with zipfile.ZipFile('/kaggle/input/aerial-cactus-identification/train.zip','r') as z:
    z.extractall('/kaggle/output/kaggle/working')


# In[ ]:


with zipfile.ZipFile('/kaggle/input/aerial-cactus-identification/test.zip','r') as z:
    z.extractall('/kaggle/output/kaggle/working')


# In[ ]:


TRAIN_IMG_PATH = "/kaggle/output/kaggle/working/train"
TEST_IMG_PATH = '/kaggle/output/kaggle/working/test'
LABELS_CSV_PATH = '/kaggle/input/aerial-cactus-identification/train.csv'
SAMPLE_SUB_PATH = "/kaggle/input/aerial-cactus-identification/sample_submission.csv"


# In[ ]:


class CactusDataset(Dataset):
    def __init__(self,img_dir,dataframe,transform=None):
        self.labels_frame = dataframe
        self.img_dir = img_dir
        self.transform = transform
    def __len__(self):
        return len(self.labels_frame)
    def __getitem__(self,idx):
        image = Image.open(os.path.join(self.img_dir,self.labels_frame.id[idx]))
        label = self.labels_frame.has_cactus[idx]
        if self.transform:
            image = self.transform(image)
        return [image,label]
# train_transforms = transforms.Compose([transforms.RandomResizedCrop(224),
#                                        transforms.RandomHorizontalFlip(0.5),
#                                        transforms.ToTensor(),
#                                        transforms.Normalize([0.485, 0.456, 0.406], 
#                                                             [0.229, 0.224, 0.225])])
# test_transforms = transforms.Compose([transforms.Resize(256),
#                                       transforms.CenterCrop(224),
#                                       transforms.ToTensor(),
#                                       transforms.Normalize([0.485, 0.456, 0.406], 
#                                                            [0.229, 0.224, 0.225])])
train_transforms = transforms.Compose([transforms.RandomHorizontalFlip(0.5),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406], 
                                                            [0.229, 0.224, 0.225])])
test_transforms = transforms.Compose([
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406], 
                                                           [0.229, 0.224, 0.225])])


# In[ ]:


dframe = pd.read_csv(LABELS_CSV_PATH)
cut = int(len(dframe)*0.9)
train, test = np.split(dframe, [cut], axis=0)
test = test.reset_index(drop=True)

train_ds = CactusDataset(TRAIN_IMG_PATH, train, train_transforms)
test_ds = CactusDataset(TRAIN_IMG_PATH, test, test_transforms)
datasets = {"train": train_ds, "val": test_ds}
trainloader = DataLoader(train_ds, batch_size=32,
                        shuffle=True, num_workers=0)

testloader = DataLoader(test_ds, batch_size=32,
                        shuffle=True, num_workers=0)


# In[ ]:


train


# In[ ]:


for i in testloader:
    print(i[0].shape)
    break


# In[ ]:


class Net(nn.Module):
    def __init__(self):
        super(Net,self).__init__()
        self.conv1 = nn.Conv2d(3,64,3,1,padding=1)
        self.conv2 = nn.Conv2d(64,64,3,1,padding=1)
        self.conv3 = nn.Conv2d(64,128,3,1,padding=1)
        self.conv4 = nn.Conv2d(128,128,3,1,padding=1)
        self.conv5 = nn.Conv2d(128,256,3,1,padding=1)
        self.conv6 = nn.Conv2d(256,256,3,1,padding=1)
        self.conv7 = nn.Conv2d(256,256,3,1,padding=1)
        self.fc1 = nn.Linear(256*4*4,2048)
#         self.bn1 = nn.BatchNorm1d(num_features=2048)
        self.fc2 = nn.Linear(2048,1024)
#         self.bn2 = nn.BatchNorm1d(num_features=1024)
        self.fc3 = nn.Linear(1024,2)
    def forward(self,x):
        
        x = self.conv1(x) 
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x,2,2) #112 
        x = self.conv3(x) 
        x = F.relu(x)
        x = self.conv4(x)
        x = F.relu(x)
        x = F.max_pool2d(x,2,2) #56
        x = self.conv5(x) 
        x = F.relu(x)
        x = self.conv6(x) 
        x = F.relu(x)
        x = self.conv7(x)
        x = F.relu(x)
        x = F.max_pool2d(x,2,2) #28
        x = self.fc1(x.view(-1,256*4*4))
        x = F.relu(x)
        x = F.dropout(x,p=0.5)
#         x = self.bn1(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = F.dropout(x,p=0.5)
#         x = self.bn2(x)
        x = self.fc3(x)
        return F.log_softmax(x, dim=1)
        
        


# In[ ]:


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
lr = 0.003
momentum  = 0.5
model = Net().to(device)
optimizer = torch.optim.SGD(model.parameters(),lr=lr,momentum=momentum)


# In[ ]:


def train(model ,device, train_loader,optimizer,epoch):
    model.train()
    for idx,(data, target) in enumerate(train_loader):
        data, target = data.to(device),target.to(device)
        data = data.view(-1,3,32,32)
        pred = model(data)
        loss = F.nll_loss(pred, target)
            
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if idx % 100 == 0:
            print("Train Epoch: {}, iteration: {}, Loss: {}".format(
                epoch, idx, loss.item()))


# In[ ]:


def test(model, device, test_loader):
    model.eval()
    total_loss = 0.
    correct = 0.
    with torch.no_grad():
        for idx, (data, target) in enumerate(test_loader):
            data, target = data.to(device), target.to(device)
            data = data.view(-1,3,32,32)
            output = model(data) # batch_size * 10
            total_loss += F.nll_loss(output, target, reduction="sum").item() 
            pred = output.argmax(dim=1) # batch_size * 1
            correct += pred.eq(target.view_as(pred)).sum().item()
            

    total_loss /= len(test_loader.dataset)
    acc = correct/len(test_loader.dataset) * 100.
    
    print("Test loss: {}, Accuracy: {}".format(total_loss, acc))
    return acc


# In[ ]:


get_ipython().run_cell_magic('time', '', 'num_epoch = 200\nfor epoch in range(num_epoch):\n    train(model,device,trainloader,optimizer,epoch)\n    acc = test(model,device,testloader)\n    if acc>=99.8 and epoch>150:\n        print(\'acc={}\'.format(acc),\'End...\')\n        break\n\n        \n# torch.save(model.state_dict,"cactus.pt")')


# In[ ]:


sub_df = pd.read_csv(SAMPLE_SUB_PATH)


# In[ ]:


sub_df.head()


# In[ ]:


sub_ds = CactusDataset(TEST_IMG_PATH, sub_df, test_transforms)


# In[ ]:


sub_transforms = transforms.Compose([
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406], 
                                                           [0.229, 0.224, 0.225])])

subloader = DataLoader(sub_ds, batch_size=1,
                        shuffle=False, num_workers=0)


# In[ ]:


# test_data = pd.read_csv('/kaggle/input/aerial-cactus-identification/sample_submission.csv')
# list =[]
# for i in range(len(test_data)):
# #     print(test_data.iloc[i][0])
#     image = Image.open(os.path.join('/kaggle/output/kaggle/working/test',test_data.iloc[i][0]))
#     list.append(np.array(image)/255)
    
# test_tensor = torch.Tensor(list)
# test_tensor = test_tensor.view(-1,3,32,32)
# loader = torch.utils.data.DataLoader(
#     test_tensor,
#     batch_size=batch_size,shuffle=False,
#     num_workers=1,pin_memory=True)
result = np.array([])
with torch.no_grad():
    for i in subloader:
#         print(i[0])
#         break
        v = model(i[0].to(device)).to('cpu').numpy().argmax(axis=1)
#         print(v)
        result = np.append(result,v)
sub_df.has_cactus = result
sub_df.to_csv('submission.csv',index=False)


# In[ ]:


len(subloader)


# In[ ]:


sub_df


# In[ ]:




