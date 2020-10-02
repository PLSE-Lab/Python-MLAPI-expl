#!/usr/bin/env python
# coding: utf-8

# In[ ]:


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


# In[ ]:


get_ipython().system('pip install -i https://test.pypi.org/simple/ supportlib')
get_ipython().system('git clone https://github.com/davidtvs/pytorch-lr-finder')
import supportlib.gettingdata as getdata


# In[ ]:


print(os.listdir('/kaggle/input'))


# In[ ]:


import pandas as pd
import torch
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
from torchvision import transforms
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data.sampler import SubsetRandomSampler
import os
import cv2
import random
from sklearn.preprocessing import MultiLabelBinarizer
import torchvision.models as models
df = pd.read_csv('../input/train_v2.csv')


# In[ ]:


df.head(10)


# In[ ]:


tags = df['tags']
tags = tags.str.split()
mlb = MultiLabelBinarizer()
y_train = mlb.fit_transform(tags)
name = df['image_name']
name = np.asarray(name)


# In[ ]:


im_size = 256
img_dir = '../input/train-jpg'
batch_size = 16
epoch = 10
valid_size = 0.1
test_size = 0.2


# In[ ]:


import os
a = os.listdir('../input/train-tif-v2')


# In[ ]:


from fastai.vision import *


# In[ ]:


name = np.asarray(os.listdir(img_dir))


# In[ ]:


a = name[0]
a = a[:-3]


# In[ ]:


class Amazon_dataset(Dataset):
    def __init__(self,image_dir,y_train,name,transform = None):

        self.img_dir = image_dir
        self.y_train = y_train
        self.transform = transform
        self.id = name
    def __len__(self):
        return len(os.listdir(self.img_dir))
    def __getitem__(self,idx):
        print(self.id[idx])
        im_id =  self.id[idx] + '.jpg'
        img_name = os.path.join(self.img_dir, im_id)
        image = cv2.imread(img_name)
        if self.transform:
            image = self.transform(image)
        label = torch.from_numpy(self.y_train[idx])
        label = label.type(torch.cuda.FloatTensor)
        return image,label


# In[ ]:


# Data transform
transform = transforms.Compose([transforms.ToPILImage(),
                                transforms.Resize((im_size,im_size)),
                                transforms.RandomVerticalFlip(p=0.5),
                                transforms.ToTensor(),
                                transforms.Normalize([0.311, 0.340, 0.299], [0.167, 0.144, 0.138])
                    ])
inv_normalize = transforms.Normalize(
    mean=[-0.311/0.167, -0.340/0.144, -0.299/0.138],
    std=[1/0.167, 1/0.144, 1/0.138]
)

#Data laoder
amazon_data = Amazon_dataset(img_dir,y_train,name,transform)
testloader = DataLoader(amazon_data, batch_size=batch_size)


# In[ ]:


import numpy as np
data_len = len(amazon_data)
indices = list(range(data_len))
np.random.shuffle(indices)
split1 = int(np.floor(valid_size * data_len))
split2 = int(np.floor(test_size * data_len))
valid_idx , test_idx, train_idx = indices[:split1], indices[split1:split2] , indices[split2:] 
train_sampler = SubsetRandomSampler(train_idx)
valid_sampler = SubsetRandomSampler(valid_idx)
test_sampler = SubsetRandomSampler(test_idx)
train_loader = DataLoader(amazon_data, batch_size=batch_size , sampler=train_sampler)
valid_loader = DataLoader(amazon_data, batch_size=batch_size , sampler=valid_sampler)
test_loader = DataLoader(amazon_data, batch_size=batch_size , sampler=test_sampler)


# In[ ]:


def class_plot(n_figures , data , encoder ,inv_normalize):
    n_row = int(n_figures/3)
    fig,axes = plt.subplots(figsize=(14, 10), nrows = n_row, ncols=3)
    for ax in axes.flatten():
        a = random.randint(0,40479)
        image,label = data[a]
        label = label.cpu().numpy()
        n_classes = label.shape[0]
        label = np.reshape(label,(1,n_classes))
        label = encoder.inverse_transform(label)
        image = inv_normalize(image)
        image = image.numpy().transpose(1,2,0)
        im = ax.imshow(image)
        ax.set_title(label)
        ax.axis('off')
    plt.show()
        


# In[ ]:


encoder['1']


# In[ ]:


class_plot(12, amazon_data , mlb , inv_normalize)


# In[ ]:


class Classifier(nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()
        self.resnet = models.resnet34(pretrained = True)
        self.num_ftrs = self.resnet.fc.in_features
        self.l1 = nn.Linear(1000 , 512)
        self.l2 = nn.Linear(512,17)
    def forward(self, input):
        x = self.resnet(input)
        x = x.view(x.size(0),-1)
        x = F.relu(self.l1(x))
        x = F.sigmoid(self.l2(x))
        return x


# In[ ]:


get_ipython().system('pip install torchsummary')
from torchsummary import summary
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
classifier = Classifier().to(device)
#summary(classifier,(3,256,256))


# In[ ]:


optimizer = optim.Adam(classifier.parameters(), lr=0.01)


# In[ ]:


from lr_finder import LRFinder
criterion = nn.BCELoss()


# In[ ]:


lr_finder = LRFinder(classifier, optimizer, criterion, device="cuda")
lr_finder.range_test(train_loader, end_lr=100, num_iter=1000)
lr_finder.plot()


# In[ ]:


## from sklearn.metrics import f1_score
losses1 = []
f1score1 = []
dataloader = train_loader
lr = 0.0001
for i in range(epoch):
    y_pred = []
    y_true = []
    lr = lr/2
    optimizer = optim.Adam(classifier.parameters(), lr=lr)
    for batch_idx, (data, target) in enumerate(dataloader):
        data, target = Variable(data), Variable(target)
        data = data.type(torch.cuda.FloatTensor)
        target = target.type(torch.cuda.FloatTensor)
        optimizer.zero_grad()
        output = classifier(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        output = output.cpu().detach().numpy()
        y_pred.append(output)
        target = target.cpu().numpy()
        y_true.append(target)
        if batch_idx % 100 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(i, batch_idx * len(data), len(dataloader.dataset),100. * batch_idx / len(dataloader), loss.item()))
    #y_pred = get_pred(y_pred)
    acc,f_score = f2_score(y_true,y_pred)
    losses1.append(loss.item())
    f1score1.append(f_score.item())
    print('Train Epoch: {} \tf1_score: {:.6f} Accuracy:{}'.format(i , f_score.item() ,acc.item()))


# In[ ]:


from fastai import metrics


# In[ ]:


def predict(image , encoder ,threshold = 0.65):
    image = np.reshape(image,(1,3,256,256))
    image = image.type(torch.cuda.FloatTensor)
    output = classifier(image)
    output = output.cpu().detach()
    output = torch.ge(output, threshold).float().numpy()
    return encoder.inverse_transform(output)


# In[ ]:


predict(image , mlb)


# In[ ]:


pred = []
for batch_idx, (data) in enumerate(testloader):
    data = Variable(data)
    data = data.type(torch.cuda.FloatTensor)
    output = classifier(data)
    output = output.cpu().detach()
    y_pred = torch.ge(output, 0.15).float().numpy()
    pred.append(y_pred)


# In[ ]:


def f2_score(y_true, y_pred, threshold=0.60):
    leng = len(y_true)
    y_pred = y_pred[0:leng-1]
    y_true = y_true[0:leng-1]
    y_pred = np.asarray(y_pred)
    y_true = np.asarray(y_true)
    y_pred = y_pred.astype(float)
    y_true = y_true.astype(float)
    y_pred = torch.from_numpy(y_pred)
    y_true = torch.from_numpy(y_true)
    return metrics.accuracy_thresh(y_pred,y_true,threshold) , metrics.fbeta(y_pred,y_true,threshold)
acc,f_score = f2_score(y_true,y_pred)
print(acc)
print(f_score)

