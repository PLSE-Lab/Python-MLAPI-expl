#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import tqdm
import time
import torchvision
import torch.nn as nn
from tqdm import tqdm_notebook as tqdm

from PIL import Image, ImageFile
from torch.utils.data import Dataset
import torch
import torch.optim as optim
from torchvision import transforms as T
from torch.optim import lr_scheduler
import os

print(os.listdir("../input"))
device = torch.device("cuda:0")


# In[ ]:


class RetinopathyDatasetTrain(Dataset):

    def __init__(self, csv_file , transforms=None):

        self.data = pd.read_csv(csv_file)
        self.transform = transforms

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_name = os.path.join('../input/train_images', self.data.loc[idx, 'id_code'] 
                                + '.png')
        data = Image.open(img_name)
        if self.transform:
            data = self.transform(data)
            
        label =self.data.loc[idx, 'diagnosis']
        return data,label


# In[ ]:


transform = T.Compose([T.Resize(32),
                      T.CenterCrop(32),
                      T.ToTensor(),
                      T.Normalize(mean=[0.5,0.5,0.5],std = [0.5,0.5,0.5])])



train_dataset = RetinopathyDatasetTrain(csv_file='../input/train.csv',transforms=transform )


# In[ ]:


img,label= train_dataset[9]
print(img.size(),label)
   


# In[ ]:


class RetinopathyDatasetTest(Dataset):

    def __init__(self, csv_file , transforms=None):

        self.data = pd.read_csv(csv_file)
        self.transform = transforms

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_name = os.path.join('../input/test_images', self.data.loc[idx, 'id_code'] 
                                + '.png')
        data = Image.open(img_name)
        if self.transform:
            data = self.transform(data)
            
        return data


# In[ ]:


test_dataset = RetinopathyDatasetTest(csv_file='../input/test.csv',transforms=transform )


# In[ ]:


imgs= test_dataset[0]
print(imgs.size())


# In[ ]:


import math

import torch
import torch.nn as nn
import torch.nn.functional as F


# In[ ]:


class Bottleneck(nn.Module):
    def __init__(self, in_planes, growth_rate):
        super(Bottleneck, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, 4*growth_rate, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(4*growth_rate)
        self.conv2 = nn.Conv2d(4*growth_rate, growth_rate, kernel_size=3, padding=1, bias=False)

    def forward(self, x):
        out = self.conv1(F.relu(self.bn1(x)))
        out = self.conv2(F.relu(self.bn2(out)))
        out = torch.cat([out,x], 1)
        return out


class Transition(nn.Module):
    def __init__(self, in_planes, out_planes):
        super(Transition, self).__init__()
        self.bn = nn.BatchNorm2d(in_planes)
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=1, bias=False)

    def forward(self, x):
        out = self.conv(F.relu(self.bn(x)))
        out = F.avg_pool2d(out, 2)
        return out


class DenseNet(nn.Module):
    def __init__(self, block, nblocks, growth_rate=12, reduction=0.5, num_classes=5):
        super(DenseNet, self).__init__()
        self.growth_rate = growth_rate

        num_planes = 2*growth_rate
        self.conv1 = nn.Conv2d(3, num_planes, kernel_size=3, padding=1, bias=False)

        self.dense1 = self._make_dense_layers(block, num_planes, nblocks[0])
        num_planes += nblocks[0]*growth_rate
        out_planes = int(math.floor(num_planes*reduction))
        self.trans1 = Transition(num_planes, out_planes)
        num_planes = out_planes

        self.dense2 = self._make_dense_layers(block, num_planes, nblocks[1])
        num_planes += nblocks[1]*growth_rate
        out_planes = int(math.floor(num_planes*reduction))
        self.trans2 = Transition(num_planes, out_planes)
        num_planes = out_planes

        self.dense3 = self._make_dense_layers(block, num_planes, nblocks[2])
        num_planes += nblocks[2]*growth_rate
        out_planes = int(math.floor(num_planes*reduction))
        self.trans3 = Transition(num_planes, out_planes)
        num_planes = out_planes

        self.dense4 = self._make_dense_layers(block, num_planes, nblocks[3])
        num_planes += nblocks[3]*growth_rate

        self.bn = nn.BatchNorm2d(num_planes)
        self.linear = nn.Linear(num_planes, num_classes)

    def _make_dense_layers(self, block, in_planes, nblock):
        layers = []
        for i in range(nblock):
            layers.append(block(in_planes, self.growth_rate))
            in_planes += self.growth_rate
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.trans1(self.dense1(out))
        out = self.trans2(self.dense2(out))
        out = self.trans3(self.dense3(out))
        out = self.dense4(out)
        out = F.avg_pool2d(F.relu(self.bn(out)), 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


# In[ ]:


net = DenseNet(Bottleneck, [6,12,24,16], growth_rate=32)


# In[ ]:


x = torch.randn(1,3,32,32)
y = net(x)
print(y.size())
    


# In[ ]:


print(torch.cuda.is_available())


# In[ ]:


trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=32,
                                          shuffle=True, num_workers=4)

testloader = torch.utils.data.DataLoader(test_dataset, batch_size=8, 
                                         shuffle=False, num_workers=4)


# In[ ]:


import torch.optim as optim
net=net.to(device)
criterion=nn.CrossEntropyLoss()
optimizer=torch.optim.Adam(net.parameters(),lr=3e-04)


# In[ ]:


print(device)


# In[ ]:


def train(epoch):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in tqdm(enumerate(trainloader)):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        print(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
            % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))


# In[ ]:


for epoch in range(0,3):
    train(epoch)


# In[ ]:





# In[ ]:


test_preds10 = np.zeros((len(test_dataset), 1))
def test():
    with torch.no_grad():
        for batch_idx, inputs in tqdm(enumerate(testloader)):
            inputs = inputs.to(device)
            outputs = net(inputs)
            _, predicted = outputs.max(1)
            test_preds10[i * 8:(i + 1) * 8] = predicted.detach().cpu().squeeze().numpy().ravel().reshape(-1, 1)
            
            print(test_preds10[i * 8:(i + 1) * 8])


# In[ ]:


# test_data_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4)
test()


# In[ ]:


print(test_preds10.shape)


# In[ ]:


sample = pd.read_csv("../input/sample_submission.csv")
sample.diagnosis = test_preds10.astype(int)
sample.to_csv("submission.csv", index=False)


# In[ ]:


sample

