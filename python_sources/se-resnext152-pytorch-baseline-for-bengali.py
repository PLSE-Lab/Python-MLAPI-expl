#!/usr/bin/env python
# coding: utf-8

# **This is a simple resnext50 baseline kernel for bengali.ai competition,i will try to gradually update this kernel**

# 
# # If you find this kernel interesting, please drop an UPVOTE. It motivates me to produce more quality content :)
# 

# # se_resnext50 PyTorch baseline
# 
# 
# - References (ResNet):
#   - https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py
#   - https://arxiv.org/pdf/1512.03385.pdf
#   
#   
# - Acknowledgements:
#   - Original kernels: https://www.kaggle.com/hanjoonchoe/grapheme-resnet-18-n-l-inference-lb-0-8566 and https://www.kaggle.com/hanjoonchoe/grapheme-resnet-18-naive-learning-3
#   
#   
# - **Kindly upvote the kernel if you found it helpful, including the original author's!**

# # Part 1

# In[ ]:


from __future__ import print_function, division, absolute_import
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import torch.nn.functional as F
import os

# Any results you write to the current directory are saved as output.
import torch
import torch.nn as nn
from torch.utils.data import Dataset,DataLoader
from torchvision import transforms,models
from tqdm import tqdm_notebook as tqdm
import math
import torch.utils.model_zoo as model_zoo


# In[ ]:


train = pd.read_csv('/kaggle/input/bengaliai-cv19/train.csv')
data0 = pd.read_feather('/kaggle/usr/lib/resize_and_load_with_feather_format_much_faster/train_data_0.feather')
data1 = pd.read_feather('/kaggle/usr/lib/resize_and_load_with_feather_format_much_faster/train_data_1.feather')
data2 = pd.read_feather('/kaggle/usr/lib/resize_and_load_with_feather_format_much_faster/train_data_2.feather')
data3 = pd.read_feather('/kaggle/usr/lib/resize_and_load_with_feather_format_much_faster/train_data_3.feather')


# In[ ]:


ls /kaggle/usr/lib/resize_and_load_with_feather_format_much_faster/


# In[ ]:


data_full = pd.concat([data0,data1,data2,data3],ignore_index=True)


# In[ ]:


class GraphemeDataset(Dataset):
    def __init__(self,df,label,_type='train'):
        self.df = df
        self.label = label
    def __len__(self):
        return len(self.df)
    def __getitem__(self,idx):
        label1 = self.label.vowel_diacritic.values[idx]
        label2 = self.label.grapheme_root.values[idx]
        label3 = self.label.consonant_diacritic.values[idx]
        image = self.df.iloc[idx][1:].values.reshape(64,64).astype(np.float)
        return image,label1,label2,label3


# ## resnext50_32x4d Model

# In[ ]:



class Selayer(nn.Module):

    def __init__(self, inplanes):
        super(Selayer, self).__init__()
        self.global_avgpool = nn.AdaptiveAvgPool2d(1)
        self.conv1 = nn.Conv2d(inplanes, int(inplanes / 16), kernel_size=1, stride=1)
        self.conv2 = nn.Conv2d(int(inplanes / 16), inplanes, kernel_size=1, stride=1)
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):

        out = self.global_avgpool(x)

        out = self.conv1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.sigmoid(out)

        return x * out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, cardinality, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes * 2, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes * 2)

        self.conv2 = nn.Conv2d(planes * 2, planes * 2, kernel_size=3, stride=stride,
                               padding=1, groups=cardinality, bias=False)
        self.bn2 = nn.BatchNorm2d(planes * 2)

        self.conv3 = nn.Conv2d(planes * 2, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)

        self.selayer = Selayer(planes * 4)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        out = self.selayer(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class SeResNeXt(nn.Module):

    def __init__(self, block, layers, cardinality=32, num_classes=1000):
        super(SeResNeXt, self).__init__()
        self.cardinality = cardinality
        self.inplanes = 64

        self.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, self.cardinality, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, self.cardinality))
                             
        # vowel_diacritic
        self.fc1 = nn.Linear(2048,11)
        # grapheme_root
        self.fc2 = nn.Linear(2048,168)
        # consonant_diacritic
        self.fc3 = nn.Linear(2048,7)
        return nn.Sequential(*layers)
        

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        
        x1 = self.fc1(x)
        x2 = self.fc2(x)
        x3 = self.fc3(x)
        
        return x1,x2,x3


def se_resnext50(**kwargs):
    """Constructs a SeResNeXt-50 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = SeResNeXt(Bottleneck, [3, 4, 6, 3], **kwargs)
    return model


def se_resnext101(**kwargs):
    """Constructs a SeResNeXt-101 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = SeResNeXt(Bottleneck, [3, 4, 23, 3], **kwargs)
    return model


def se_resnext152(**kwargs):
    """Constructs a SeResNeXt-152 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = SeResNeXt(Bottleneck, [3, 8, 36, 3], **kwargs)
    return model


# In[ ]:


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# ## Training Model
# 
# 

# In[ ]:


model = se_resnext50().to(device)
optimizer = optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
#scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=1e-4, max_lr=0.05)
criterion = nn.CrossEntropyLoss()
batch_size=64


# In[ ]:


'''def criterion(input, target, size_average=True):
    """Categorical cross-entropy with logits input and one-hot target"""
    l = -(target * torch.log(F.softmax(input, dim=1) + 1e-10)).sum(1)
    if size_average:
        l = l.mean()
    else:
        l = l.sum()
    return l'''


# In[ ]:


get_ipython().run_cell_magic('time', '', "epochs = 200\nmodel.train()\nlosses = []\naccs = []\nfor epoch in range(epochs):\n    reduced_index =train.groupby(['grapheme_root', 'vowel_diacritic', 'consonant_diacritic']).apply(lambda x: x.sample(5)).image_id.values\n    reduced_train = train.loc[train.image_id.isin(reduced_index)]\n    reduced_data = data_full.loc[data_full.image_id.isin(reduced_index)]\n    train_image = GraphemeDataset(reduced_data,reduced_train)\n    train_loader = torch.utils.data.DataLoader(train_image,batch_size=batch_size,shuffle=True)\n    \n    #print('epochs {}/{} '.format(epoch+1,epochs))\n    running_loss = 0.0\n    running_acc = 0.0\n    for idx, (inputs,labels1,labels2,labels3) in tqdm(enumerate(train_loader),total=len(train_loader)):\n        inputs = inputs.to(device)\n        labels1 = labels1.to(device)\n        labels2 = labels2.to(device)\n        labels3 = labels3.to(device)\n        \n        optimizer.zero_grad()\n        outputs1,outputs2,outputs3 = model(inputs.unsqueeze(1).float())\n        loss1 = criterion(outputs1,labels1)\n        loss2 = criterion(outputs2,labels2)\n        loss3 = criterion(outputs3,labels3)\n        running_loss += loss1+loss2+loss3\n        running_acc += (outputs1.argmax(1)==labels1).float().mean()\n        running_acc += (outputs2.argmax(1)==labels2).float().mean()\n        running_acc += (outputs3.argmax(1)==labels3).float().mean()\n        (loss1+loss2+loss3).backward()\n        optimizer.step()\n    #scheduler.step()\n    losses.append(running_loss/len(train_loader))\n    accs.append(running_acc/(len(train_loader)*3))\n    print('acc : {:.4f}%'.format(running_acc/(len(train_loader)*3)))\n    print('loss : {:.4f}'.format(running_loss/len(train_loader)))\ntorch.save(model.state_dict(), 'se_resnext50.pth')")


# In[ ]:


import matplotlib.pyplot as plt
fig,ax = plt.subplots(1,2,figsize=(15,5))
ax[0].plot(losses)
ax[0].set_title('loss')
ax[1].plot(accs)
ax[1].set_title('acc')


# # inference kernel : https://www.kaggle.com/mobassir/se-resnext50-pytorch-inference

# **References**
# 
# * https://www.kaggle.com/khoongweihao/resnet-34-pytorch-starter-kit/data
# * https://www.kaggle.com/hanjoonchoe/grapheme-resnet-18-naive-learning-3
