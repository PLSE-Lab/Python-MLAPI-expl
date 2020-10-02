#!/usr/bin/env python
# coding: utf-8

# """Wide ResNet-50-2 model from
#     `"Wide Residual Networks" <https://arxiv.org/pdf/1605.07146.pdf>`_
#     The model is the same as ResNet except for the bottleneck number of channels
#     which is twice larger in every block. The number of channels in outer 1x1
#     convolutions is the same, e.g. last block in ResNet-50 has 2048-512-2048
#     channels, and in Wide ResNet-50-2 has 2048-1024-2048.
#  """
# # TRAINING PART

# In[ ]:


import numpy as np
import pandas as pd 
import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset,DataLoader
from torchvision import transforms,models
from tqdm import tqdm_notebook as tqdm
import math
import torch.nn.functional as F
from torch.nn import init

for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# Load feather file for training

# In[ ]:


train = pd.read_csv('train.csv')
data0 = pd.read_feather('train_data_0.feather')
data1 = pd.read_feather('train_data_1.feather')
data2 = pd.read_feather('train_data_2.feather')
data3 = pd.read_feather('train_data_3.feather')


# In[ ]:


data_full = pd.concat([data0,data1,data2,data3],ignore_index=True)
len(data_full)


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
        image = self.df.iloc[idx][1:].values.reshape(64,64).astype(np.float)    #128 was 64
        return image,label1,label2,label3


# **** Model code

# In[ ]:


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)

def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class Bottleneck(nn.Module):
    expansion = 4
    __constants__ = ['downsample']

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=1000, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None):
        super(ResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(1, self.inplanes, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        # vowel_diacritic
        self.fc1 = nn.Linear(512 * block.expansion,11)
        # grapheme_root
        self.fc2 = nn.Linear(512 * block.expansion,168)
        # consonant_diacritic
        self.fc3 = nn.Linear(512 * block.expansion,7)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def _forward_impl(self, x):
        # See note [TorchScript super()]
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x1 = self.fc1(x)
        x2 = self.fc2(x)
        x3 = self.fc3(x)
        return x1,x2,x3

    def forward(self, x):
        return self._forward_impl(x)


def _resnet(arch, block, layers, pretrained, progress, **kwargs):
    model = ResNet(block, layers, **kwargs)
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls[arch],
                                              progress=progress)
        model.load_state_dict(state_dict)
    return model


# In[ ]:


def wide_resnet50_2(pretrained=False, progress=True, **kwargs):
    """
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    kwargs['width_per_group'] = 64 * 2
    return _resnet('wide_resnet50_2', Bottleneck, [3, 4, 6, 3],
                   pretrained, progress, **kwargs)


# In[ ]:


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# In[ ]:


model=wide_resnet50_2().to(device)
model.load_state_dict(torch.load('G:/Bengal Ai Hand Written Grapheme  Reconition/densenet169Weights/wideres50moreAcc99.158.pth'))
optimizer = torch.optim.Adam(model.parameters(), lr=4e-4)
#scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=1e-4, max_lr=0.05)
criterion = nn.CrossEntropyLoss()
batch_size=32   #increasing batch size degrades accuracy(in my case)whereas it decrease runtime also


# In[ ]:


#training + validation
epochs = 100
model.train()
losses = []
accs = []
val_losses = []
val_accs = []
for epoch in range(epochs):
    train_index =train.groupby(['grapheme_root', 'vowel_diacritic', 'consonant_diacritic']).apply(lambda x: x.sample(80)).image_id.values
    reduced_train = train.loc[train.image_id.isin(train_index)]
    train_data = data_full.loc[data_full.image_id.isin(train_index)]
    
    train_image = GraphemeDataset(train_data,reduced_train)
    
    ##data for training
    train_loader = torch.utils.data.DataLoader(train_image,batch_size=batch_size,shuffle=True)
    
    test_index =train.groupby(['grapheme_root', 'vowel_diacritic', 'consonant_diacritic']).apply(lambda x: x.sample(20)).image_id.values
    reduced_test = train.loc[train.image_id.isin(test_index)]
    test_data = data_full.loc[data_full.image_id.isin(test_index)]
    
    test_image = GraphemeDataset(test_data,reduced_test)
    
    ##data for test
    test_loader = torch.utils.data.DataLoader(test_image,batch_size=batch_size,shuffle=True)
    
    print('epochs {}/{} '.format(epoch+1,epochs))
    running_loss = 0.0
    running_acc = 0.0
    for idx, (inputs,labels1,labels2,labels3) in tqdm(enumerate(train_loader),total=len(train_loader)):
        inputs = inputs.to(device)
        labels1 = labels1.to(device)
        labels2 = labels2.to(device)
        labels3 = labels3.to(device)
        
        optimizer.zero_grad()
        outputs1,outputs2,outputs3 = model(inputs.unsqueeze(1).float())
        loss1 = criterion(outputs1,labels1)
        loss2 = criterion(outputs2,labels2)
        loss3 = criterion(outputs3,labels3)
        running_loss += loss1+loss2+loss3
        running_acc += (outputs1.argmax(1)==labels1).float().mean()
        running_acc += (outputs2.argmax(1)==labels2).float().mean()
        running_acc += (outputs3.argmax(1)==labels3).float().mean()
        (loss1+loss2+loss3).backward()
        optimizer.step()
        del(inputs)
        del(labels1)
        del(labels2)
        del(labels3)
        del(outputs1)
        del(outputs2)
        del(outputs3)
        del(loss1)
        del(loss2)
        del(loss3)
    #scheduler.step()
    losses.append(running_loss/len(train_loader))
    accs.append(running_acc/(len(train_loader)*3))
    print('acc : {:.4f}%'.format(running_acc*100/(len(train_loader)*3)))
    print('loss : {:.4f}'.format(running_loss/len(train_loader)))
    
    
    ## data feed for validation
    with torch.no_grad():
        running_loss = 0.0
        running_acc = 0.0
        for idx, (inputs,labels1,labels2,labels3) in tqdm(enumerate(test_loader),total=len(test_loader)):  #here tqdm is used for progressbar and total=len(t..) means prpgress bar highest value is len(t..)
            
            inputs = inputs.to(device)
            labels1 = labels1.to(device)
            labels2 = labels2.to(device)
            labels3 = labels3.to(device)
            
            outputs1,outputs2,outputs3 = model(inputs.unsqueeze(1).float())
            
            loss1 = criterion(outputs1,labels1)
            loss2 = criterion(outputs2,labels2)
            loss3 = criterion(outputs3,labels3)
            running_loss += loss1+loss2+loss3
            running_acc += (outputs1.argmax(1)==labels1).float().mean()
            running_acc += (outputs2.argmax(1)==labels2).float().mean()
            running_acc += (outputs3.argmax(1)==labels3).float().mean()
            del(inputs)
            del(labels1)
            del(labels2)
            del(labels3)
            del(outputs1)
            del(outputs2)
            del(outputs3)
            del(loss1)
            del(loss2)
            del(loss3)
        val_losses.append(running_loss/len(test_loader))
        val_accs.append(running_acc/(len(test_loader)*3))
        print('val_acc : {:.4f}%'.format(running_acc*100/(len(test_loader)*3)))
        path=os.path.join('G:/Bengal Ai Hand Written Grapheme  Reconition/densenet169Weights','wideres50moreAcc{0:.3f}.pth'.format(running_acc*100/(len(test_loader)*3)))
        if(running_acc*100/(len(test_loader)*3)>96.5):
            torch.save(model.state_dict(),path)     
        print('val_loss : {:.4f}'.format(running_loss/len(test_loader)))


# In[ ]:


import matplotlib.pyplot as plt
fig,ax = plt.subplots(1,2,figsize=(15,5))
ax[0].plot(losses,label='train')
ax[0].plot(val_losses,label='valid')
ax[0].set_title('loss')
ax[0].legend()
ax[1].plot(accs,label='train')
ax[1].plot(val_accs,label='valid')
ax[1].set_title('acc')
ax[0].legend()
plt.show()
fig.savefig('WideResnt50_8020split.png', dpi=100)


# # INFERENCE PART

# In[ ]:


test = pd.read_csv('/kaggle/input/bengaliai-cv19/test.csv')


# In[ ]:


class GraphemeDataset(Dataset):
    def __init__(self,df,_type='train'):
        self.df = df
    def __len__(self):
        return len(self.df)
    def __getitem__(self,idx):
        image = self.df.iloc[idx][1:].values.reshape(64,64).astype(float)
        return image


# In[ ]:


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = wide_resnet50_2().to(device)  #execute **Model code in the TRAINING section
model.load_state_dict(torch.load('/kaggle/input/wideres50/wideres50moreAcc99.428.pth'))


# In[ ]:


def Resize(df,size=64):
    resized = {} 
    df = df.set_index('image_id')
    for i in tqdm(range(df.shape[0])):
        image = cv2.resize(df.loc[df.index[i]].values.reshape(137,236),(size,size))
        resized[df.index[i]] = image.reshape(-1)
    resized = pd.DataFrame(resized).T.reset_index()
    resized.columns = resized.columns.astype(str)
    resized.rename(columns={'index':'image_id'},inplace=True)
    return resized


# In[ ]:


import cv2
model.eval()
test_data = ['test_image_data_0.parquet','test_image_data_1.parquet','test_image_data_2.parquet','test_image_data_3.parquet']
predictions = []
batch_size=1
for fname in test_data:
    data = pd.read_parquet(f'/kaggle/input/bengaliai-cv19/{fname}')
    data = Resize(data)
    test_image = GraphemeDataset(data)
    test_loader = torch.utils.data.DataLoader(test_image,batch_size=1,shuffle=False)
    with torch.no_grad():
        for idx, (inputs) in tqdm(enumerate(test_loader),total=len(test_loader)):
            inputs.to(device)
            outputs1,outputs2,outputs3 = model(inputs.unsqueeze(1).float().cuda())
            predictions.append(outputs3.argmax(1).cpu().detach().numpy())
            predictions.append(outputs2.argmax(1).cpu().detach().numpy())
            predictions.append(outputs1.argmax(1).cpu().detach().numpy())


# In[ ]:


submission = pd.read_csv('/kaggle/input/bengaliai-cv19/sample_submission.csv')
submission.target = np.hstack(predictions)
submission.head(10)


# In[ ]:


submission.to_csv('submission.csv',index=False)


# In[ ]:




