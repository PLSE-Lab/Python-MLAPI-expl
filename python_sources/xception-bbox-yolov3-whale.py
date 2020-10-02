#!/usr/bin/env python
# coding: utf-8

# In[ ]:


get_ipython().system('pip install albumentations > /dev/null')


# In[ ]:


# !pip install pretrainedmodels > /dev/null


# In[ ]:


ls ../input


# In[ ]:


import os

import albumentations
from albumentations import torch as AT
# import pretrainedmodels

import numpy as np
import pandas as pd

import cv2

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader, Dataset

from PIL import Image
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

from tqdm import tqdm, tqdm_notebook

from matplotlib import pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import warnings
warnings.simplefilter("ignore", category=DeprecationWarning)


# In[ ]:


BBOX_TRAIN = '../input/bboxsplit/bounding_boxes_train.csv'
BBOX_TEST = '../input/bboxsplit/bounding_boxes_test.csv'
train_df = pd.read_csv("../input/humpback-whale-identification/train.csv")
train_df.head()


# In[ ]:


# del_ind = []
# for i in range(len(train_df)):
#     if train_df.iloc[i]['Id'] == 'new_whale':
#         del_ind.append(i)
# train_df = train_df.drop(train_df.index[del_ind])


# In[ ]:


for i in range(4):
    newdf = train_df.groupby("Id").filter(lambda x: len(x) == i+1 )
#     newdf =pd.concat([newdf]*(4-i), ignore_index=False)
    train_df = train_df.append([newdf]*(4-i), ignore_index=True)


# In[ ]:


train_df.shape, train_df.Id.nunique()


# In[ ]:


NUM_CLASSES = train_df.Id.nunique()


# In[ ]:


NUM_CLASSES


# In[ ]:


train_df.Id.value_counts().iloc[1:].hist(bins=40)


# In[ ]:


RESIZE_H = 299
RESIZE_W = 299

data_transforms = albumentations.Compose([
    albumentations.Resize(RESIZE_H, RESIZE_W),
    albumentations.HorizontalFlip(),
    albumentations.OneOf([
        albumentations.RandomContrast(),
        albumentations.RandomBrightness(),
    ]),
    albumentations.ShiftScaleRotate(rotate_limit=10, scale_limit=0.15),
    albumentations.JpegCompression(80),
    albumentations.HueSaturationValue(),
#     albumentations.Normalize(),
    AT.ToTensor()
])

data_transforms_test = albumentations.Compose([
    albumentations.Resize(RESIZE_H, RESIZE_W),
#     albumentations.Normalize(),
    AT.ToTensor()
])


# In[ ]:


def prepare_labels(y):
    # From here: https://www.kaggle.com/pestipeti/keras-cnn-starter
    values = np.array(y)
    label_encoder = LabelEncoder()
    integer_encoded = label_encoder.fit_transform(values)

    onehot_encoder = OneHotEncoder(sparse=False)
    integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
    onehot_encoded = onehot_encoder.fit_transform(integer_encoded)

    y = onehot_encoded
    return y, label_encoder


# In[ ]:


y, lab_encoder = prepare_labels(train_df['Id'])


# In[ ]:


class WhaleDataset(Dataset):
    def __init__(self, datafolder, datatype='train', df=None, df_bbox = None, transform=None, y=None):
        self.datafolder = datafolder
        self.datatype = datatype
        self.y = y
        if self.datatype == 'train':
            self.df = df.values
        self.df_bbox = df_bbox
        if self.datatype == 'train':
            self.image_files_list = list(df.Image)
        else:
            self.image_files_list = [s for s in os.listdir(datafolder)]
        self.transform = transform


    def __len__(self):
        return len(self.image_files_list)
    
    def __getitem__(self, idx):
        if self.datatype == 'train':
            img_name = os.path.join(self.datafolder, self.df[idx][0])
            bbox = self.df_bbox.loc[self.df[idx][0]]
            x0, y0, x1, y1 = int(bbox['x0']), int(bbox['y0']), int(bbox['x1']),  int(bbox['y1'])
            label = self.y[idx]
            
        elif self.datatype == 'test':
            img_name = os.path.join(self.datafolder, self.image_files_list[idx])
            bbox = self.df_bbox.loc[self.image_files_list[idx]]
            x0, y0, x1, y1 = int(bbox['x0']), int(bbox['y0']), int(bbox['x1']),  int(bbox['y1'])
            label = np.zeros((NUM_CLASSES,))
        img = cv2.imread(img_name)
        img = img[y0:y1, x0:x1]
#         img = Image.fromarray(img)
        image = self.transform(image = img)['image']
        if self.datatype == 'train':
            return image, label
        elif self.datatype == 'test':
            # so that the images will be in a correct order
            return image, label, self.image_files_list[idx]


# In[ ]:


bbox_df_train = pd.read_csv(BBOX_TRAIN).set_index('Image')
bbox_df_test = pd.read_csv(BBOX_TEST).set_index('Image')


# In[ ]:


bbox = bbox_df_train.loc['0000e88ab.jpg']


# In[ ]:


img = cv2.imread('../input/humpback-whale-identification/train/0000e88ab.jpg')
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img = img[int(bbox['y0']):int(bbox['y1']), int(bbox['x0']):int(bbox['x1'])]
image = Image.fromarray(img)


# In[ ]:


image


# In[ ]:


train_dataset = WhaleDataset(
    datafolder='../input/humpback-whale-identification/train/', 
    datatype='train', 
    df=train_df, df_bbox =  bbox_df_train,
    transform=data_transforms, 
    y=y
)

test_set = WhaleDataset(
    datafolder='../input/humpback-whale-identification/test/', 
    datatype='test', df_bbox =  bbox_df_test,
    transform=data_transforms_test
)


# In[ ]:


batch_size = 32
num_workers = 0

train_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=num_workers, pin_memory=True)
test_loader = DataLoader(test_set, batch_size=batch_size, num_workers=num_workers, pin_memory=True)


# In[ ]:


"""
Ported to pytorch thanks to [tstandley](https://github.com/tstandley/Xception-PyTorch)
@author: tstandley
Adapted by cadene
Creates an Xception Model as defined in:
Francois Chollet
Xception: Deep Learning with Depthwise Separable Convolutions
https://arxiv.org/pdf/1610.02357.pdf
This weights ported from the Keras implementation. Achieves the following performance on the validation set:
Loss:0.9173 Prec@1:78.892 Prec@5:94.292
REMEMBER to set your image size to 3x299x299 for both test and validation
normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                  std=[0.5, 0.5, 0.5])
The resize parameter of the validation transform should be 333, and make sure to center crop at 299x299
"""
from __future__ import print_function, division, absolute_import
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo
from torch.nn import init

__all__ = ['xception']

pretrained_settings = {
    'xception': {
        'imagenet': {
            'url': 'http://data.lip6.fr/cadene/pretrainedmodels/xception-43020ad28.pth',
            'input_space': 'RGB',
            'input_size': [3, 299, 299],
            'input_range': [0, 1],
            'mean': [0.5, 0.5, 0.5],
            'std': [0.5, 0.5, 0.5],
            'num_classes': 1000,
            'scale': 0.8975 # The resize parameter of the validation transform should be 333, and make sure to center crop at 299x299
        }
    }
}


class SeparableConv2d(nn.Module):
    def __init__(self,in_channels,out_channels,kernel_size=1,stride=1,padding=0,dilation=1,bias=False):
        super(SeparableConv2d,self).__init__()

        self.conv1 = nn.Conv2d(in_channels,in_channels,kernel_size,stride,padding,dilation,groups=in_channels,bias=bias)
        self.pointwise = nn.Conv2d(in_channels,out_channels,1,1,0,1,1,bias=bias)

    def forward(self,x):
        x = self.conv1(x)
        x = self.pointwise(x)
        return x


class Block(nn.Module):
    def __init__(self,in_filters,out_filters,reps,strides=1,start_with_relu=True,grow_first=True):
        super(Block, self).__init__()

        if out_filters != in_filters or strides!=1:
            self.skip = nn.Conv2d(in_filters,out_filters,1,stride=strides, bias=False)
            self.skipbn = nn.BatchNorm2d(out_filters)
        else:
            self.skip=None

        self.relu = nn.ReLU(inplace=True)
        rep=[]

        filters=in_filters
        if grow_first:
            rep.append(self.relu)
            rep.append(SeparableConv2d(in_filters,out_filters,3,stride=1,padding=1,bias=False))
            rep.append(nn.BatchNorm2d(out_filters))
            filters = out_filters

        for i in range(reps-1):
            rep.append(self.relu)
            rep.append(SeparableConv2d(filters,filters,3,stride=1,padding=1,bias=False))
            rep.append(nn.BatchNorm2d(filters))

        if not grow_first:
            rep.append(self.relu)
            rep.append(SeparableConv2d(in_filters,out_filters,3,stride=1,padding=1,bias=False))
            rep.append(nn.BatchNorm2d(out_filters))

        if not start_with_relu:
            rep = rep[1:]
        else:
            rep[0] = nn.ReLU(inplace=False)

        if strides != 1:
            rep.append(nn.MaxPool2d(3,strides,1))
        self.rep = nn.Sequential(*rep)

    def forward(self,inp):
        x = self.rep(inp)

        if self.skip is not None:
            skip = self.skip(inp)
            skip = self.skipbn(skip)
        else:
            skip = inp

        x+=skip
        return x


class Xception(nn.Module):
    """
    Xception optimized for the ImageNet dataset, as specified in
    https://arxiv.org/pdf/1610.02357.pdf
    """
    def __init__(self, num_classes=1000):
        """ Constructor
        Args:
            num_classes: number of classes
        """
        super(Xception, self).__init__()
        self.num_classes = num_classes
        self.conv1 = nn.Conv2d(3, 32, 3,2, 0, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(32,64,3,bias=False)
        self.bn2 = nn.BatchNorm2d(64)
        #do relu here
        self.block1=Block(64,128,2,2,start_with_relu=False,grow_first=True)
        self.block2=Block(128,256,2,2,start_with_relu=True,grow_first=True)
        self.block3=Block(256,728,2,2,start_with_relu=True,grow_first=True)

        self.block4=Block(728,728,3,1,start_with_relu=True,grow_first=True)
        self.block5=Block(728,728,3,1,start_with_relu=True,grow_first=True)
        self.block6=Block(728,728,3,1,start_with_relu=True,grow_first=True)
        self.block7=Block(728,728,3,1,start_with_relu=True,grow_first=True)

        self.block8=Block(728,728,3,1,start_with_relu=True,grow_first=True)
        self.block9=Block(728,728,3,1,start_with_relu=True,grow_first=True)
        self.block10=Block(728,728,3,1,start_with_relu=True,grow_first=True)
        self.block11=Block(728,728,3,1,start_with_relu=True,grow_first=True)

        self.block12=Block(728,1024,2,2,start_with_relu=True,grow_first=False)

        self.conv3 = SeparableConv2d(1024,1536,3,1,1)
        self.bn3 = nn.BatchNorm2d(1536)

        #do relu here
        self.conv4 = SeparableConv2d(1536,2048,3,1,1)
        self.bn4 = nn.BatchNorm2d(2048)
        self.fc = nn.Linear(2048, num_classes)
    def features(self, input):
        x = self.conv1(input)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        x = self.block6(x)
        x = self.block7(x)
        x = self.block8(x)
        x = self.block9(x)
        x = self.block10(x)
        x = self.block11(x)
        x = self.block12(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu(x)

        x = self.conv4(x)
        x = self.bn4(x)
        return x

    def logits(self, features):
        x = self.relu(features)

        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = x.view(x.size(0), -1)
        x = self.last_linear(x)
        return x

    def forward(self, input):
        x = self.features(input)
        x = self.logits(x)
        return x


def xception(num_classes=1000, pretrained='imagenet'):
    model = Xception(num_classes=num_classes)
    if pretrained is not None:
        settings = pretrained_settings['xception'][pretrained]
        assert num_classes == settings['num_classes'],             "num_classes should be {}, but is {}".format(settings['num_classes'], num_classes)
#         model = Xception(num_classes=num_classes)
        model.load_state_dict(model_zoo.load_url(settings['url']))
        model.input_space = settings['input_space']
        model.input_size = settings['input_size']
        model.input_range = settings['input_range']
        model.mean = settings['mean']
        model.std = settings['std']
    # TODO: ugly
    model.last_linear = model.fc
    del model.fc
    return model
class Xception_base(nn.Module):
    """
    Xception optimized for the ImageNet dataset, as specified in
    https://arxiv.org/pdf/1610.02357.pdf
    """
    def __init__(self, num_classes=1000, pretrained='imagenet'):
        """ Constructor
        Args:
            num_classes: number of classes
        """
        super(Xception_base, self).__init__()
        if not pretrained is None:
            base_num_classes = 1000
            self.base_model = Xception(num_classes = base_num_classes)
            settings = pretrained_settings['xception'][pretrained]
            assert base_num_classes == settings['num_classes'],                 "num_classes should be {}, but is {}".format(settings['num_classes'], num_classes)
    #         model = Xception(num_classes=num_classes)
            self.base_model.load_state_dict(model_zoo.load_url(settings['url']))
            self.base_model.input_space = settings['input_space']
            self.base_model.input_size = settings['input_size']
            self.base_model.input_range = settings['input_range']
            self.base_model.mean = settings['mean']
            self.base_model.std = settings['std']
        else:
            self.base_model = Xception(num_classes=num_classes)
        del self.base_model.fc #= SeparableConv2d(2048,1536,3,1,1)
        self.conv5 = SeparableConv2d(2048, 3072,3,1,1)
        self.bn5 = nn.BatchNorm2d(3072)
        self.conv6 = SeparableConv2d(3072, 5120,3,1,1)
        self.bn6 = nn.BatchNorm2d(5120)
        self.last_linear = nn.Linear(5120, num_classes)
#         self.fc = nn.Linear(2048, num_classes)
    def forward(self, input):
        x = self.base_model.features(input)
        x = self.base_model.relu(x)
        x = self.conv5(x)
        x = self.bn5(x)
        x = self.base_model.relu(x)
        x = self.conv6(x)
        x = self.bn6(x)
        x = self.base_model.relu(x)
        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = x.view(x.size(0), -1)
        x = self.last_linear(x)
        return x


# In[ ]:


ls ../input


# In[ ]:


model = Xception_base(5005, pretrained = None)


# In[ ]:


model.cuda();


# In[ ]:


def save_checkpoint(state, is_best, fpath='checkpoint_5005.pth'):
    torch.save(state, fpath)
    if is_best:
        torch.save(state, 'best_model.pth')


# In[ ]:


criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0005)
try:
    checkpoint = torch.load('../input/cpoint/checkpoint_5005.pth')
    model.load_state_dict(checkpoint['state_dict'])
    start = checkpoint['epoch']
    print(start)
    optimizer.load_state_dict(checkpoint['optimizer'])
except:
    start = 0
scheduler = lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)


# In[ ]:


def cuda(x):
    return x.cuda(non_blocking=True) if torch.cuda.is_available() else x
def eval_(output, target, maxk):
    e_batch_size = target.size(0)
    _, pred = output.topk(maxk, 1, True, True)
#     pred = pred.t()
    correct = pred.eq(target.topk(1, 1, True, True)[1].expand_as(pred))
    res = []
    k = maxk
    correct_k = correct[:k].view(-1).float().sum(0)
    total = correct_k.item()
    res.append(correct_k.mul_(100.0 / batch_size))
    return res, total


# In[ ]:


n_epochs = start + 6
cur_epoch = 0
sum_true_pred = 0
for epoch in range(start, n_epochs):
    train_loss = []
    cur_epoch = epoch
    save_checkpoint({'epoch': epoch,
                     'state_dict': model.state_dict(),
                     'optimizer' : optimizer.state_dict(),
                    }, False)
    for batch_i, (data, target) in tqdm_notebook(enumerate(train_loader), total = len(train_loader)):
        data, target = cuda(data), cuda(target)

        optimizer.zero_grad()
        output = model(data)
        
        loss = criterion(output, target.float())
        with open('results.txt', 'a') as file:
            file.write(str(loss.item())+'\n')
        sum_true_pred+=(eval_(output, target, 5))[1]
        train_loss.append(loss.item())
        loss.backward()
        optimizer.step()
    scheduler.step()
    
    print(f'Epoch {epoch}, train loss: {np.mean(train_loss):.4f}')
save_checkpoint({'epoch': cur_epoch,
         'state_dict': model.state_dict(),
         'optimizer' : optimizer.state_dict(),
        }, False)


# In[ ]:


sub = pd.read_csv('../input/humpback-whale-identification/sample_submission.csv')
model.eval()
for (data, target, name) in tqdm_notebook(test_loader):
    data = cuda(data)
    output = model(data)
    output = output.cpu().detach().numpy()
    for i, (e, n) in enumerate(list(zip(output, name))):
        sub.loc[sub['Image'] == n, 'Id'] = ' '.join(lab_encoder.inverse_transform(e.argsort()[-5:][::-1]))
sub.to_csv('submission.csv', index=False)

