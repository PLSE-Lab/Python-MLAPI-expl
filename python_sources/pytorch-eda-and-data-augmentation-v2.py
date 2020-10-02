#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
import torchvision
import torch.nn as nn
from tqdm import tqdm
from PIL import Image, ImageFile
from torch.utils.data import Dataset
import torch
from torchvision import transforms
import os
import matplotlib.pyplot as plt
import seaborn as sns
import collections.abc
from collections import OrderedDict
from sklearn.model_selection import train_test_split
import re
import math
from typing import Optional
import torch.nn.init as init
from inspect import isfunction

device = torch.device("cuda:0")
ImageFile.LOAD_TRUNCATED_IMAGES = True
epochs = 15


# In[ ]:


from albumentations import (
    HorizontalFlip, IAAPerspective, ShiftScaleRotate, CLAHE, RandomRotate90,
    Transpose, ShiftScaleRotate, Blur, OpticalDistortion, GridDistortion, HueSaturationValue,
    IAAAdditiveGaussianNoise, GaussNoise, MotionBlur, MedianBlur, IAAPiecewiseAffine,
    IAASharpen, IAAEmboss, RandomContrast, RandomBrightness, Flip, OneOf, Compose, RandomGamma, 
    ElasticTransform, ChannelShuffle,RGBShift, Rotate
)


# In[ ]:


class RetinopathyDatasetTest(Dataset):
    def __init__(self, data,mode = 'test'):
        #self.data = pd.read_csv(csv_file)
        self.mode = mode
        self.data = data.reset_index()
        self.img_dir = '../input/aptos2019-blindness-detection/test_images' if mode == 'test' else '../input/aptos2019-blindness-detection/train_images' 
        _,_,_,self.transform = data_transforms('center') 
        
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_name = os.path.join(self.img_dir, self.data.loc[idx, 'id_code'] + '.png')
        image = Image.open(img_name)
        image = image.resize((320, 320), resample=Image.BILINEAR)
        image = self.transform(image)
        if self.mode == 'test':
            return {'image': transforms.ToTensor()(image)}
        else:
            return {'image': transforms.ToTensor()(image),'label': self.data.loc[idx,'diagnosis']}


# First, let's try to visualize how the images look like within test/training set.

# In[ ]:



for i,path in enumerate(os.listdir('../input/aptos2019-blindness-detection/test_images')):
    img_path = os.path.join('../input/aptos2019-blindness-detection/test_images',path)
    im = Image.open(img_path,'r')
    ax = plt.subplot(3,3,i + 1)
    ax.imshow(im)
    if i == 8:
        break


# In[ ]:



for i,path in enumerate(os.listdir('../input/aptos2019-blindness-detection/train_images')):
    img_path = os.path.join('../input/aptos2019-blindness-detection/train_images',path)
    im = Image.open(img_path,'r')
    ax = plt.subplot(3,3,i + 1)
    ax.imshow(im)
    if i == 8:
        break


# Okay, it looks like there is a great variety in shape and color in both of train/test dataset.
# Therefore, we would like to investigate the effect of data augmentation. Especially, considering we have different sizes/aspect ratios, we have to crop the images no matter what.
# One of the topics I would look into is the center crop vs random crop.
# 
#   
# 

# ## Class balance
# Also, let's see the class balance within the training set.
# If there is a huge class imbalance, the likelihood is they are optimized to predict specific class(es), and we definetely would want to avoid that.

# In[ ]:


training_class = pd.read_csv('../input/aptos2019-blindness-detection/train.csv')
sns.countplot(training_class['diagnosis'])


# Not that terrible. but probably should consider correcting the balance still.
# The strategy to correct the balance here is to simply apply data augmentation more on images of less frequent classes. 

# In[ ]:


def data_transforms(mode = 'random',img_size = 256):
    general_aug = Compose([
        OneOf([
            Transpose(),
            HorizontalFlip(),
            RandomRotate90()
            ]),
        ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.2, rotate_limit=15, p=.2),
        OneOf([
            OpticalDistortion(p=0.2),
            GridDistortion(distort_limit=0.2, p=.1),
            ElasticTransform(),
            ], p=1.)
        ], p=1)
    image_specific = Compose([
        OneOf([
            IAASharpen(),
            IAAEmboss(),
            RandomContrast(),
            RandomBrightness(),
            ], p=0.3)
        ])
    all_transf_pre = [
            transforms.RandomCrop(round(1.2 * img_size))
            ]

    all_trans_after = [
            transforms.CenterCrop(img_size)
            ]
    center_crop = [
            transforms.CenterCrop(img_size)
    ]
    normalize = [
            transforms.ToTensor()
            ]

    def get_augment(aug):
        def augment(image):
            return Image.fromarray(aug(image=np.array(image))['image'])
        return [augment]

    def normalize_to_full_image(img):
        return img
        #img = np.array(img).astype(np.float32)
        #img -= img.min()
        #img /= img.max()
        #img *= 255
        #return img.astype(np.uint8)

    pre_crop = transforms.Compose(all_transf_pre) 
    train_img_transform = transforms.Compose(get_augment(general_aug) + get_augment(image_specific) + [normalize_to_full_image])
    norm_transform = transforms.Compose(all_trans_after + normalize)
    val_transform = transforms.Compose(all_trans_after) if mode == 'random' else transforms.Compose(center_crop)

    return pre_crop, train_img_transform, norm_transform, val_transform


# For the augmentation, I adopted a variety of techniques, which can be categorized  as follows.
# 1. Shape transformation
# This includes affine transformation, such as rotation, shifting and scaling, flipping and nonlinear transformation.    
# 2. Color transformation
# This includes the change in brightness and contrast. 
# 
# In this kernel, I first resized the image to set the aspect ratio of each examples equal and then crops twice before and after applying any data augmentation to keep the black region resulting from data augmentation as small as possible.
# Below are some examples of augmented samples

# In[ ]:


pre_crop, train_img_transform, _, center_crop = data_transforms()
fig = plt.figure()
for i,path in enumerate(os.listdir('../input/aptos2019-blindness-detection/train_images')):
    img_path = os.path.join('../input/aptos2019-blindness-detection/train_images',path)
    im = Image.open(img_path,'r')
    im = pre_crop(im.resize((320, 320), resample=Image.BILINEAR))
    ax = fig.add_subplot(3,2,2 * i + 1)
    ax.imshow(center_crop(im))
    ax.set_title('original')
    ax = fig.add_subplot(3,2,2 * i + 2)
    ax.set_title('augmented')
    im = train_img_transform(im)
    ax.imshow(center_crop(im))
    
    if i == 2:
        break
fig.suptitle('original vs augmented')
fig.tight_layout()


# In[ ]:


def this_collate_fn(batch):
    elem = batch[0]
    return {key:torch.cat([d[key] for d in batch],dim = 0) for key in elem} 


# In[ ]:


train_df = pd.read_csv('../input/aptos2019-blindness-detection/train.csv')
tr, val = train_test_split(train_df, stratify=train_df.diagnosis, test_size=0.05)


# In[ ]:


class RetinopathyDatasetTrain(Dataset):
    def __init__(self, data ,img_size = 224):
        #self.data = pd.read_csv(csv_file)
        self.data = data.reset_index()
        most_freq_class_num = len(self.data.query('diagnosis == 0'))
        self.aug_times = {str(diagnosis):np.round(most_freq_class_num / len(self.data.query('diagnosis == ' + str(diagnosis)))) for diagnosis in self.data['diagnosis'].unique()}
        
        #self.images = [Image.open(os.path.join('../input/aptos2019-blindness-detection/train_images',path),'r') for i,path in enumerate(os.listdir('../input/aptos2019-blindness-detection/train_images'))] 
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        self.pre_crop, self.train_img_transform, self.norm_transform, _ = data_transforms()
        img_name = os.path.join('../input/aptos2019-blindness-detection/train_images', self.data.loc[idx, 'id_code'] + '.png')
        image = Image.open(img_name)
        image = self.pre_crop(image.resize((320, 320), resample=Image.BILINEAR))
        key = str(self.data.loc[idx,'diagnosis'])
        aug_time = int(self.aug_times[key])
        img_list = [self.norm_transform(image)] + [self.norm_transform(self.train_img_transform(image)) for i in range(aug_time)]
        labels = [torch.tensor(self.data.loc[idx,'diagnosis'])] * len(img_list)
        return {'image': torch.stack(img_list),'label':torch.stack(labels,dim = 0)}


# In[ ]:


class SEModule(nn.Module):

    def __init__(self, channels, reduction):
        super(SEModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(channels, channels // reduction, kernel_size=1,
                             padding=0)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(channels // reduction, channels, kernel_size=1,
                             padding=0)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        module_input = x
        x = self.avg_pool(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return module_input * x


class Bottleneck(nn.Module):
    """
    Base class for bottlenecks that implements `forward()` method.
    """
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

        if self.downsample is not None:
            residual = self.downsample(x)

        out = self.se_module(out) + residual
        out = self.relu(out)

        return out


class SEBottleneck(Bottleneck):
    """
    Bottleneck for SENet154.
    """
    expansion = 4

    def __init__(self, inplanes, planes, groups, reduction, stride=1,
                 downsample=None):
        super(SEBottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes * 2, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes * 2)
        self.conv2 = nn.Conv2d(planes * 2, planes * 4, kernel_size=3,
                               stride=stride, padding=1, groups=groups,
                               bias=False)
        self.bn2 = nn.BatchNorm2d(planes * 4)
        self.conv3 = nn.Conv2d(planes * 4, planes * 4, kernel_size=1,
                               bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.se_module = SEModule(planes * 4, reduction=reduction)
        self.downsample = downsample
        self.stride = stride


# In[ ]:


class SEResNetBottleneck(Bottleneck):
    """
    ResNet bottleneck with a Squeeze-and-Excitation module. It follows Caffe
    implementation and uses `stride=stride` in `conv1` and not in `conv2`
    (the latter is used in the torchvision implementation of ResNet).
    """
    expansion = 4

    def __init__(self, inplanes, planes, groups, reduction, stride=1,
                 downsample=None):
        super(SEResNetBottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False,
                               stride=stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, padding=1,
                               groups=groups, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.se_module = SEModule(planes * 4, reduction=reduction)
        self.downsample = downsample
        self.stride = stride


class SEResNeXtBottleneck(Bottleneck):
    """
    ResNeXt bottleneck type C with a Squeeze-and-Excitation module.
    """
    expansion = 4

    def __init__(self, inplanes, planes, groups, reduction, stride=1,
                 downsample=None, base_width=4):
        super(SEResNeXtBottleneck, self).__init__()
        width = math.floor(planes * (base_width / 64)) * groups
        self.conv1 = nn.Conv2d(inplanes, width, kernel_size=1, bias=False,
                               stride=1)
        self.bn1 = nn.BatchNorm2d(width)
        self.conv2 = nn.Conv2d(width, width, kernel_size=3, stride=stride,
                               padding=1, groups=groups, bias=False)
        self.bn2 = nn.BatchNorm2d(width)
        self.conv3 = nn.Conv2d(width, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.se_module = SEModule(planes * 4, reduction=reduction)
        self.downsample = downsample
        self.stride = stride


class SENet(nn.Module):

    def __init__(self, block, layers, groups, reduction, dropout_p=0.2,
                 inplanes=128, input_3x3=True, downsample_kernel_size=3,
                 downsample_padding=1, num_classes=1000):
        """
        Parameters
        ----------
        block (nn.Module): Bottleneck class.
            - For SENet154: SEBottleneck
            - For SE-ResNet models: SEResNetBottleneck
            - For SE-ResNeXt models:  SEResNeXtBottleneck
        layers (list of ints): Number of residual blocks for 4 layers of the
            network (layer1...layer4).
        groups (int): Number of groups for the 3x3 convolution in each
            bottleneck block.
            - For SENet154: 64
            - For SE-ResNet models: 1
            - For SE-ResNeXt models:  32
        reduction (int): Reduction ratio for Squeeze-and-Excitation modules.
            - For all models: 16
        dropout_p (float or None): Drop probability for the Dropout layer.
            If `None` the Dropout layer is not used.
            - For SENet154: 0.2
            - For SE-ResNet models: None
            - For SE-ResNeXt models: None
        inplanes (int):  Number of input channels for layer1.
            - For SENet154: 128
            - For SE-ResNet models: 64
            - For SE-ResNeXt models: 64
        input_3x3 (bool): If `True`, use three 3x3 convolutions instead of
            a single 7x7 convolution in layer0.
            - For SENet154: True
            - For SE-ResNet models: False
            - For SE-ResNeXt models: False
        downsample_kernel_size (int): Kernel size for downsampling convolutions
            in layer2, layer3 and layer4.
            - For SENet154: 3
            - For SE-ResNet models: 1
            - For SE-ResNeXt models: 1
        downsample_padding (int): Padding for downsampling convolutions in
            layer2, layer3 and layer4.
            - For SENet154: 1
            - For SE-ResNet models: 0
            - For SE-ResNeXt models: 0
        num_classes (int): Number of outputs in `last_linear` layer.
            - For all models: 1000
        """
        super(SENet, self).__init__()
        self.inplanes = inplanes
        if input_3x3:
            layer0_modules = [
                ('conv1', nn.Conv2d(3, 64, 3, stride=2, padding=1,
                                    bias=False)),
                ('bn1', nn.BatchNorm2d(64)),
                ('relu1', nn.ReLU(inplace=True)),
                ('conv2', nn.Conv2d(64, 64, 3, stride=1, padding=1,
                                    bias=False)),
                ('bn2', nn.BatchNorm2d(64)),
                ('relu2', nn.ReLU(inplace=True)),
                ('conv3', nn.Conv2d(64, inplanes, 3, stride=1, padding=1,
                                    bias=False)),
                ('bn3', nn.BatchNorm2d(inplanes)),
                ('relu3', nn.ReLU(inplace=True)),
            ]
        else:
            layer0_modules = [
                ('conv1', nn.Conv2d(3, inplanes, kernel_size=7, stride=2,
                                    padding=3, bias=False)),
                ('bn1', nn.BatchNorm2d(inplanes)),
                ('relu1', nn.ReLU(inplace=True)),
            ]
        # To preserve compatibility with Caffe weights `ceil_mode=True`
        # is used instead of `padding=1`.
        layer0_modules.append(('pool', nn.MaxPool2d(3, stride=2,
                                                    ceil_mode=True)))
        self.layer0 = nn.Sequential(OrderedDict(layer0_modules))
        self.layer1 = self._make_layer(
            block,
            planes=64,
            blocks=layers[0],
            groups=groups,
            reduction=reduction,
            downsample_kernel_size=1,
            downsample_padding=0
        )
        self.layer2 = self._make_layer(
            block,
            planes=128,
            blocks=layers[1],
            stride=2,
            groups=groups,
            reduction=reduction,
            downsample_kernel_size=downsample_kernel_size,
            downsample_padding=downsample_padding
        )
        self.layer3 = self._make_layer(
            block,
            planes=256,
            blocks=layers[2],
            stride=2,
            groups=groups,
            reduction=reduction,
            downsample_kernel_size=downsample_kernel_size,
            downsample_padding=downsample_padding
        )
        self.layer4 = self._make_layer(
            block,
            planes=512,
            blocks=layers[3],
            stride=2,
            groups=groups,
            reduction=reduction,
            downsample_kernel_size=downsample_kernel_size,
            downsample_padding=downsample_padding
        )
        self.avg_pool = nn.AvgPool2d(7, stride=1)
        self.dropout = nn.Dropout(dropout_p) if dropout_p is not None else None
        self.last_linear = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, planes, blocks, groups, reduction, stride=1,
                    downsample_kernel_size=1, downsample_padding=0):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=downsample_kernel_size, stride=stride,
                          padding=downsample_padding, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, groups, reduction, stride,
                            downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups, reduction))

        return nn.Sequential(*layers)

    def features(self, x):
        x = self.layer0(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x

    def logits(self, x):
        x = self.avg_pool(x)
        if self.dropout is not None:
            x = self.dropout(x)
        x = x.view(x.size(0), -1)
        x = self.last_linear(x)
        return x

    def forward(self, x):
        x = self.features(x)
        x = self.logits(x)
        return x




def se_resnext50_32x4d(num_classes=1000, pretrained= True):
    model = SENet(SEResNeXtBottleneck, [3, 4, 6, 3], groups=32, reduction=16,
                  dropout_p=None, inplanes=64, input_3x3=False,
                  downsample_kernel_size=1, downsample_padding=0,
                  num_classes=num_classes)
    if pretrained:
        pretrained_state = torch.load("../input/seresnext50/seresnext50.pth")
        #model_dict = model.state_dict()
        #pretrained_state = {k: v for k, v in pretrained_state.items() if k in model_dict}
        model.load_state_dict(pretrained_state)
    return model

def se_resnext101_32x4d(num_classes=1000, pretrained='imagenet'):
    model = SENet(SEResNeXtBottleneck, [3, 4, 23, 3], groups=32, reduction=16,
                  dropout_p=None, inplanes=64, input_3x3=False,
                  downsample_kernel_size=1, downsample_padding=0,
                  num_classes=num_classes)
    if pretrained:
        pretrained_state = torch.load("../input/pritrained-se-resnet-101/se_resnext101_32x4d-3b2fe3d8.pth")
        #model_dict = model.state_dict()
        #pretrained_state = {k: v for k, v in pretrained_state.items() if k in model_dict}
        model.load_state_dict(pretrained_state)
    return model


# In[ ]:


#model = torchvision.models.resnext50_32x4d(pretrained=False)
model = se_resnext101_32x4d()
model.last_linear = nn.Linear(8192,5)
model = model.to(device)
optimizer = torch.optim.Adam(model.parameters(),lr = 1e-3)


# In[ ]:


train_ds = RetinopathyDatasetTrain(tr)
val_ds = RetinopathyDatasetTest(val,mode = 'val')
train_loader = torch.utils.data.DataLoader(train_ds, batch_size=8,collate_fn=this_collate_fn, shuffle = True, num_workers=4)
val_loader = torch.utils.data.DataLoader(val_ds, batch_size=32, shuffle = False, num_workers=4)


# In[ ]:


#test_preds = np.zeros((len(test_dataset), 1))
criterion = nn.CrossEntropyLoss()
for epoch in range(epochs):
    train_loss = []
    val_loss = []
    model.train()
    for i, x_batch in enumerate(train_loader):
        
        model.zero_grad()
        img = x_batch["image"]
        img = img.to(device).float()
        label = x_batch['label'].to(device).long()
        output = model(img)
        loss = criterion(output,label)
        train_loss.append(loss)
        loss.backward()
        optimizer.step()   
    model.eval()
    with torch.no_grad():
        for j,x_batch in enumerate(val_loader):
            img = x_batch["image"]
            img = img.to(device).float()
            label = x_batch['label'].to(device).long()
            output = model(img)
            loss = criterion(output,label)
            val_loss.append(loss)
    train_mean_loss = torch.mean(torch.stack(train_loss)).data.cpu().numpy()
    val_mean_loss = torch.mean(torch.stack(val_loss)).data.cpu().numpy()
        
    print(f'Epoch {epoch}, train loss: {train_mean_loss:.4f}, valid loss: {val_mean_loss:.4f}.')


# In[ ]:



for param in model.parameters():
    param.requires_grad = False

model.eval()


# In[ ]:



test_df = pd.read_csv('../input/aptos2019-blindness-detection/test.csv')
test_dataset = RetinopathyDatasetTest(test_df,mode = 'test')
test_data_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4)


# In[ ]:



test_preds = np.zeros((len(test_dataset), 1))

for i, x_batch in enumerate(test_data_loader):
    x_batch = x_batch["image"]
    _,pred = torch.max(model(x_batch.to(device)),1)
    test_preds[i * 32:(i + 1) * 32] = pred.cpu().squeeze().numpy().ravel().reshape(-1, 1)
"""    


# In[ ]:



sample = pd.read_csv("../input/aptos2019-blindness-detection/sample_submission.csv")
sample.diagnosis = test_preds.astype(int)
sample.to_csv("submission.csv", index=False)


# * To do; improve some data augmentations to avoid obviously wrong ones
# 
