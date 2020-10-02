#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import argparse
from itertools import islice
import json
from pathlib import Path
import shutil
import warnings
from typing import Dict
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
from sklearn.metrics import fbeta_score
from sklearn.exceptions import UndefinedMetricWarning
import torch
from torch import nn, cuda
from multiprocessing.pool import ThreadPool
from torch.optim import Adam
from tqdm import tqdm
from pathlib import Path
from typing import Callable, List
from torch.nn.parallel.data_parallel import data_parallel
import cv2
import pandas as pd
from PIL import Image
import torch
import torch.utils.data.sampler as torchSampler
from torch.utils.data import Dataset
import random
import math
import torch.nn.functional
import torch.functional as F
import torchvision.models as M
from torchvision.transforms import (
    ToTensor, Normalize, Compose, Resize, CenterCrop, RandomCrop,
    RandomHorizontalFlip,ColorJitter,RandomRotation)
from scipy import ndimage
from __future__ import print_function, division, absolute_import
from collections import OrderedDict

import torch.nn as nn
from torch.utils import model_zoo
from functools import partial
def multi_apply(func, *args, **kwargs):
    pfunc = partial(func, **kwargs) if kwargs else func
    map_results = map(pfunc, *args)
    return tuple(map(list, zip(*map_results)))
import os


# In[ ]:


random.seed(233)
SIZE = 352

def random_scale(image, size_=-1):
    if size_ == -1:
        sizes = [round(SIZE * 0.8), SIZE, round(SIZE * 1.2)]
        size_ = random.choice(sizes)
    image = cv2.resize(image, (size_, size_))

    if size_ == SIZE:
        return image
    x_start = random.randint(0, abs(SIZE - size_))
    y_start = random.randint(0, abs(SIZE - size_))
    if size_ > SIZE:
        return image[x_start:x_start + SIZE, y_start:y_start + SIZE]
    else:
        image_zero = np.zeros((SIZE, SIZE, 3))
        image_zero[x_start:x_start + size_, y_start:y_start + size_] = image
        return image_zero.astype(np.uint8)

tensor_transform = Compose([
    Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
def transform_test(image):
    raw_image = image.copy()
    images = []
    for size in [SIZE]:
        image = raw_image.copy()

        image_changed = random_scale(image, size)
        image = np.transpose(image_changed, (2, 0, 1)).astype(np.float)
        image = torch.from_numpy(image).div(255).float()
        images.append(tensor_transform(image))

        image = np.fliplr(image_changed)
        image = np.transpose(image, (2, 0, 1)).astype(np.float)
        image = torch.from_numpy(image).div(255).float()
        images.append(tensor_transform(image))

    return images


# In[ ]:


ON_KAGGLE = True
N_CLASSES = 1103
DATA_ROOT = Path('../input/imet-2019-fgvc6')


# In[ ]:


def gmean_df(df: pd.DataFrame) -> pd.DataFrame:
    return df.groupby(level=0).agg(lambda x: gmean(list(x)))


def mean_df(df: pd.DataFrame) -> pd.DataFrame:
    return df.groupby(level=0).mean()


def load_model(model: nn.Module, path: Path) -> Dict:
    state = torch.load(str(path))
    model.load_state_dict(state['model'])
    print('Loaded model from epoch {epoch}, step {step:,}'.format(**state))
    return state


# In[ ]:



def load_model(model: nn.Module, path: Path) -> Dict:
    state = torch.load(str(path))
    model.load_state_dict(state['model'])
    print('Loaded model from epoch {epoch}, step {step:,}'.format(**state))
    return state
__all__ = ['SENet', 'senet154', 'se_resnet50', 'se_resnet101', 'se_resnet152',
           'se_resnext50_32x4d', 'se_resnext101_32x4d']

pretrained_settings = {
   
    'se_resnext101_32x4d': {
        'imagenet': {
            'url': 'http://data.lip6.fr/cadene/pretrainedmodels/se_resnext101_32x4d-3b2fe3d8.pth',
            'input_space': 'RGB',
            'input_size': [3, 224, 224],
            'input_range': [0, 1],
            'mean': [0.485, 0.456, 0.406],
            'std': [0.229, 0.224, 0.225],
            'num_classes': 1000
        }
    },
}


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
        # self.avg_pool = nn.AvgPool2d(7, stride=1)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
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


def initialize_pretrained_model(model, num_classes, settings):
    assert num_classes == settings['num_classes'],         'num_classes should be {}, but is {}'.format(
            settings['num_classes'], num_classes)
#     model.load_state_dict(model_zoo.load_url(settings['url']))
    model.last_linear = nn.Linear(model.last_linear.in_features, 1103)
    model.input_space = settings['input_space']
    model.input_size = settings['input_size']
    model.input_range = settings['input_range']
    model.mean = settings['mean']
    model.std = settings['std']
def senet154(num_classes=1000, pretrained='imagenet'):
    model = SENet(SEBottleneck, [3, 8, 36, 3], groups=64, reduction=16,
                  dropout_p=0.2, num_classes=num_classes)
    if pretrained is not None:
        settings = pretrained_settings['senet154'][pretrained]
        initialize_pretrained_model(model, num_classes, settings)
    return model

def se_resnext101_32x4d(num_classes=1000, pretrained='imagenet'):
    model = SENet(SEResNeXtBottleneck, [3, 4, 23, 3], groups=32, reduction=16,
                  dropout_p=None, inplanes=64, input_3x3=False,
                  downsample_kernel_size=1, downsample_padding=0,
                  num_classes=num_classes)
    if pretrained is not None:
        settings = pretrained_settings['se_resnext101_32x4d'][pretrained]
        initialize_pretrained_model(model, num_classes, settings)
    return model
def se_resnet152(num_classes=1000, pretrained='imagenet'):
    model = SENet(SEResNetBottleneck, [3, 8, 36, 3], groups=1, reduction=16,
                  dropout_p=None, inplanes=64, input_3x3=False,
                  downsample_kernel_size=1, downsample_padding=0,
                  num_classes=num_classes)
    if pretrained is not None:
        settings = pretrained_settings['se_resnet152'][pretrained]
        initialize_pretrained_model(model, num_classes, settings)
    return model
class SEResNet152(nn.Module):
    def __init__(self, num_classes,
                 pretrained=True, net_cls=M.resnet50, dropout=True):
        super().__init__()
        self.net = se_resnet152(num_classes=1000,pretrained='imagenet')
        # self.net.avgpool = AvgPool()
        if dropout:
            self.net.last_linear = nn.Sequential(
                nn.Dropout(0.2),
                nn.Linear(self.net.last_linear.in_features, num_classes),
            )
        else:
            self.net.last_linear = nn.Linear(self.net.last_linear.in_features, 1103)

    def fresh_params(self):
        return self.net.parameters()

    def forward(self, x):
        return self.net(x)
class SERNet101_32x4d(nn.Module):
    def __init__(self, num_classes,
                 pretrained=True, net_cls=M.resnet50, dropout=False):
        super().__init__()
        self.net = se_resnext101_32x4d(num_classes=1000,pretrained='imagenet')
        # self.net.avgpool = AvgPool()
        if dropout:
            self.net.fc = nn.Sequential(
                nn.Dropout(),
                nn.Linear(self.net.fc.in_features, num_classes),
            )
        else:
            self.net.last_linear = nn.Linear(self.net.last_linear.in_features, 1103)

    def fresh_params(self):
        return self.net.parameters()

    def forward(self, x):
        return self.net(x)
class AvgPool(nn.Module):
    def forward(self, x):
        return torch.nn.functional.avg_pool2d(x, x.shape[2:])


def create_net(net_cls, pretrained: bool):
    if ON_KAGGLE and pretrained:
        net = net_cls()
        model_name = net_cls.__name__
        weights_path = f'../input/{model_name}/{model_name}.pth'
        net.load_state_dict(torch.load(weights_path))
    else:
        net = net_cls(pretrained=pretrained)
    return net
class ResNet(nn.Module):
    def __init__(self, num_classes,
                 pretrained=True, net_cls=M.resnet152, dropout=False):
        super().__init__()
        self.net = create_net(net_cls, pretrained=False)
        self.net.avgpool = AvgPool()
        if dropout:
            self.net.fc = nn.Sequential(
                nn.Dropout(),
                nn.Linear(self.net.fc.in_features, num_classes),
            )
        else:
            self.net.fc = nn.Linear(self.net.fc.in_features, num_classes)
            # self.net.fc = nn.Sequential(
            #     nn.BatchNorm1d(self.net.fc.in_features,eps=1e-05,momentum=0.1,affine=True,track_running_stats=True),
            #     nn.Dropout(p=0.5),
            #     nn.Linear(in_features=self.net.fc.in_features,out_features=1500,bias=True),
            #     nn.ReLU(),
            #     nn.BatchNorm1d(1500,eps=1e-05,momentum=0.1,affine=True,track_running_stats=True),
            #     nn.Dropout(p=0.5),
            #     nn.Linear(in_features=1500,out_features=num_classes,bias=True)
            # )

    def fresh_params(self):
        return self.net.parameters()

    def forward(self, x):
        return self.net(x)
def se_resnext50_32x4d(num_classes=1000, pretrained='imagenet'):
    model = SENet(SEResNeXtBottleneck, [3, 4, 6, 3], groups=32, reduction=16,
                  dropout_p=None, inplanes=64, input_3x3=False,
                  downsample_kernel_size=1, downsample_padding=0,
                  num_classes=num_classes)
    if pretrained is not None:
        settings = pretrained_settings['se_resnext50_32x4d'][pretrained]
        initialize_pretrained_model(model, num_classes, settings)
    return model
# class SERNext50_32x4d(nn.Module):
#     def __init__(self, num_classes,
#                  pretrained=True, net_cls=M.resnet50, dropout=False):
#         super().__init__()
#         self.net = se_resnext50_32x4d(num_classes=1000, pretrained='imagenet')
#         # self.net.avgpool = AvgPool()
#         self.net.last_linear = nn.Linear(self.net.last_linear.in_features, 1103)

#     def fresh_params(self):
#         return self.net.parameters()

#     def forward(self, x):
#         return self.net(x)


# In[ ]:


import os
print(os.listdir("../input"))


# In[ ]:


models = []

model = se_resnext101_32x4d()
model.cuda()
load_model(model,Path('../input/0518-serx101-fd1/0518-serx101-fd1.pt'))
model.eval()
models.append(model)



# In[ ]:


class Dataset_met(Dataset):
    def __init__(self, names, labels=None, mode='train', transform=None):
        super(Dataset_met, self).__init__()
        self.names = names
        self.labels = labels
        self.mode = mode
        self.transform = transform

    def __len__(self):
        return len(self.names)

    def load_data(self, name, m='train'):
        image=cv2.imread('../input/imet-2019-fgvc6/{}/{}.png'.format(m, name))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image
    def __getitem__(self, index):
        if self.mode in ['train', 'valid']:
            name = self.names[index]
            label = self.labels[index]
            image = self.load_data(name)
            image = self.transform(image)
            label_zero = np.zeros(NUM_CLASSES)
            for l in label.split(' '):
                label_zero[int(l)] = 1
            return image, label_zero
        else:
            name = self.names[index].replace('.png', '')
            image = self.load_data(name, self.mode)
            image = self.transform(image)
            return image, name


# In[ ]:


NUM_TTA = 2
def test_collate(batch):
    batch_size = len(batch)
    images = []
    names = []
    for b in range(batch_size):
        images.extend(batch[b][0])
        names.append(batch[b][1])
    images = torch.stack(images, 0)
    return images, names
test_names = os.listdir('../input/imet-2019-fgvc6/test')


# In[ ]:


weights = [1]*1


# In[ ]:


dst_test = Dataset_met(test_names, mode='test', transform=transform_test)
dataloader_test = DataLoader(dst_test, shuffle=False, batch_size=64, num_workers=2, collate_fn=test_collate)


# In[ ]:


with torch.no_grad():
    all_names = []
    all_outs = []
    for data in tqdm(dataloader_test):
        images, names = data
        images = images.cuda()
        outs = 0
        for w, model in zip(weights, models):
            out = model(images)
            out = torch.sigmoid(out).data.cpu().numpy()
            outs += out * w
        outs /= sum(weights)
        outs_zero = 0
        for i in range(NUM_TTA):
            outs_zero += outs[i::NUM_TTA]
        outs = outs_zero/NUM_TTA
  
        for out, name in zip(outs, names):
            out_indexs = np.nonzero(out > 0.105)[0]
            str_out = ''
            for o in out_indexs:
                str_out += '{} '.format(int(o))
            if len(out_indexs) == 0:
                str_out = str(np.argmax(out))
            else:
                str_out = str_out[:-1]
            print(str_out)
            all_outs.append(str_out)
            all_names.append(name)
# id,attribute_ids
    pd.DataFrame({'id':all_names, 'attribute_ids':all_outs}).to_csv(
        'submission.csv', index=None)

