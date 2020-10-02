#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
import collections
import cv2
import datetime
import gc
import glob
import h5py
import logging
import math
import operator
import os 
import pickle
import random
import re
import sklearn
import scipy.signal
import scipy.stats as stats
import seaborn as sns
import string
import sys
import time
import torch
torch.backends.cudnn.benchmark = True
import torch.nn as nn
import torch.utils.data
import torch.nn.functional as F
from contextlib import contextmanager
from collections import Counter, defaultdict, OrderedDict
from sklearn import metrics
from sklearn import model_selection
from sklearn.model_selection import KFold, GroupKFold, StratifiedKFold
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import mean_squared_log_error, roc_auc_score, average_precision_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from skimage import measure
from torch.nn import CrossEntropyLoss, MSELoss
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau, MultiStepLR, ExponentialLR
from torch.utils import model_zoo
from torch.utils.data import (Dataset,DataLoader, RandomSampler, SequentialSampler,
                              TensorDataset)
import tensorflow as tf

from tqdm import tqdm
tqdm.pandas()

import warnings
warnings.filterwarnings('ignore')


# In[ ]:


DATA_PATH = '../input/trends-assessment-prediction/'
TEST_MAP_PATH = DATA_PATH + 'fMRI_test/'
TRAIN_MAP_PATH = DATA_PATH + 'fMRI_train/'
EXP_ID = 'exp1'
device = 'cuda'
EPOCHS = 3
fold_id = 0
SEED = 718
num_folds = 5

TRAIN_BATCH_SIZE = 32
VALID_BATCH_SIZE = 16


fnc_df = pd.read_csv(f"{DATA_PATH}/fnc.csv")
loading_df = pd.read_csv(f"{DATA_PATH}/loading.csv")
labels_df = pd.read_csv(f"{DATA_PATH}/train_scores.csv")


fnc_features, loading_features = list(fnc_df.columns[1:]), list(loading_df.columns[1:])
df = fnc_df.merge(loading_df, on="Id")
labels_df["is_train"] = True

df = df.merge(labels_df, on="Id", how="left")

train_ids = list(df[df["is_train"] == True].index)
test_ids = list(df[df["is_train"] != True]['Id'].index)


df['bin_age'] = pd.cut(df['age'], [i for i in range(0, 100, 10)], labels=False)

df_train = df.iloc[train_ids].reset_index(drop=True)
df_test = df.iloc[test_ids].reset_index(drop=True)


train_idx = list(df_train.index) # [:500]
target_cols = ['age', 'domain1_var1', 'domain1_var2', 'domain2_var1', 'domain2_var2']

# df_test = df_test.head(100)

print(df_test.shape)
df_test.head()


# In[ ]:


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import math
from functools import partial

__all__ = [
    'resnet10', 'resnet18', 'resnet34', 'resnet50', 'resnet101',
    'resnet152', 'resnet200'
]

def conv3x3x3(in_planes, out_planes, stride=1, dilation=1):
    # 3x3x3 convolution with padding
    return nn.Conv3d(
        in_planes,
        out_planes,
        kernel_size=3,
        dilation=dilation,
        stride=stride,
        padding=dilation,
        bias=False)

def downsample_basic_block(x, planes, stride, no_cuda=False):
    out = F.avg_pool3d(x, kernel_size=1, stride=stride)
    zero_pads = torch.Tensor(out.size(0), planes - out.size(1), out.size(2), out.size(3), out.size(4)).zero_()
    if not no_cuda:
        if isinstance(out.data, torch.cuda.FloatTensor):
            zero_pads = zero_pads.cuda()

    out = Variable(torch.cat([out.data, zero_pads], dim=1))
    return out

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, dilation=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3x3(inplanes, planes, stride=stride, dilation=dilation)
        self.bn1 = nn.BatchNorm3d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3x3(planes, planes, dilation=dilation)
        self.bn2 = nn.BatchNorm3d(planes)
        self.downsample = downsample
        self.stride = stride
        self.dilation = dilation

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, dilation=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv3d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm3d(planes)
        self.conv2 = nn.Conv3d(
            planes, planes, kernel_size=3, stride=stride, dilation=dilation, padding=dilation, bias=False)
        self.bn2 = nn.BatchNorm3d(planes)
        self.conv3 = nn.Conv3d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm3d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
        self.dilation = dilation

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

        out += residual
        out = self.relu(out)
        return out

class ResNet3D(nn.Module):

    def __init__(self,
                 block,
                 layers,
                 shortcut_type='B',
                 num_class = 5,
                 no_cuda=False):

        self.inplanes = 64
        self.no_cuda = no_cuda
        super(ResNet3D, self).__init__()

        # 3D conv net
        self.conv1 = nn.Conv3d(53, 64, kernel_size=7, stride=(2, 2, 2), padding=(3, 3, 3), bias=False)
        # self.conv1 = nn.Conv3d(1, 64, kernel_size=7, stride=(2, 2, 2), padding=(3, 3, 3), bias=False)
        self.bn1 = nn.BatchNorm3d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(kernel_size=(3, 3, 3), stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0], shortcut_type)
        self.layer2 = self._make_layer(
            block, 64*2, layers[1], shortcut_type, stride=2)
        self.layer3 = self._make_layer(
            block, 128*2, layers[2], shortcut_type, stride=1, dilation=2)
        self.layer4 = self._make_layer(
            block, 256*2, layers[3], shortcut_type, stride=1, dilation=4)

        self.fea_dim = 256*2 * block.expansion
        self.fc = nn.Sequential(nn.Linear(self.fea_dim, num_class, bias=True))

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                m.weight = nn.init.kaiming_normal_(m.weight, mode='fan_out')
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, shortcut_type, stride=1, dilation=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:

            if shortcut_type == 'A':
                downsample = partial(
                    downsample_basic_block,
                    planes=planes * block.expansion,
                    stride=stride,
                    no_cuda=self.no_cuda)
            else:
                downsample = nn.Sequential(
                    nn.Conv3d(
                        self.inplanes,
                        planes * block.expansion,
                        kernel_size=1,
                        stride=stride,
                        bias=False), nn.BatchNorm3d(planes * block.expansion))

        layers = []
        layers.append(block(self.inplanes, planes, stride=stride, dilation=dilation, downsample=downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, dilation=dilation))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1( x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = F.adaptive_avg_pool3d(x, (1, 1, 1))
        emb_3d = x.view((-1, self.fea_dim))
        out = self.fc(emb_3d)
        return out


def resnet10(**kwargs):
    """Constructs a ResNet-18 model.
    """
    model = ResNet3D(BasicBlock, [1, 1, 1, 1],**kwargs)
    return model


def resnet34(**kwargs):
    """Constructs a ResNet-34 model.
    """
    model = ResNet3D(BasicBlock, [3, 4, 6, 3], **kwargs)
    return model


# In[ ]:


class TReNDSDatasetTest:
    def __init__(self, df, target_cols, map_path):
        self.df = df
        self.map_path = map_path

    def __len__(self):
        return len(self.df)

    def __getitem__(self, item):

        IDX = self.df.iloc[item].Id        
        path = self.map_path + str(IDX)
        
        all_maps = h5py.File(path + '.mat', 'r')['SM_feature'][()]

        return {
            'features': torch.tensor(all_maps, dtype=torch.float32),
        }


# In[ ]:


test_dataset = TReNDSDatasetTest(df=df_test, target_cols=target_cols, map_path=TEST_MAP_PATH)
test_loader = torch.utils.data.DataLoader(
                test_dataset, shuffle=False, 
                batch_size=TRAIN_BATCH_SIZE,
                num_workers=0, pin_memory=True)

del test_dataset
gc.collect()


# In[ ]:


def test_fn(data_loader, model, device):
    model.eval()
    
    preds = []
    with torch.no_grad():
        tk0 = tqdm(data_loader, total=len(data_loader))
        for bi, d in enumerate(tk0):
            features = d["features"].to(device, dtype=torch.float32)

            outputs = model(features)
            outputs = outputs.cpu().detach().numpy()
            preds.append(outputs)
    
    return preds


# In[ ]:


ls ../input/trends-exp1/


# In[ ]:


'''
exp1

age metric: 0.17694576
domain1_var1 metric: 0.1504816
domain1_var2 metric: 0.14616399
domain2_var1 metric: 0.17815593
domain2_var2 metric: 0.18233863
all_metric: 0.16808325201272964
save model at score=0.16808325201272964 on epoch=12
'''


# In[ ]:


EXP_ID = 'exp1'

model = resnet10()
model.load_state_dict(torch.load(f'../input/trends-{EXP_ID}/{EXP_ID}_fold0.pth'))
model = model.to(device)


# In[ ]:


test_preds = test_fn(test_loader, model, 'cuda')
test_preds = np.concatenate(test_preds, 0)
print(test_preds.shape)


# In[ ]:


df_test.loc[:, target_cols] = test_preds


# In[ ]:


sub_df = pd.melt(df_test[["Id", "age", "domain1_var1", "domain1_var2", "domain2_var1", "domain2_var2"]], id_vars=["Id"], value_name="Predicted")
sub_df["Id"] = sub_df["Id"].astype("str") + "_" +  sub_df["variable"].astype("str")

sub_df = sub_df.drop("variable", axis=1).sort_values("Id")
assert sub_df.shape[0] == df_test.shape[0]*5

sub_df.to_csv("submission_3dconv.csv", index=False)
print(sub_df.shape)
sub_df.head()


# In[ ]:




