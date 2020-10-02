#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# In[ ]:


from typing import Tuple

import albumentations as A
import pandas as pd
import torch
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import cv2
from torch.nn import functional as F


def get_image(item: pd.Series, image_dir: str):
    image_path = os.path.join(image_dir, item.filename)
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image


def get_transform(image_size, normalize=True, train=True):
    if train:
        transforms = [
            A.Resize(height=image_size, width=image_size),
            A.Transpose(p=0.5),
            A.VerticalFlip(p=0.5),
            A.HorizontalFlip(p=0.5),
        ]
    else:
        transforms = [
        ]

    if normalize:
        transforms.append(A.Normalize())

    transforms.extend([
        ToTensorV2(),
    ])
    return A.Compose(transforms)


class ShopeeDataset(Dataset):
    def __init__(
            self, df: pd.DataFrame, image_dir: str,
            transform=None,
    ):
        self.df = df
        self.image_dir = image_dir
        self.transform = transform

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx) -> Tuple:
        item = self.df.iloc[idx]
        image = get_image(item, self.image_dir)
        image_id = item.filename
        # get label
        label = item.category
        data = {
            'image': image,
            'labels': label,
        }

        if self.transform is not None:
            data = self.transform(**data)

        image = data['image']
        label = torch.tensor(data['labels'], dtype=torch.long)

        return image, label


# In[ ]:


import torch
from torch import nn

BACKBONE='resnext101_32x8d_swsl'


class ResNetHead(nn.Module):
    def __init__(self, in_features: int, n_classes: int, use_neck=0):
        super().__init__()

        self.pooling = nn.AdaptiveAvgPool2d((1, 1))
        self.fc1 = nn.Linear(in_features, n_classes)
        self.use_neck = use_neck

    def forward(self, x):
        if not self.use_neck:
            x = self.pooling(x)
            x = torch.flatten(x, start_dim=1)
        x = self.apply_fc_out(x)
        return x

    def apply_fc_out(self, x):
        return self.fc1(x)
    
    
class ResNetBase(nn.Module):
    def __init__(self, backbone='resnext50_32x4d_ssl'):
        super(ResNetBase, self).__init__()
        if backbone.endswith('l'):
            self.backbone = torch.hub.load(
                'facebookresearch/semi-supervised-ImageNet1K-models',
                backbone,
            )
        else:
            self.backbone = getattr(models, backbone)(pretrained=True)
        self.out_features = self.backbone.fc.in_features

    def forward(self, x):
        base = self.backbone
        x = base.conv1(x)
        x = base.bn1(x)
        x = base.relu(x)
        x = base.maxpool(x)

        x = base.layer1(x)
        x = base.layer2(x)
        x = base.layer3(x)
        x = base.layer4(x)
        return x


def build_model(backbone: str, n_classes: int, **kwargs) -> nn.Module:
    return Model(backbone=backbone, n_classes=n_classes, **kwargs)


class Model(nn.Module):
    def __init__(self, *, backbone: str, n_classes: int, use_neck=0):
        super().__init__()

        self.backbone = ResNetBase(backbone)
        self.in_features = self.backbone.out_features
        self.use_neck = use_neck
        if self.use_neck:
            self.hidden_dim = 1024
            self.neck = Neck(self.in_features, 1024)
            self.in_features = self.hidden_dim
        self.head = ResNetHead(self.in_features, n_classes, self.use_neck)

    def forward(self, x):
        x = self.backbone(x)
        if self.use_neck:
            x = self.neck(x)
        x = self.head(x)
        return x


# In[ ]:


class WrappedModel(nn.Module):
    def __init__(self, module):
        super(WrappedModel, self).__init__()
        self.module = module # that I actually define.
    def forward(self, x):
        return self.module(x)


# In[ ]:


def load_models(model_files):
    models = []
    for model_f in model_files:
        model_f = os.path.join(model_dir, model_f)
        model = build_model(n_classes=42, backbone=BACKBONE)
        model = WrappedModel(model)
        model.load_state_dict(torch.load(model_f, map_location=lambda storage, loc: storage), strict=True)
        model.eval()
        model.to(device)
        models.append(model)
        print(f'{model_f} loaded!')
    return models


# In[ ]:


image_test_dir = '../input/shopeetestcentercrop/test_256x256/test_256x256/'
image_test_dir_2 = '../input/shopeetestcentercrop/test_center_crop_128x128/test_center_crop_128x128/'
# model_dir = '../input/shopeemodels/'
model_dir = '../input/shopeeresnext101cutmix/'

device = torch.device('cuda')

df_test = pd.read_csv('../input/shopee-product-detection-open/test.csv')

model_files = [
    'best_fold0.pth',
    'best_fold1.pth',
    'best_fold2_839.pth',
    'best_fold3.pth',
#     'best_fold2_837.pth',
]
models = load_models(model_files)


# In[ ]:


IMAGE_SIZE = 128
BATCH_SIZE = 64


# In[ ]:


test_dataset = ShopeeDataset(
        df_test, image_dir=image_test_dir,
        transform=get_transform(image_size=IMAGE_SIZE, train=False),
)
data_test_loader = DataLoader(
        test_dataset, batch_size=BATCH_SIZE,
        shuffle=False
)


# In[ ]:


FINAL_PRED = []
with torch.no_grad():
    for (data, targets) in tqdm(data_test_loader):
        data = data.to(device, dtype=torch.float)

        preds = []
        for model in models:
            model = model.to(device)
            output = model(data)
            preds.append(output)

        # ----ensemble models----#
        proba = torch.mean(torch.stack([pred for pred in preds], dim=0), dim=0)


        pred = F.softmax(proba, dim=1).data.cpu().numpy().argmax(axis=1)
        for i in range(len(pred)):
            FINAL_PRED.append(pred[i])


# # TTA

# In[ ]:


# LOGITS_FINAL = []
# for i, model in enumerate(models):
#     LOGITS = []
#     LOGITS2 = []
#     test_dataset = ShopeeDataset(
#         df_test, image_dir=image_test_dir,
#         transform=get_transform(image_size=IMAGE_SIZE, train=False),
#     )
#     data_test_loader = DataLoader(
#         test_dataset, batch_size=BATCH_SIZE,
#         shuffle=False
#     )

#     test_dataset2 = ShopeeDataset(
#         df_test, image_dir=image_test_dir_2,
#         transform=get_transform(image_size=IMAGE_SIZE, train=False),
#     )
#     data_test_loader2 = DataLoader(
#         test_dataset2, batch_size=BATCH_SIZE,
#         shuffle=False
#     )
                            
#     with torch.no_grad():
#         for (data, targets) in tqdm(data_test_loader):
#             data = data.to(device)
#             logits = model(data)
#             LOGITS.append(logits)

#         for (data, targets) in tqdm(data_test_loader2):
#             data = data.to(device)
#             logits = model(data)
#             LOGITS2.append(logits)

#     LOGITS_TMP = (F.softmax(torch.cat(LOGITS), dim=1).cpu() + F.softmax(torch.cat(LOGITS2), dim=1).cpu()) / 2
#     LOGITS_FINAL.append(LOGITS_TMP)

# model_num = len(LOGITS_FINAL)
# PREDS = 0

# for ans in LOGITS_FINAL:
#     PREDS += ans/model_num
# PREDS = PREDS.numpy().argmax(axis=1)
# FINAL_PRED = PREDS


# In[ ]:


df_test['category'] = FINAL_PRED


# In[ ]:


df_test['category'] = df_test['category'].apply(lambda x: f'{x:02}')


# In[ ]:


df_test.to_csv('submission.csv', index=False)


# In[ ]:


df_test[:20]


# In[ ]:




