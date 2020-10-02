#!/usr/bin/env python
# coding: utf-8

# In[ ]:


get_ipython().system(' python ../input/mlcomp/mlcomp/mlcomp/setup.py')


# In[ ]:


import os
import json
import gc

import cv2
import keras
from keras import backend as K
from keras import layers
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Model, load_model
from keras.layers import Input
from keras.layers.convolutional import Conv2D, Conv2DTranspose
from keras.layers.pooling import MaxPooling2D
from keras.layers.merge import concatenate
from keras.optimizers import Adam
from keras.callbacks import Callback, ModelCheckpoint
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import train_test_split

import warnings
warnings.filterwarnings('ignore')
import os
import matplotlib.pyplot as plt


# In[ ]:


import numpy as np
import cv2
import albumentations as A
from tqdm import tqdm_notebook
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.jit import load

from mlcomp.contrib.transform.albumentations import ChannelTranspose
from mlcomp.contrib.dataset.classify import ImageDataset
from mlcomp.contrib.transform.rle import rle2mask, mask2rle
from mlcomp.contrib.transform.tta import TtaWrap


# In[ ]:


unet_se_resnext50_32x4d =     load('../input/severstal-mlcomp-catalyst-train-0-90672-offline/unet_se_resnext50_32x4d.pth').cuda()
unet_mobilenet2 = load('../input/severstal-mlcomp-catalyst-train-0-90672-offline/unet_mobilenet2.pth').cuda()
unet_resnet34 = load('../input/severstal-mlcomp-catalyst-train-0-90672-offline/unet_resnet34.pth').cuda()

# unet_se_resnext50_32x4d = \
#     load('../input/severstal-mlcomp-catalyst-train-0-90672-offline/unet_se_resnext50_32x4d.pth')
# unet_mobilenet2 = load('../input/severstal-mlcomp-catalyst-train-0-90672-offline/unet_mobilenet2.pth')
# unet_resnet34 = load('../input/severstal-mlcomp-catalyst-train-0-90672-offline/unet_resnet34.pth')


# # Preprocessing

# In[ ]:


# os.listdir('../input/severstal-mlcomp-catalyst-train-0-90672-offline')


# In[ ]:


train_df = pd.read_csv('../input/severstal-steel-defect-detection/train.csv')
train_df['ImageId'] = train_df['ImageId_ClassId'].apply(lambda x: x.split('_')[0])
train_df['ClassId'] = train_df['ImageId_ClassId'].apply(lambda x: x.split('_')[1])
train_df['hasMask'] = ~ train_df['EncodedPixels'].isna()

print(train_df.shape)
train_df.head()


# In[ ]:


mask_count_df = train_df.groupby('ImageId').agg(np.sum).reset_index()
mask_count_df.sort_values('hasMask', ascending=False, inplace=True)
print(mask_count_df.shape)
mask_count_df.head()


# In[ ]:


sub_df = pd.read_csv('../input/severstal-steel-defect-detection/sample_submission.csv')
sub_df['ImageId'] = sub_df['ImageId_ClassId'].apply(lambda x: x.split('_')[0])
test_imgs = pd.DataFrame(sub_df['ImageId'].unique(), columns=['ImageId'])
test_imgs.head()


# In[ ]:


non_missing_train_idx = mask_count_df[mask_count_df['hasMask'] > 0]
non_missing_train_idx.head()


# # Step 1: Remove test images without defects
# 
# Most of the stuff below is hidden, since it's copied from my previous kernels.

# In[ ]:


def load_img(code, base, resize=True):
    path = f'{base}/{code}'
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    if resize:
        img = cv2.resize(img, (256, 256))
    
    return img

def validate_path(path):
    if not os.path.exists(path):
        os.makedirs(path)


# In[ ]:


BATCH_SIZE = 64
def create_test_gen():
    return ImageDataGenerator(rescale=1/255.).flow_from_dataframe(
        test_imgs,
        directory='../input/severstal-steel-defect-detection/test_images',
        x_col='ImageId',
        class_mode=None,
        target_size=(256, 256),
        batch_size=BATCH_SIZE,
        shuffle=False
    )

test_gen = create_test_gen()


# In[ ]:


remove_model = load_model('../input/severstal-predict-missing-masks/model.h5')
remove_model.summary()


# In[ ]:


test_missing_pred = remove_model.predict_generator(
    test_gen,
    steps=len(test_gen),
    verbose=1
)

test_imgs['allMissing'] = test_missing_pred
test_imgs.head()


# In[ ]:


filtered_test_imgs = test_imgs[test_imgs['allMissing'] < 0.9996]
print(filtered_test_imgs.shape)
filtered_test_imgs.head()


# `filtered_sub_df` contains all of the images with at least one mask. `null_sub_df` contains all the images with exactly 4 missing masks.

# In[ ]:


filtered_mask = sub_df['ImageId'].isin(filtered_test_imgs["ImageId"].values)
filtered_sub_df = sub_df[filtered_mask].copy()
null_sub_df = sub_df[~filtered_mask].copy()
null_sub_df['EncodedPixels'] = null_sub_df['EncodedPixels'].apply(
    lambda x: ' ')

filtered_sub_df.reset_index(drop=True, inplace=True)
filtered_test_imgs.reset_index(drop=True, inplace=True)

print(filtered_sub_df.shape)
print(null_sub_df.shape)

# filtered_sub_df.head()


# # step 2: severstal-mlcomp-catalyst-infer-0-90672

# In[ ]:


class Model:
    def __init__(self, models):
        self.models = models
    
    def __call__(self, x):
        res = []
        x = x.cuda()
        with torch.no_grad():
            for m in self.models:
                res.append(m(x))
        res = torch.stack(res)
        return torch.mean(res, dim=0)

model = Model([unet_se_resnext50_32x4d, unet_mobilenet2, unet_resnet34])

def create_transforms(additional):
    res = list(additional)
    # add necessary transformations
    res.extend([
        A.Normalize(
            mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)
        ),
        ChannelTranspose()
    ])
    res = A.Compose(res)
    return res

img_folder = '/kaggle/input/severstal-steel-defect-detection/test_images'
batch_size = 2
num_workers = 0

# Different transforms for TTA wrapper
transforms = [
    [],
    [A.HorizontalFlip(p=1)]
]

transforms = [create_transforms(t) for t in transforms]
datasets = [TtaWrap(ImageDataset(img_folder=img_folder, transforms=t), tfms=t) for t in transforms]
loaders = [DataLoader(d, num_workers=num_workers, batch_size=batch_size, shuffle=False) for d in datasets]


# In[ ]:


thresholds = [0.5, 0.5, 0.5, 0.5]
min_area = [600, 600, 1000, 2000]

res = []
# Iterate over all TTA loaders
total = len(datasets[0])//batch_size
for loaders_batch in tqdm_notebook(zip(*loaders), total=total):
    preds = []
    image_file = []
    for i, batch in enumerate(loaders_batch):
#         features = batch['features']
        features = batch['features'].cuda()
        p = torch.sigmoid(model(features))
        # inverse operations for TTA
        p = datasets[i].inverse(p)
        preds.append(p)
        image_file = batch['image_file']
    
    # TTA mean
    preds = torch.stack(preds)
    preds = torch.mean(preds, dim=0)
    preds = preds.detach().cpu().numpy()
    
    # Batch post processing
    for p, file in zip(preds, image_file):
        file = os.path.basename(file)
        # Image postprocessing
        for i in range(4):
            p_channel = p[i]
            imageid_classid = file+'_'+str(i+1)
            p_channel = (p_channel>thresholds[i]).astype(np.uint8)
            if p_channel.sum() < min_area[i]:
                p_channel = np.zeros(p_channel.shape, dtype=p_channel.dtype)
            
            res.append({
                'ImageId_ClassId': imageid_classid,
                'EncodedPixels': mask2rle(p_channel)
            })


# ensemble

# In[ ]:


df = pd.DataFrame(res)


# In[ ]:


null_sub_df.shape


# In[ ]:


print((df.loc[df['ImageId_ClassId'].isin(list(null_sub_df["ImageId_ClassId"])),'EncodedPixels'] != '').sum() )


# In[ ]:


df.loc[df['ImageId_ClassId'].isin(list(null_sub_df["ImageId_ClassId"])), 'EncodedPixels'] = ''


# In[ ]:


df.to_csv('submission.csv', index=False)	

