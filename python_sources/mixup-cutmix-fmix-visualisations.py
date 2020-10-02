#!/usr/bin/env python
# coding: utf-8

# In this kernel we visualize Mixed Sample Data Augmentations (MSDAs) like Mixup, Cutmix, Fmix. We first visualize them individually and then altogether aswell.
# As mentioned by The Zoo in their writeup, they have used FMix. They found FMix to work better than Cutmix and the images were also more natural than Cutmix as quoted by them.
# 
# Please upvote the kernel if you find the visaulizations helpful.
# 

# In[ ]:


from os.path import exists
if not exists('fmix.zip'):
    get_ipython().system('wget -O fmix.zip https://github.com/ecs-vlc/fmix/archive/master.zip')
    get_ipython().system('unzip -qq fmix.zip')
    get_ipython().system('mv FMix-master/* ./')
    get_ipython().system('rm -r FMix-master')


# In[ ]:


import os
import sys
sys.path.append('../input/efficientnet-pytorch/EfficientNet-PyTorch/EfficientNet-PyTorch-master')


# In[ ]:


import glob
import math
import time
import numpy as np
import pandas as pd
import cv2
import PIL.Image
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
from tqdm import tqdm_notebook as tqdm
import torch
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
import torch.nn.functional as F
import albumentations
from fmix import  sample_mask


device = torch.device('cpu')


# In[ ]:


HEIGHT = 137
WIDTH = 236

data_dir = '../input/bengaliaicv19feather'
batch_size = 512
num_workers = 4

df_train = pd.read_csv(os.path.join('../input/bengaliai-cv19', f'train.csv'))


# # Data & Dataset

# In[ ]:


def read_data(files):
    tmp = []
    for f in files:
        F = os.path.join(data_dir, f)
        data = pd.read_feather(F)
        res = data.iloc[:, 1:].values
        imgs = []
        for i in tqdm(range(res.shape[0])):
            img = res[i].squeeze().reshape(HEIGHT, WIDTH)
#             img = cv2.resize(img, (137, 236))
            imgs.append(img)
        imgs = np.asarray(imgs)
        
        tmp.append(imgs)
    tmp = np.concatenate(tmp, 0)
    return tmp


class BengaliDataset(Dataset):
    def __init__(self, csv, data, idx, split, mode, transform=None):

        self.csv = csv.reset_index()
        self.data = data
        self.idx = np.asarray(idx)
        self.split = split
        self.mode = mode
        self.transform = transform

    def __len__(self):
        return self.idx.shape[0]

    def __getitem__(self, index):
        index = self.idx[index]
        this_img_id = self.csv.iloc[index].image_id
        
        image = self.data[index]
        image = 255 - image

        if self.transform is not None:
            image_origin = image.astype(np.float32).copy()
            res = self.transform(image=image)
            image = res['image'].astype(np.float32)
        else:
            image_origin = image.astype(np.float32).copy()
            image = image.astype(np.float32)

        image /= 255
        image = image[np.newaxis, :, :]
        image = np.repeat(image, 3, 0)  # 1ch to 3ch
        ###
        image_origin /= 255
        image_origin = image_origin[np.newaxis, :, :]
        image_origin = np.repeat(image_origin, 3, 0)  # 1ch to 3ch
        ###

        if self.mode == 'test':
            return torch.tensor(image)
        else:
            label_1 = self.csv.iloc[index].grapheme_root
            label_2 = self.csv.iloc[index].vowel_diacritic
            label_3 = self.csv.iloc[index].consonant_diacritic
            label = [label_1, label_2, label_3]
            return torch.tensor(image), torch.tensor(image_origin), torch.tensor(label)


# In[ ]:


img_data = read_data(['train_image_data_0.feather'])


# In[ ]:


def rand_bbox(size, lam):
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w = np.int(W * cut_rat)
    cut_h = np.int(H * cut_rat)

    # uniform
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)
    return bbx1, bby1, bbx2, bby2

def cutmix(data, target, alpha):
    indices = torch.randperm(data.size(0))
    shuffled_data = data[indices]
    shuffled_target = target[indices]

    lam = np.clip(np.random.beta(alpha, alpha),0.3,0.4)
    bbx1, bby1, bbx2, bby2 = rand_bbox(data.size(), lam)
    new_data = data.clone()
    new_data[:, :, bby1:bby2, bbx1:bbx2] = data[indices, :, bby1:bby2, bbx1:bbx2]
    # adjust lambda to exactly match pixel ratio
    lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (data.size()[-1] * data.size()[-2]))
    targets = (target, shuffled_target, lam)

    return new_data, targets

def mixup(data, target, alpha):
    indices = torch.randperm(data.size(0))
    shuffled_data = data[indices]
    shuffled_target = target[indices]

    lam = np.clip(np.random.beta(alpha, alpha),0.3,0.7)
    data = lam*data + (1-lam)*shuffled_data
    targets = (target, shuffled_target, lam)

    return data, targets


def fmix(data, targets, alpha, decay_power, shape, max_soft=0.0):
    lam, mask = sample_mask(alpha, decay_power, shape, max_soft)
    indices = torch.randperm(data.size(0)).to(device)
    shuffled_data = data[indices]
    shuffled_targets = targets[indices]
    x1 = torch.from_numpy(mask)*data
    x2 = torch.from_numpy(1-mask)*shuffled_data
    targets=(targets, shuffled_targets, lam)
    
    return (x1+x2), targets


# # Mixup

# Two images are mixed wiht weights: lambda and 1-lambda. lambda is generated from symmetric beta distribution with parameter alpha. There is no cutting and pasting involved in Mixup. 
# Below are some images generated from Mixup. 1st and 3rd column contain original image. 2nd and 4th column contain their respective mixedup versions.
# 
# Here is the link to the paper for details: https://arxiv.org/abs/1710.09412 

# In[ ]:


df_show = df_train.iloc[:1000]
dataset_show = BengaliDataset(df_show, img_data, list(range(df_show.shape[0])), 'train', 'train', transform=None)
dataloader = torch.utils.data.DataLoader(dataset_show, batch_size=batch_size,num_workers=num_workers, shuffle=False)
iter_data = iter(dataloader)
data, data_org, target = next(iter_data)
data, target = mixup(data_org, target, 1.)


from pylab import rcParams
rcParams['figure.figsize'] = 20,10
for i in range(3):
    f, axarr = plt.subplots(1,4)
    for p in range(0,3,2):
        idx = np.random.randint(0, len(data))
        img_org = data_org[idx]
        new_img = data[idx]
        axarr[p].imshow(img_org.permute(1,2,0))
        axarr[p+1].imshow(new_img.permute(1,2,0))
        axarr[p].set_title('original')
        axarr[p+1].set_title('mixup image')
        axarr[p].axis('off')
        axarr[p+1].axis('off')


# # Cutmix

# 1. Cutmix involves cutting a **rectangular** portion of a random image and then pasting it onto the concerned image at the same spot from where the portion was cut. How is the size of the portion decided? 
# 2. For that, lambda is generated from symmetric beta distribution with parameter alpha. Cut fraction - denoted by 'cut_rat' in code - is then calulated as square root of 1-lambda. 
# 3. Lengths of sides of rectangles are then calculated by multiplying that fraction with the height and weight of an image. A random (x,y) coordinate is generated from uniform distribution with higher limit as width and height. This (x,y) coordinate is then the centre of the rectangular portion to be cut. 
# 4. The boundary coordinates are then obtained by subtracting and adding length/2 to the centre 'x' coordinate and also breadth/2 to the centre 'y' coordinate. Thus we have 4 coordinates namely, (bbx1,cy), (bbx2,cy), (bby1,cx), (bby2,cx); each of them is on the centre of the rectangular sides. Thus, a rectangular portion to be cut is generated.
# 
# Here is the link to the paper for details: https://arxiv.org/abs/1905.04899

# In[ ]:


df_show = df_train.iloc[:1000]
dataset_show = BengaliDataset(df_show, img_data, list(range(df_show.shape[0])), 'train', 'train', transform=None)
dataloader = torch.utils.data.DataLoader(dataset_show, batch_size=batch_size,num_workers=num_workers, shuffle=False)
iter_data = iter(dataloader)
data, data_org, target = next(iter_data)
data, target = cutmix(data_org, target, 1.)


from pylab import rcParams
rcParams['figure.figsize'] = 20,40
for i in range(3):
    f, axarr = plt.subplots(1,4)
    for p in range(0,3,2):
        idx = np.random.randint(0, len(data))
        img_org = data_org[idx]
        new_img = data[idx]
        axarr[p].imshow(img_org.permute(1,2,0))
        axarr[p+1].imshow(new_img.permute(1,2,0))
        axarr[p].set_title('original')
        axarr[p+1].set_title('cutmix image')
        axarr[p].axis('off')
        axarr[p+1].axis('off')


# # FMix

# 1. FMix involves cutting **random shaped** portion from a random images and pasting it onto the concerned image.
# 2. It doesn't involve explicit cutting and pasting, but there is generation of masks which define which portions of the image to take into consideration. 
# 3. Masks are obtained by applying threshold to low frequency images sampled from Fourier space. 
# 
# Here is the link to the paper for further details: https://arxiv.org/abs/2002.12047

# In[ ]:


df_show = df_train.iloc[:1000]
dataset_show = BengaliDataset(df_show, img_data, list(range(df_show.shape[0])), 'train', 'train', transform=None)
dataloader = torch.utils.data.DataLoader(dataset_show, batch_size=batch_size,num_workers=num_workers, shuffle=False)
iter_data = iter(dataloader)
data, data_org, target = next(iter_data)
data, target = fmix(data_org, target,alpha=1., decay_power=3., shape=(137,236))


from pylab import rcParams
rcParams['figure.figsize'] = 20,10
for i in range(3):
    f, axarr = plt.subplots(1,4)
    for p in range(0,3,2):
        idx = np.random.randint(0, len(data))
        img_org = data_org[idx]
        new_img = data[idx]
        axarr[p].imshow(img_org.permute(1,2,0))
        axarr[p+1].imshow(new_img.permute(1,2,0))
        axarr[p].set_title('original')
        axarr[p+1].set_title('fmix image')
        axarr[p].axis('off')
        axarr[p+1].axis('off')


# # All together

# In[ ]:


df_show = df_train.iloc[:1000]
dataset_show = BengaliDataset(df_show, img_data, list(range(df_show.shape[0])), 'train', 'train', transform=None)
dataloader = torch.utils.data.DataLoader(dataset_show, batch_size=batch_size,num_workers=num_workers, shuffle=False)
iter_data = iter(dataloader)
data, data_org, target_org = next(iter_data)
fmix_data, target = fmix(data_org, target_org,alpha=1., decay_power=3., shape=(137,236))
cutmix_data, target = cutmix(data_org, target_org, 1.)
mixup_data, target = mixup(data_org, target_org, 1.)

from pylab import rcParams
rcParams['figure.figsize'] = 20,10
for i in range(5):
    f, axarr = plt.subplots(1,4)
    idx = np.random.randint(0, len(data))
    img_org = data_org[idx]
    mix_img = mixup_data[idx]
    cut_img = cutmix_data[idx]
    fmix_img = fmix_data[idx] 
    axarr[0].imshow(img_org.permute(1,2,0))
    axarr[1].imshow(mix_img.permute(1,2,0))
    axarr[2].imshow(cut_img.permute(1,2,0))
    axarr[3].imshow(fmix_img.permute(1,2,0))
    axarr[0].set_title('original')
    axarr[1].set_title('mixup image')
    axarr[2].set_title('cutmix image')
    axarr[3].set_title('fmix image')
    axarr[0].axis('off')
    axarr[1].axis('off')
    axarr[2].axis('off')
    axarr[3].axis('off')

