#!/usr/bin/env python
# coding: utf-8

# # Introduction
# 
# As mentioned [here](https://www.kaggle.com/c/alaska2-image-steganalysis/discussion/146610#823749), SRNet is a top NN for steganalysis. I would like to check the out-of-box performance of SRNet on this dataset. You can find the model and pretrained weight [here](https://www.kaggle.com/naivelamb/srnet-model-weight).
# 
# If you like it, please upvote the notebook and the dataset. 

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import sys
from scipy import misc
from glob import glob

from PIL import Image, ImageOps
import cv2

from tqdm.notebook import tqdm

import albumentations as albu
from albumentations.pytorch import ToTensor

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, sampler

sys.path.append('/kaggle/input/srnet-model-weight/')

from model import Srnet


# In[ ]:


BATCH_SIZE = 40
TESTPATH = '/kaggle/input/alaska2-image-steganalysis/Test/'
CHKPT = '/kaggle/input/srnet-model-weight/SRNet_model_weights.pt'

df_sub = pd.read_csv('/kaggle/input/alaska2-image-steganalysis/sample_submission.csv')


# ## Dataset and transform

# In[ ]:


def transform_test():
    transform = albu.Compose([
        albu.Normalize(mean=(0.485,0.456,0.406), std=(0.229,0.224,0.225)),
        ToTensor()
    ])
    return transform


# In[ ]:


class AlaskaDataset(Dataset):
    def __init__(self, df, data_folder, transform):
        self.df = df
        self.root = data_folder
        self._transorm = transform

    def __getitem__(self, idx):
        image_id = self.df.Id.iloc[idx]
        image_path = os.path.join(self.root, image_id)
        img = cv2.imread(image_path)
        augment = self._transorm(image=img)
        img = augment['image']
        img = torch.mean(img, axis=0, keepdim=True) # SRNet requres input image to be one channel
        return img

    def __len__(self):
        return len(self.df)


# In[ ]:


# build dataset
test_transform = transform_test()
test_dataset = AlaskaDataset(df_sub, TESTPATH, test_transform)
test_data = DataLoader(
    test_dataset,
    batch_size = BATCH_SIZE,
    num_workers=2,
    shuffle=False
    )


# # Model

# In[ ]:


model = Srnet().cuda()
ckpt = torch.load(CHKPT)
model.load_state_dict(ckpt['model_state_dict'])


# # Predict

# In[ ]:


all_outputs = []
model.eval()
with torch.no_grad():
    for inputs in tqdm(test_data):
        inputs = inputs.cuda()
        outputs = model(inputs)
        pred = outputs.data.cpu().numpy()
        # the output of SRNet is log_softmax, convert it to probability
        pred = np.exp(pred[:, 1])/ (np.exp(pred[:, 0]) + np.exp(pred[:, 1]))
        all_outputs.append(pred)
all_outputs = np.concatenate(all_outputs)


# In[ ]:


df_sub['Label']= all_outputs
df_sub.to_csv('submission.csv', index=None)


# In[ ]:


df_sub.head()


# # What to do next?
# 
# 1. Try to finetune SRNet on the provided dataset. 
# 2. Design your own network structure! 
