#!/usr/bin/env python
# coding: utf-8

# This kernel is forked from [mask-rcnn with augmentation and multiple masks](https://www.kaggle.com/abhishek/mask-rcnn-with-augmentation-and-multiple-masks)
# 
# # Import Cool Stuff

# In[ ]:


from __future__ import print_function

from collections import defaultdict, deque
import datetime
import pickle
import time
import torch.distributed as dist
import errno
from fastai import metrics

import cv2
import collections
import os
import numpy as np
import torch
import torch.utils.data
from PIL import Image, ImageFile
import pandas as pd
from tqdm import tqdm_notebook as tqdm
from torchvision import transforms
import torchvision
import random

ImageFile.LOAD_TRUNCATED_IMAGES = True


# In[ ]:


import platform
print(f'Python version: {platform.python_version()}')
print(f'PyTorch version: {torch.__version__}')


# In[ ]:


def seed_everything(seed=73):
    '''
      Make PyTorch deterministic.
    '''    
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True


# In[ ]:


seed_everything()


# In[ ]:


IS_DEBUG = False


# # Utility Functions (hidden)

# In[ ]:


def warmup_lr_scheduler(optimizer, warmup_iters, warmup_factor):

    def f(x):
        if x >= warmup_iters:
            return 1
        alpha = float(x) / warmup_iters
        return warmup_factor * (1 - alpha) + alpha

    return torch.optim.lr_scheduler.LambdaLR(optimizer, f)


# ## Dice Loss

# In[ ]:


class DiceLoss(torch.nn.Module):
    def __init__(self):
        super(DiceLoss, self).__init__()
 
    def forward(self, logits, targets):
        ''' fastai.metrics.dice uses argmax() which is not differentiable, so it 
          can NOT be used in training, however it can be used in prediction.
          see https://github.com/fastai/fastai/blob/master/fastai/metrics.py#L53
        '''
        N = targets.size(0)
        preds = torch.sigmoid(logits)
        #preds = logits.argmax(dim=1) # do NOT use argmax in training, because it is NOT differentiable
        # https://github.com/tensorflow/tensorflow/blob/r1.12/tensorflow/python/keras/backend.py#L96
        EPSILON = 1e-7
 
        preds_flat = preds.view(N, -1)
        targets_flat = targets.view(N, -1)
 
        intersection = (preds_flat * targets_flat).sum()#.float()
        union = (preds_flat + targets_flat).sum()#.float()
        
        loss = (2.0 * intersection + EPSILON) / (union + EPSILON)
        loss = 1 - loss / N
        return loss


# # Training Function

# In[ ]:


import torch.nn.functional as F

def train_one_epoch(model, optimizer, data_loader, device, epoch):
    model.train()
    loss_func = DiceLoss()

    lr_scheduler = None
    if epoch == 0:
        warmup_factor = 1. / 1000
        warmup_iters = min(1000, len(data_loader) - 1)

        lr_scheduler = warmup_lr_scheduler(optimizer, warmup_iters, warmup_factor)

    lossf=None
    inner_tq = tqdm(data_loader, total=len(data_loader), leave=False, desc= f'Iteration {epoch}')
    for images, masks in inner_tq:
        y_preds = model(images.to(device))
        y_preds = y_preds['out'][:, 1, :, :] #

        loss = loss_func(y_preds, masks.to(device))

        if torch.cuda.device_count() > 1:
            loss = loss.mean() # mean() to average on multi-gpu.

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if lr_scheduler is not None:
            lr_scheduler.step()

        if lossf:
            lossf = 0.98*lossf+0.02*loss.item()
        else:
            lossf = loss.item()
        inner_tq.set_postfix(loss = lossf)


# # RLE to Mask

# In[ ]:


def rle2mask(rle, width, height):
    mask= np.zeros(width * height, dtype=np.uint8)
    array = np.asarray([int(x) for x in rle.split()])
    starts = array[0::2]
    lengths = array[1::2]

    current_position = 0
    for index, start in enumerate(starts):
        current_position += start
        mask[current_position:current_position+lengths[index]] = 1
        current_position += lengths[index]

    return mask.reshape(width, height)


# # SIIM Dataset Class

# In[ ]:


class SIIMDataset(torch.utils.data.Dataset):
    def __init__(self, df_path, img_dir):
        self.df = pd.read_csv(df_path)
        self.df = self.df[self.df[' EncodedPixels'] != ' -1']
        if IS_DEBUG:
            self.df = self.df.sample(frac=0.01, random_state=73)
        self.height = 1024
        self.width = 1024
        self.image_dir = img_dir
        self.image_info = collections.defaultdict(dict)

        counter = 0
        for index, row in tqdm(self.df.iterrows(), total=len(self.df)):
            image_id = row['ImageId']
            image_path = os.path.join(self.image_dir, image_id)
            if os.path.exists(image_path + '.png') and row[" EncodedPixels"].strip() != "-1":
                self.image_info[counter]["image_id"] = image_id
                self.image_info[counter]["image_path"] = image_path
                self.image_info[counter]["annotations"] = row[" EncodedPixels"].strip()
                counter += 1

    def __getitem__(self, idx):
        img_path = self.image_info[idx]["image_path"]
        img = Image.open(img_path + '.png').convert("RGB")
        width, height = img.size
        info = self.image_info[idx]

        mask = rle2mask(info['annotations'], width, height)
        mask = mask.T
#         mask = np.expand_dims(mask, axis=0)
        mask = torch.as_tensor(mask, dtype=torch.float)

        img = transforms.ToTensor()(img)
        
        return img, mask

    def __len__(self):
        return len(self.image_info)


# # Create Dataset

# In[ ]:


dataset_train = SIIMDataset("../input/siim-dicom-images/train-rle.csv", "../input/siim-png-images/input/train_png")


# In[ ]:


len(dataset_train)


# # Create DeepLabV3 Model

# In[ ]:


model_ft = torchvision.models.segmentation.deeplabv3_resnet50(pretrained=False, num_classes=2)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_ft.to(device)
NUM_GPUS = torch.cuda.device_count()
if NUM_GPUS > 1:
    model_ft = torch.nn.DataParallel(model_ft)
_ = model_ft.to(device)


# # Create Data Loader

# In[ ]:


data_loader = torch.utils.data.DataLoader(
    dataset_train, batch_size=2*NUM_GPUS, shuffle=True, num_workers=NUM_GPUS,drop_last=True
)


# # Define Training Parameters

# In[ ]:


# construct an optimizer
params = [p for p in model_ft.parameters() if p.requires_grad]
optimizer = torch.optim.SGD(params, lr=0.001, momentum=0.9, weight_decay=0.0005)


# In[ ]:


# and a learning rate scheduler
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                               step_size=5,
                                               gamma=0.1)


# # Train Model

# In[ ]:


num_epochs = 5
for epoch in range(num_epochs):
    train_one_epoch(model_ft, optimizer, data_loader, device, epoch)
    lr_scheduler.step()


# # Mask to RLE helper

# In[ ]:


def mask_to_rle(img, width, height):
    rle = []
    lastColor = 0
    currentPixel = 0
    runStart = -1
    runLength = 0

    for x in range(width):
        for y in range(height):
            currentColor = img[x][y]
            if currentColor != lastColor:
                if currentColor == 1:
                    runStart = currentPixel
                    runLength = 1
                else:
                    rle.append(str(runStart))
                    rle.append(str(runLength))
                    runStart = -1
                    runLength = 0
                    currentPixel = 0
            elif runStart > -1:
                runLength += 1
            lastColor = currentColor
            currentPixel+=1
    # https://www.kaggle.com/c/siim-acr-pneumothorax-segmentation/discussion/98317
    if lastColor == 255:
        rle.append(runStart)
        rle.append(runLength)
    return " " + " ".join(rle)


# # Convert Model to Evaluation Mode

# In[ ]:


model_ft.eval()
for param in model_ft.parameters():
    param.requires_grad = False
model_ft.to(torch.device('cuda'))
assert model_ft.training == False


# In[ ]:


torch.save(model_ft.state_dict(), 'deeplabv3.pth')
torch.cuda.empty_cache()


# # Get Test Data

# In[ ]:


sample_df = pd.read_csv("../input/siim-acr-pneumothorax-segmentation/sample_submission.csv")

# this part was taken from @raddar's kernel: https://www.kaggle.com/raddar/better-sample-submission
masks_ = sample_df.groupby('ImageId')['ImageId'].count().reset_index(name='N')
masks_ = masks_.loc[masks_.N > 1].ImageId.values
###
sample_df = sample_df.drop_duplicates('ImageId', keep='last').reset_index(drop=True)


# In[ ]:


tt = transforms.ToTensor()
sublist = []
counter = 0
threshold = 0.5
for index, row in tqdm(sample_df.iterrows(), total=len(sample_df)):
    image_id = row['ImageId']
    if image_id in masks_:
        img_path = os.path.join('../input/siim-png-images/input/test_png', image_id + '.png')

        img = Image.open(img_path).convert("RGB")
        width, height = img.size
        img = img.resize((1024, 1024), resample=Image.BILINEAR)
        img = tt(img)
        img = img.reshape((1, *img.numpy().shape))
        logits = model_ft(img.to(device))['out'][:, 1, :, :]
        preds = torch.sigmoid(logits)[0].cpu().numpy()
        mask = (preds > 0.5).astype(np.uint8).T
        if np.count_nonzero(mask) == 0:
            rle = " -1"
        else:
            rle = mask_to_rle(mask, width, height)
    else:
        rle = " -1"
    sublist.append([image_id, rle])

submission_df = pd.DataFrame(sublist, columns=sample_df.columns.values)
submission_df.to_csv("submission.csv", index=False)
print(counter)

