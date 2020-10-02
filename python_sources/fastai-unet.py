#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
    #    print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[ ]:


import torch
import warnings


# In[ ]:


import fastai
from fastai.vision import *
from fastai.callbacks import *
from fastai.utils.mem import *
from fastai.imports import *
from fastai.callbacks.hooks import *


# In[ ]:


from tqdm import tqdm


# In[ ]:


from datetime import datetime

warnings.filterwarnings("ignore")


# In[ ]:


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using devic', device)


# In[ ]:


ROOT_DIR = '/kaggle/input/understanding_cloud_organization/'


# In[ ]:


os.listdir(ROOT_DIR)


# In[ ]:


TRAIN_CSV_PATH = os.path.join(ROOT_DIR, 'train.csv')
train_df = pd.read_csv(TRAIN_CSV_PATH)


# In[ ]:


train_df.head()


# In[ ]:


train_df['ImageId'] = train_df['Image_Label'].apply(lambda x: x.split('_')[0])
train_df['ClassId'] = train_df['Image_Label'].apply(lambda x: x.split('_')[1])
train_df['hasMask'] = ~ train_df['EncodedPixels'].isna()


# In[ ]:


train_df.head(20)


# In[ ]:


train = train_df.pivot(index='ImageId', columns='ClassId', values='EncodedPixels')


# In[ ]:


# TODO remove use_partial_data()
item_list = (SegmentationItemList.
             from_df(df=train.reset_index(), path=os.path.join(ROOT_DIR, 'train_images'), cols="ImageId")
             .use_partial_data(sample_pct=0.1)  # use only a subset of data to speedup tests
             .split_by_rand_pct(0.2))


# In[ ]:


item_list


# In[ ]:


class MultiLabelImageSegment(ImageSegment): 
    
    def reconstruct(self, t:Tensor): 
        return MultiClassImageSegment(t)


# In[ ]:


def bce_logits_floatify(input, target, reduction='mean'):
    return F.binary_cross_entropy_with_logits(input, target.float(), reduction=reduction)


# In[ ]:


def rle_to_mask(rle, shape):
    mask_img = open_mask_rle(rle, shape)
    mask = mask_img.px.permute(0, 2, 1)
    return mask

class MultiLabelSegmentationLabelList(SegmentationLabelList):
    """Return a single image segment with all classes"""
    # adapted from https://forums.fast.ai/t/how-to-load-multiple-classes-of-rle-strings-from-csv-severstal-steel-competition/51445/2
    
    def __init__(self, items:Iterator, src_img_size=None, classes:Collection=None, **kwargs):
        super().__init__(items=items, classes=classes, **kwargs)
        self.loss_func = bce_logits_floatify
        self.src_img_size = src_img_size
        # add attributes to copy by new() 
        self.copy_new += ["src_img_size"]
    
    def open(self, rles):        
        # load mask at full resolution
        masks = torch.zeros((len(self.classes), *self.src_img_size)) # shape CxHxW
        for i, rle in enumerate(rles):
            if isinstance(rle, str):  # filter out NaNs
                masks[i] = rle_to_mask(rle, self.src_img_size)
        return MultiLabelImageSegment(masks)
    
    def analyze_pred(self, pred, thresh:float=0.0):
        # binarize masks
        return (pred > thresh).float()
    
    def reconstruct(self, t:Tensor): 
        return MultiLabelImageSegment(t)


# In[ ]:


def get_masks_rle(img):
    """Get RLE-encoded masks for this image"""
    img = img.split("/")[-1]  # get filename only
    return train.loc[img, class_names].to_list()


# In[ ]:


img_size = (84*2, 132*2)
img_size


# In[ ]:


train_img_dims = (1400, 2100)
class_names = ["Fish", "Flower", "Gravel", "Sugar"]


# In[ ]:


classes = [0, 1, 2, 3] # no need for a "void" class: if a pixel isn't in any mask, it is not labelled
item_list = item_list.label_from_func(func=get_masks_rle, label_cls=MultiLabelSegmentationLabelList, 
                                      classes=classes, src_img_size=train_img_dims)


# In[ ]:


item_list = item_list.add_test_folder(os.path.join(ROOT_DIR, 'test_images'), label="")


# In[ ]:


batch_size = 16

# TODO add data augmentation
tfms = ([], [])
# tfms = get_transforms()

item_list = item_list.transform(tfms, tfm_y=True, size=img_size)


# In[ ]:


data = (item_list
        .databunch(bs=batch_size)
        .normalize(imagenet_stats) # use same stats as pretrained model
       )  
assert data.test_ds is not None


# In[ ]:


def dice_metric(pred, targs, threshold=0):
    pred = (pred > threshold).float()
    targs = targs.float()  # make sure target is float too
    return 2.0 * (pred*targs).sum() / ((pred+targs).sum() + 1.0)


# In[ ]:


def fmt_now():
    return datetime.today().strftime('%Y%m%d-%H%M%S')
from time import time


# In[ ]:


metrics = [dice_metric]


# In[ ]:


learn = unet_learner(data, models.resnet34, metrics=metrics, wd=1e-2)
learn.model_dir = "/kaggle/working/"  # point to writable directory


# In[ ]:


critation = BCEWithLogitsFlat()
learn.loss = critation


# In[ ]:


lr=1e-4
print("training_started")


# In[ ]:


import gc
gc.collect()
free = gpu_mem_get_free_no_cache()
torch.cuda.empty_cache()


# In[ ]:


learn.fit_one_cycle(10, max_lr=lr)


# In[ ]:


learn.save(f"_unet_resnet34_stage1", return_path=True)


# In[ ]:


learn.unfreeze()


# In[ ]:


free = gpu_mem_get_free_no_cache()
free


# In[ ]:


learn.fit_one_cycle(10, slice(1e-5, 1e-4))


# In[ ]:


learn.save(f"_unet_resnet34_stage2", return_path=True)


# In[ ]:


os.listdir('../working')


# In[ ]:


preds, _ = learn.get_preds(ds_type=DatasetType.Test, with_loss=False)


# In[ ]:


def resize_pred_masks(preds, shape=(4, 350, 525)):
    """Resize predicted masks and return them as a generator"""
    for p in range(preds.shape[0]):
        mask = MultiLabelImageSegment(preds[p])
        yield mask.resize(shape)


# In[ ]:


pred_masks = resize_pred_masks(preds)


# In[ ]:


test_fnames = [p.name for p in data.test_dl.items]
len(test_fnames)


# In[ ]:


def write_submission_file(filename, test_fnames, preds, threshold=0):
    with open(filename, mode='w') as f:
        f.write("Image_Label,EncodedPixels\n")

        for img_name, masks in zip(tqdm(test_fnames), resize_pred_masks(preds)):
            binary_masks = masks.px > threshold # TODO use activation instead
            
            for class_idx, class_name in enumerate(class_names):
                rle = rle_encode(binary_masks[class_idx].numpy().T)
                f.write(f"{img_name}_{class_name},{rle}\n")

    print(f"Wrote '{f.name}'.")


# In[ ]:


submission_file = f"{fmt_now()}_submission.csv"


# In[ ]:


write_submission_file(submission_file, test_fnames, preds)


# In[ ]:





# In[ ]:





# In[ ]:


import gc
gc.collect()


# In[ ]:


free = gpu_mem_get_free_no_cache()
free


# In[ ]:


torch.cuda.empty_cache()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




