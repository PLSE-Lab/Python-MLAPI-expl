#!/usr/bin/env python
# coding: utf-8

# In[ ]:


get_ipython().system('pip install fastai==0.7.0 --no-deps')
get_ipython().system('pip install torch==0.4.1 torchvision==0.2.1')


# In[ ]:


from fastai.conv_learner import *
from fastai.dataset import *
from tqdm import tqdm
import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
import matplotlib.pyplot as plt
import math
import cv2

arch = resnext50
num_workers = 4
TRAIN = '../input/humpback-whale-identification/train/'
TEST = '../input/humpback-whale-identification/test/'
LABELS = '../input/humpback-whale-identification/train.csv'
SAMPLE_SUB = '../input/humpback-whale-identification/sample_submission.csv'
BBOX = '../input/generating-whale-bounding-boxes/bounding_boxes.csv'


# In[ ]:


df = pd.read_csv(LABELS).set_index('Image')
new_whale_df = df[df.Id == "new_whale"] # only new_whale dataset
train_df = df[~(df.Id == "new_whale")] # no new_whale dataset, used for training
unique_labels = np.unique(train_df.Id.values)

labels_dict = dict()
labels_list = []
for i in range(len(unique_labels)):
    labels_dict[unique_labels[i]] = i
    labels_list.append(unique_labels[i])
train_df.Id = train_df.Id.apply(lambda x: labels_dict[x])
train_labels = np.asarray(train_df.Id.values)
test_names = [f for f in os.listdir(TEST)]


# In[ ]:


train_df['image_name'] = train_df.index
tr_n = train_df['image_name'].values
# Yes, we will validate on the subset of training data
val_n = train_df['image_name']
print('Train/val:', len(tr_n), len(val_n))
print('Train classes', len(train_df.loc[tr_n].Id.unique()))
print('Val classes', len(train_df.loc[val_n].Id.unique()))


# In[ ]:


class HWIDataset(FilesDataset):
    def __init__(self, fnames, path, transform):
        self.train_df = train_df
        super().__init__(fnames, transform, path)

    def get_x(self, i):
        img = open_image(os.path.join(self.path, self.fnames[i]))
        bbox = bbox_df.loc[self.fnames[i]]
        x0, y0, x1, y1 = bbox['x0'], bbox['y0'], bbox['x1'],  bbox['y1']
        if not (x0 >= x1 or y0 >= y1):
            img = img[y0:y1, x0:x1,:]
        img = cv2.resize(img, (self.sz, self.sz))
        return img

    def get_y(self, i):
        if (self.path == TEST): return 0
        return self.train_df.loc[self.fnames[i]]['Id']

    def get_c(self):
        return len(unique_labels)


# In[ ]:


def get_data(sz, batch_size):
    """
    Read data and do augmentations
    """
    aug_tfms = []
    tfms = tfms_from_model(arch, sz, crop_type=CropType.NO, tfm_y=TfmType.NO,
                           aug_tfms=aug_tfms)
    ds = ImageData.get_ds(HWIDataset, (tr_n[:-(len(tr_n) % batch_size)], TRAIN),
                          (val_n, TRAIN), tfms, test=(test_names, TEST))
    md = ImageData("./", ds, batch_size, num_workers=num_workers, classes=None)
    return md


# In[ ]:


image_size = 448
batch_size = 8
md = get_data(image_size, batch_size)


# In[ ]:


best_th = 0.38


# In[ ]:



preds_t = np.load("../input/resnext50-sz224-b32-591/rx50_preds.npy") + np.load("../input/fork-of-two-resnext50-sz448-674/rx50_448_preds.npy")
preds_t /= 2
np.save("rx50_448_preds.npy",preds_t)


# Finally, our submission.

# In[ ]:


sample_df = pd.read_csv(SAMPLE_SUB)
sample_list = list(sample_df.Image)
labels_list = ["new_whale"]+labels_list
pred_list = [[labels_list[i] for i in p.argsort()[-5:][::-1]] for p in preds_t]
pred_dic = dict((key, value) for (key, value) in zip(md.test_ds.fnames,pred_list))
pred_list_cor = [' '.join(pred_dic[id]) for id in sample_list]
df = pd.DataFrame({'Image':sample_list,'Id': pred_list_cor})
df.to_csv('submission.csv', header=True, index=False)
df.head()


# In[ ]:




