#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))
import gc
gc.collect()

# Any results you write to the current directory are saved as output.


# In[ ]:


get_ipython().system('pip install pretrainedmodels')

get_ipython().run_line_magic('reload_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')
get_ipython().run_line_magic('matplotlib', 'inline')

get_ipython().system('pip install fastai==1.0.52')
import fastai
fastai.__version__

from fastai import *
from fastai.vision import *

from torchvision.models import *
import pretrainedmodels

from utils import *
import sys

from fastai.callbacks.tracker import EarlyStoppingCallback
from fastai.callbacks.tracker import SaveModelCallback

path = Path('../input/wiki-face-data/wiki_crop/wiki_crop/')
path.ls()


# In[ ]:


import fastai; fastai.__version__


# In[ ]:


import scipy.io
import numpy as np
import pandas as pd
from datetime import datetime, timedelta


# In[ ]:


mat = scipy.io.loadmat('../input/wiki-face-data/wiki_crop/wiki_crop/wiki.mat')
mat


# In[ ]:


columns = ["dob", "photo_taken", "full_path", "gender", "name", "face_location", 
           "face_score", "second_face_score", 'celeb_names', 'celeb_id']


# In[ ]:


instances = mat['wiki'][0][0][0].shape[1]

df = pd.DataFrame(index = range(0,instances), columns = columns)


# In[ ]:


for i in mat:
    if i == "wiki":
        current_array = mat[i][0][0]
        for j in range(len(current_array)):
            #print(columns[j],": ",current_array[j])
            df[columns[j]] = pd.DataFrame(current_array[j][0])


# In[ ]:


df.head()


# In[ ]:


df.shape


# In[ ]:


def datenum_to_datetime(datenum):
    
    try:
        days = datenum % 1
        hours = days % 1 * 24
        minutes = hours % 1 * 60
        seconds = minutes % 1 * 60
        exact_date = datetime.fromordinal(int(datenum))            + timedelta(days=int(days))            + timedelta(hours=int(hours))            + timedelta(minutes=int(minutes))            + timedelta(seconds=round(seconds))            - timedelta(days=366)
    
        return exact_date.year
    
    except(ValueError, TypeError, OverflowError):
        
        return np.nan  


# In[ ]:


df['date_of_birth'] = df['dob'].apply(datenum_to_datetime) 


# In[ ]:


df['date_of_birth'].value_counts()


# In[ ]:


df['age'] = df['photo_taken'] - df['date_of_birth']

#remove pictures does not include face
df = df[df['face_score'] != -np.inf]

#some pictures include more than one face, remove them
df = df[df['second_face_score'].isna()]

#check threshold
df = df[df['face_score'] >= 3.5]

df = df.drop(columns = ['name','face_score','second_face_score','date_of_birth','face_location'])

#some guys seem to be greater than 100. some of these are paintings. remove these old guys
df = df[df['age'] <= 100]

#some guys seem to be unborn in the data set
df = df[df['age'] > 0]


# In[ ]:


df.head()


# In[ ]:


df.shape


# In[ ]:


df['age'] = df['age'].apply(lambda x: int(x))


# In[ ]:


print(type(df['age']))
df['age'].value_counts()


# In[ ]:


df = df.drop(columns=['dob', 'photo_taken'])
df.head()


# In[ ]:


df_gender = df.drop(columns=['age', 'celeb_names', 'celeb_id'])
df_gender.head()


# In[ ]:


df_gender['gender'].value_counts()


# In[ ]:


df_gender['full_path'] = df_gender['full_path'].str.get(0)


# In[ ]:


df_gender.dropna(axis=0, inplace=True)
df_gender.shape


# In[ ]:


df_gender.head()


# In[ ]:


df_gender['gender'] = df_gender['gender'].map({0:'female', 1:'male'})


# In[ ]:


df_gender.head()


# In[ ]:


tfms = get_transforms(max_rotate= 10.,max_zoom=1., max_lighting=0.20, do_flip=False,
                      max_warp=0., xtra_tfms=[flip_lr(), brightness(change=(0.3, 0.60), p=0.7), contrast(scale=(0.5, 2), p=0.7),
                                              crop_pad(size=600, padding_mode='border', row_pct=0.,col_pct=0.),
                                              rand_zoom(scale=(1.,1.5)), rand_crop(),
                                              perspective_warp(magnitude=(-0.1,0.1)),
                                              #jitter(magnitude=(-0.05,0.05), p=0.5),
                                              symmetric_warp(magnitude=(-0.1,0.1)) ])

path = Path('../input/wiki-face-data/wiki_crop/wiki_crop/')

src = (ImageList.from_df(df_gender, path, cols=['full_path'], folder = '.')
        .split_by_rand_pct(0.2)
        .label_from_df())


# In[ ]:


data = (src.transform(tfms, resize_method=ResizeMethod.CROP, padding_mode='border', size=128)
        .databunch(bs=64).normalize(imagenet_stats))


# In[ ]:


data.show_batch(6, figsize=(12,12))


# In[ ]:


import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
import random


# In[ ]:


def resnet50(pretrained=False):
    pretrained = 'imagenet' if pretrained else None
    model = pretrainedmodels.resnet50(pretrained=pretrained)
    return model


# In[ ]:


opt_func = partial(optim.Adam, betas=(0.9,0.99), eps=1e-5)


# In[ ]:


learn = cnn_learner(data, models.resnet50, 
                    metrics = accuracy, model_dir = "/temp/model/", 
                    opt_func=opt_func, bn_wd=False,callback_fns=[ShowGraph]).mixup()


# In[ ]:


learn.freeze()
learn.lr_find()
learn.recorder.plot(suggestion = True)


# In[ ]:


lr = 5e-4


# In[ ]:


learn.fit_one_cycle(2, max_lr=slice(lr), wd=(1e-6, 1e-4, 1e-2), pct_start=0.5, callbacks=[SaveModelCallback(learn)])


# In[ ]:


learn.save('first_head_resnet')
learn.load('first_head_resnet')


# In[ ]:


learn.unfreeze()
learn.lr_find()
learn.recorder.plot(suggestion = True)


# In[ ]:


learn.fit_one_cycle(5, max_lr=slice(1e-5, lr/5), wd=(1e-6, 1e-4, 1e-2), 
                    callbacks=[SaveModelCallback(learn)], pct_start=0.5)


# In[ ]:


learn.save('first_body_resnet')
learn.load('first_body_resnet')


# In[ ]:


learn.show_results()


# In[ ]:


interp = ClassificationInterpretation.from_learner(learn)

losses,idxs = interp.top_losses()

len(data.valid_ds)==len(losses)==len(idxs)


# In[ ]:


interp.plot_top_losses(9, figsize=(15,11))


# In[ ]:


interp.plot_confusion_matrix(figsize=(12,12), dpi=60)


# In[ ]:


img = open_image('../input/picture-1/55632758_1035600193317551_7808214316078858240_n.jpg')
img


# In[ ]:


learn.predict(img)


# In[ ]:


learn.summary


# In[ ]:


print(learn.summary())


# In[ ]:


data_big = (src.transform(tfms, size=256)
        .databunch(num_workers=0).normalize(imagenet_stats))

learn.data = data_big
data_big.train_ds[0][0].shape


# In[ ]:


learn.freeze_to(-1)
learn.lr_find()
learn.recorder.plot(suggestion = True)


# In[ ]:


lr=1e-3


# In[ ]:


learn.fit_one_cycle(2, max_lr=slice(lr), wd=(1e-6, 1e-4, 1e-2), pct_start=0.5, callbacks=[SaveModelCallback(learn)])


# In[ ]:


learn.save('second_head_resnet')
learn.load('second_head_resnet')


# In[ ]:


learn.unfreeze()
learn.lr_find()
learn.recorder.plot(suggestion = True)


# In[ ]:


learn.fit_one_cycle(5, max_lr=slice(1e-6,1e-4), wd=(1e-6, 1e-4, 1e-2), callbacks=[SaveModelCallback(learn)], pct_start=0.5)


# In[ ]:


learn.save('second_body_resnet')
learn.load('second_body_resnet')


# In[ ]:


interp = ClassificationInterpretation.from_learner(learn)

losses,idxs = interp.top_losses()

len(data.valid_ds)==len(losses)==len(idxs)


# In[ ]:


learn.show_results()


# In[ ]:


interp.plot_top_losses(9, figsize=(15,11))


# In[ ]:


interp.plot_confusion_matrix(figsize=(12,12), dpi=60)


# # Fastai Hooks - Image Similarity Search

# In[ ]:


class SaveFeatures():
    features=None
    def __init__(self, m): 
        self.hook = m.register_forward_hook(self.hook_fn)
        self.features = None
    def hook_fn(self, module, input, output): 
        out = output.detach().cpu().numpy()
        if isinstance(self.features, type(None)):
            self.features = out
        else:
            self.features = np.row_stack((self.features, out))
    def remove(self): 
        self.hook.remove()


# In[ ]:


learn.model


# In[ ]:


# Second last layer of the model
learn.model[1][4]


# In[ ]:


len(data_big.train_ds.items), len(data_big.valid_ds.items)


# In[ ]:


sf = SaveFeatures(learn.model[1][4])


# In[ ]:


_= learn.get_preds(data_big.train_ds)
_= learn.get_preds(DatasetType.Valid)


# In[ ]:


len(sf.features)


# In[ ]:


type(sf.features)


# In[ ]:


img_path = [str(x) for x in (list(data_big.train_ds.items) +list(data_big.valid_ds.items))]
label = [data_big.classes[x] for x in (list(data_big.train_ds.y.items) +list(data_big.valid_ds.y.items))]
label_id = [x for x in (list(data_big.train_ds.y.items) +list(data_big.valid_ds.y.items))]


# In[ ]:


df_new = pd.DataFrame({'img_path': img_path, 'label': label, 'label_id': label_id})
df_new


# In[ ]:


array = np.array(sf.features)


# In[ ]:


x=array.tolist()


# In[ ]:


df_new['img_repr'] = x


# In[ ]:


df_new


# In[ ]:


df_new.shape


# In[ ]:


df_new


# # Using Annoy

# In[ ]:


from annoy import AnnoyIndex

f = len(df_new['img_repr'][0])
t = AnnoyIndex(f, metric='euclidean')


# In[ ]:


ntree = 100

for i, vector in enumerate(df_new['img_repr']):
    t.add_item(i, vector)
_  = t.build(ntree)


# In[ ]:


import time
def get_similar_images_annoy(img_index):
    start = time.time()
    base_img_id, base_vector, base_label  = df_new.iloc[img_index, [0,3,1]]
    similar_img_ids = t.get_nns_by_item(img_index, 8)
    end = time.time()
    print(f'{(end - start) * 1000} ms')
    return base_img_id, base_label, df_new.iloc[similar_img_ids]


# In[ ]:


base_image, base_label, similar_images_df = get_similar_images_annoy(500)


# In[ ]:


print(base_label)
open_image(base_image)


# In[ ]:


similar_images_df


# In[ ]:


def show_similar_images(similar_images_df):
    images = [open_image(img_id) for img_id in similar_images_df['img_path']]
    categories = [learn.data.train_ds.y.reconstruct(y) for y in similar_images_df['label_id']]
    return learn.data.show_xys(images, categories)


# In[ ]:


show_similar_images(similar_images_df)


# In[ ]:


learn.save("/kaggle/working/gender-pred-wiki")
from IPython.display import FileLinks
FileLinks('.')

