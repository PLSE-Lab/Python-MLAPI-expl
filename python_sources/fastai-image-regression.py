#!/usr/bin/env python
# coding: utf-8

# # Introduction
# 
# In this Notebook, I have tried to build a model which predicts the age of a person based on his or her face.
# 
# The techniques I have used to build this **Image Regression model** are based on Fastai's CNN  models.
# 
# The dataset I have used for this task are as follows:
# 
# 1. UTK Face Data
# 2. Appa Real Face Data
# 3. IMDB Wiki Face Data
# 
# In the course of this analysis, I have learnt many things:
# 
# 1. How to augment the data using Fastai's Image augmentation techinues.
# 2. Loading and using Cadene's Pretrained Models (https://github.com/Cadene/pretrained-models.pytorch)
# 3. How to use Image resizing technique which basically refers to gradually increase the size of image while training. This helps in achieving greater prediction accuracy.
# 4. Discriminative Layers Training technique

# ## Import Libraries

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

# Any results you write to the current directory are saved as output.

import warnings
warnings.filterwarnings('ignore')
import gc
gc.collect()


# In[ ]:


get_ipython().system('pip install pretrainedmodels')

get_ipython().run_line_magic('reload_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')
get_ipython().run_line_magic('matplotlib', 'inline')

get_ipython().system('pip install fastai==1.0.57')
import fastai

from fastai import *
from fastai.vision import *

from torchvision.models import *
import pretrainedmodels

from utils import *
import sys

from fastai.callbacks.hooks import *

from fastai.callbacks.tracker import EarlyStoppingCallback
from fastai.callbacks.tracker import SaveModelCallback

path_wiki = Path('../input/wiki-face-data/wiki_crop/wiki_crop/')
path_imdb = Path('../input/imdb-wiki-faces-dataset/imdb_crop/imdb_crop/')


# In[ ]:


import scipy.io
import numpy as np
import pandas as pd
from datetime import datetime, timedelta


# ## Loading the Data

# In[ ]:


mat_wiki = scipy.io.loadmat('../input/wiki-face-data/wiki_crop/wiki_crop/wiki.mat')
mat_imdb = scipy.io.loadmat('../input/imdb-wiki-faces-dataset/imdb_crop/imdb_crop/imdb.mat')


# In[ ]:


columns = ["dob", "photo_taken", "full_path", "gender", "name", "face_location", 
           "face_score", "second_face_score", 'celeb_names', 'celeb_id']


# In[ ]:


instances_wiki = mat_wiki['wiki'][0][0][0].shape[1]
instances_imdb = mat_imdb['imdb'][0][0][0].shape[1]


df_wiki = pd.DataFrame(index = range(0,instances_wiki), columns = columns)
df_imdb = pd.DataFrame(index = range(0,instances_imdb), columns = columns)


# In[ ]:


for i in mat_wiki:
    if i == "wiki":
        current_array = mat_wiki[i][0][0]
        for j in range(len(current_array)):
            #print(columns[j],": ",current_array[j])
            df_wiki[columns[j]] = pd.DataFrame(current_array[j][0])
            

for i in mat_imdb:
    if i == "imdb":
        current_array = mat_imdb[i][0][0]
        for j in range(len(current_array)):
            #print(columns[j],": ",current_array[j])
            df_imdb[columns[j]] = pd.DataFrame(current_array[j][0])


# In[ ]:


df_wiki.head()


# In[ ]:


df_imdb.head()


# In[ ]:


df_wiki.shape, df_imdb.shape


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


df_wiki['date_of_birth'] = df_wiki['dob'].apply(datenum_to_datetime) 
df_imdb['date_of_birth'] = df_imdb['dob'].apply(datenum_to_datetime) 


# In[ ]:


df_wiki['date_of_birth'].value_counts()
df_imdb['date_of_birth'].value_counts()


# In[ ]:


df_wiki['age'] = df_wiki['photo_taken'] - df_wiki['date_of_birth']

#remove pictures does not include face
df_wiki = df_wiki[df_wiki['face_score'] != -np.inf]

#some pictures include more than one face, remove them
df_wiki = df_wiki[df_wiki['second_face_score'].isna()]

#check threshold
df_wiki = df_wiki[df_wiki['face_score'] >= 3.5]

df_wiki = df_wiki.drop(columns = ['name','face_score','second_face_score','date_of_birth','face_location'])

#some guys seem to be greater than 100. some of these are paintings. remove these old guys
df_wiki = df_wiki[df_wiki['age'] <= 100]

#some guys seem to be unborn in the data set
df_wiki = df_wiki[df_wiki['age'] > 0]


# In[ ]:


df_imdb['age'] = df_imdb['photo_taken'] - df_imdb['date_of_birth']

#remove pictures does not include face
df_imdb = df_imdb[df_imdb['face_score'] != -np.inf]

#some pictures include more than one face, remove them
df_imdb = df_imdb[df_imdb['second_face_score'].isna()]

#check threshold
df_imdb = df_imdb[df_imdb['face_score'] >= 3.5]

df_imdb = df_imdb.drop(columns = ['name','face_score','second_face_score','date_of_birth','face_location'])

#some guys seem to be greater than 100. some of these are paintings. remove these old guys
df_imdb = df_imdb[df_imdb['age'] <= 100]

#some guys seem to be unborn in the data set
df_imdb = df_imdb[df_imdb['age'] > 0]


# In[ ]:


df_wiki.head()


# In[ ]:


df_wiki.shape, df_imdb.shape


# In[ ]:


df_wiki['age'] = df_wiki['age'].apply(lambda x: int(x))
df_imdb['age'] = df_imdb['age'].apply(lambda x: int(x))


# In[ ]:


print(type(df_wiki['age']))
df_wiki['age'].value_counts()
df_imdb['age'].value_counts()


# In[ ]:


df_wiki = df_wiki.drop(columns=['dob', 'photo_taken'])
df_imdb = df_imdb.drop(columns=['dob', 'photo_taken'])


# In[ ]:


df_age_wiki = df_wiki.drop(columns=['gender', 'celeb_names', 'celeb_id'])
df_age_imdb = df_imdb.drop(columns=['gender', 'celeb_names', 'celeb_id'])


# In[ ]:


df_age_wiki['age'].value_counts()


# In[ ]:


df_imdb['age'].value_counts()


# In[ ]:


df_age_wiki['full_path'] = df_age_wiki['full_path'].str.get(0)
df_age_imdb['full_path'] = df_age_imdb['full_path'].str.get(0)


# In[ ]:


df_age_wiki.dropna(axis=0, inplace=True)
df_age_imdb.dropna(axis=0, inplace=True)


# In[ ]:


df_age_wiki.head()


# In[ ]:


df_age_imdb.head()


# In[ ]:


df_age_wiki['age'] = df_age_wiki['age'].apply(lambda x: int(x))
df_age_imdb['age'] = df_age_imdb['age'].apply(lambda x: int(x))


# In[ ]:


df_age_wiki.shape, df_age_imdb.shape


# In[ ]:


max_age = df_age_wiki["age"].max(); print(max_age)
min_age = df_age_wiki["age"].min(); print(min_age)


# In[ ]:


max_age = df_age_imdb["age"].max(); print(max_age)
min_age = df_age_imdb["age"].min(); print(min_age)


# In[ ]:


df_age_wiki.hist()


# In[ ]:


df_age_imdb.hist()


# In[ ]:


seed = 42

# # python RNG
# import random
# random.seed(seed)

# # pytorch RNGs
# import torch
# torch.manual_seed(seed)
# torch.backends.cudnn.deterministic = True
# if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

# numpy RNG
import numpy as np
np.random.seed(seed)


# In[ ]:


path = Path('../input/')


# In[ ]:


path_utk = Path('../input/utk-face-cropped/utkcropped/utkcropped/')


# In[ ]:


def extract_age(filename):
    return float(filename.stem.split('_')[0])


# ## Preparing the Data for Fastai Models

# In[ ]:


tfms = get_transforms(max_rotate= 10.,max_zoom=1., max_lighting=0.20, do_flip=False,
                      max_warp=0., xtra_tfms=[flip_lr(), brightness(change=(0.3, 0.60), p=0.7), contrast(scale=(0.5, 2), p=0.7),
                                              crop_pad(size=600, padding_mode='border', row_pct=0.,col_pct=0.),
                                              rand_zoom(scale=(1.,1.5)), rand_crop(),
                                              perspective_warp(magnitude=(-0.1,0.1)),
                                              #jitter(magnitude=(-0.05,0.05), p=0.5),
                                              symmetric_warp(magnitude=(-0.1,0.1)) ])

path_imdb = Path('../input/imdb-wiki-faces-dataset/imdb_crop/imdb_crop/')
path_wiki = Path('../input/wiki-face-data/wiki_crop/wiki_crop/')


data_imdb = ImageList.from_df(df_age_imdb, path_imdb, cols=['full_path'], folder ='.').split_by_rand_pct(0.2, seed=42).label_from_df(label_cls=FloatList).transform(tfms, resize_method=ResizeMethod.CROP, padding_mode='border', size=128).databunch(bs=64*2,num_workers=0).normalize(imagenet_stats)
data_wiki_small = ImageList.from_df(df_age_wiki, path_wiki, cols=['full_path'], folder ='.').split_by_rand_pct(0.2, seed=42).label_from_df(label_cls=FloatList).transform(tfms, resize_method=ResizeMethod.CROP, padding_mode='border', size=128).databunch(bs=64*2,num_workers=0).normalize(imagenet_stats)
data_wiki_big = ImageList.from_df(df_age_wiki, path_wiki, cols=['full_path'], folder ='.').split_by_rand_pct(0.2, seed=42).label_from_df(label_cls=FloatList).transform(tfms, resize_method=ResizeMethod.CROP, padding_mode='border', size=256).databunch(bs=64*2,num_workers=0).normalize(imagenet_stats)


# In[ ]:


data_utk_small = ImageList.from_folder(path_utk).split_by_rand_pct(0.2, seed=42).label_from_func(extract_age, label_cls=FloatList).transform(tfms, resize_method=ResizeMethod.CROP, padding_mode='border', size=128).databunch(bs=64*2,num_workers=0).normalize(imagenet_stats)
data_utk_big = ImageList.from_folder(path_utk).split_by_rand_pct(0.2, seed=42).label_from_func(extract_age, label_cls=FloatList).transform(tfms, resize_method=ResizeMethod.CROP, padding_mode='border', size=256).databunch(bs=64*2,num_workers=0).normalize(imagenet_stats)


# In[ ]:


df_appa = pd.read_csv('/kaggle/input/appa-real-face-cropped/labels.csv')
df_appa.head()


# In[ ]:


df_appa.rename(columns = {"file_name":"full_path", "real_age":"age"}, inplace=True)
df_appa['age'] = df_appa['age'].apply(lambda x: int(float(x)))


# In[ ]:


path = Path('../input/appa-real-face-cropped/final_files/final_files/')
path_csv = '../input/appa-real-face-cropped/'
path_folder = '../input/appa-real-face-cropped/final_files/final_files/'

data_appa_small = ImageList.from_df(df_appa, path, cols=['full_path'], folder ='.').split_by_rand_pct(0.2, seed=42).label_from_df(label_cls=FloatList).transform(tfms, resize_method=ResizeMethod.CROP, padding_mode='border', size=128).databunch(bs=64*2,num_workers=0).normalize(imagenet_stats)
data_appa_big = ImageList.from_df(df_appa, path, cols=['full_path'], folder ='.').split_by_rand_pct(0.2, seed=42).label_from_df(label_cls=FloatList).transform(tfms, resize_method=ResizeMethod.CROP, padding_mode='border', size=256).databunch(bs=64*2,num_workers=0).normalize(imagenet_stats)


# ## A look at the Images

# In[ ]:


data_imdb.show_batch(4, figsize=(12,12))


# In[ ]:


data_wiki_small.show_batch(4, figsize=(12,12))


# In[ ]:


data_wiki_big.show_batch(4, figsize=(12,12))


# In[ ]:


data_utk_small.show_batch(4, figsize=(12,12))


# In[ ]:


data_utk_big.show_batch(4, figsize=(12,12))


# In[ ]:


data_appa_small.show_batch(4, figsize=(12,12))


# In[ ]:


data_appa_big.show_batch(4, figsize=(12,12))


# ## Fastai Modelling

# In[ ]:


import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
import random


# In[ ]:


opt_func = partial(optim.Adam, betas=(0.9,0.99), eps=1e-5)


# In[ ]:


def resnet(pretrained=False):
    pretrained = 'imagenet' if pretrained else None
    model = pretrainedmodels.resnet34(pretrained=pretrained)
    return model


# In[ ]:


class L1LossFlat(nn.SmoothL1Loss):
    def forward(self, input:Tensor, target:Tensor) -> Rank0Tensor:
        return super().forward(input.view(-1), target.view(-1))


# In[ ]:


df_utk_small = data_utk_small.to_df()


# In[ ]:


df_utk_small.head()


# In[ ]:


df_utk_small.rename(columns = {"x":"full_path", "y":"age"}, inplace=True)
df_utk_small['age'] = df_utk_small['age'].apply(lambda x: int(float(x)))


# In[ ]:


df_utk_small.head()


# In[ ]:


df_appa.head()


# In[ ]:


src_wiki = '../input/wiki-face-data/wiki_crop/wiki_crop/'
src_utk = '../input/utk-face-cropped/utkcropped/utkcropped/'
src_appa = '../input/appa-real-face-cropped/final_files/'

dest = '../working/wiki_utk/'

import os
import shutil
import pathlib

pathlib.Path(dest).mkdir(parents=True, exist_ok=True)

os.listdir(dest)


# In[ ]:


for root, dirs, files in os.walk(src_wiki):
    for file in files:
        path_file = os.path.join(root,file)
        shutil.copy2(path_file, dest)


# In[ ]:


for root, dirs, files in os.walk(src_utk):
    for file in files:
        path_file = os.path.join(root,file)
        shutil.copy2(path_file,dest)


# In[ ]:


for root, dirs, files in os.walk(src_appa):
    for file in files:
        path_file = os.path.join(root,file)
        shutil.copy2(path_file,dest)


# In[ ]:


df_age_wiki.head()


# In[ ]:


df_age_wiki['full_path'] = df_age_wiki['full_path'].str[3:]
df_age_wiki.head()


# In[ ]:


frames = [df_age_wiki, df_utk_small, df_appa]
df_wiki_utk_appa = pd.concat(frames)


# In[ ]:


df_wiki_utk_appa.head()


# In[ ]:


df_wiki_utk_appa.shape


# In[ ]:


os.listdir(dest)


# In[ ]:


df_wiki_utk_appa.hist()


# In[ ]:


df_wiki_utk_appa['age'].value_counts()


# In[ ]:


df_wiki_utk_appa = df_wiki_utk_appa[df_wiki_utk_appa['age'] <= 100]
df_wiki_utk_appa = df_wiki_utk_appa[df_wiki_utk_appa['age'] > 0]


# In[ ]:


df_wiki_utk_appa['age'].min(), df_wiki_utk_appa['age'].max()


# In[ ]:


df_wiki_utk_appa['age'] = df_wiki_utk_appa['age'].astype(int)


# In[ ]:


path_wiki_utk_appa = Path('../working/wiki_utk/')

np.random.seed(42)

data_wiki_small_src = (ImageList.from_df(df_wiki_utk_appa, path_wiki_utk_appa, cols=['full_path'], folder='.')
                   .split_by_rand_pct(0.2, seed=42)
                   .label_from_df(label_cls=FloatList))
                   
# data_wiki_big = ImageList.from_df(df_wiki_utk_appa, path_wiki_utk_appa, cols=['full_path'], folder='.').split_by_rand_pct(0.2, seed=42).label_from_df(label_cls=FloatList).transform(tfms, resize_method=ResizeMethod.CROP, padding_mode='border', size=256).databunch(bs=64*2,num_workers=0).normalize(imagenet_stats)


# In[ ]:


data_wiki_small = (data_wiki_small_src.transform(tfms, resize_method=ResizeMethod.CROP, padding_mode='border', size=128)
                   .databunch(bs=64,num_workers=0).normalize(imagenet_stats))


# In[ ]:


data_wiki_small.show_batch(4, figsize=(12,12))


# ## Fastai Age Model

# In[ ]:


class AgeModel(nn.Module):
    def __init__(self):
        super().__init__()
        layers = list(models.resnet34(pretrained=True).children())[:-2]
        layers += [AdaptiveConcatPool2d(), Flatten()]
        #layers += [nn.BatchNorm1d(4096, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)]
        #layers += [nn.Dropout(p=0.60)]
        #layers += [nn.Linear(4096, 1024, bias=True), nn.ReLU(inplace=True)]
        #layers += [nn.BatchNorm1d(2048, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)]
        #layers += [nn.Dropout(p=0.60)]
        #layers += [nn.Linear(2048, 1024, bias=True), nn.ReLU(inplace=True)]
        #layers += [nn.BatchNorm1d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)]
        #layers += [nn.Dropout(p=0.75)]
        #layers += [nn.Linear(1024, 256, bias=True), nn.ReLU(inplace=True)]
        #layers += [nn.BatchNorm1d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)]
        #layers += [nn.Dropout(p=0.50)]
        #layers += [nn.Linear(512,256 , bias=True), nn.ReLU(inplace=True)]
        layers += [nn.BatchNorm1d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)]
        layers += [nn.Dropout(p=0.50)]
        layers += [nn.Linear(1024, 512, bias=True), nn.ReLU(inplace=True)]
        layers += [nn.BatchNorm1d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)]
        layers += [nn.Dropout(p=0.50)]
        layers += [nn.Linear(512, 16, bias=True), nn.ReLU(inplace=True)]
        layers += [nn.Linear(16,1)]
        self.agemodel = nn.Sequential(*layers)
    def forward(self, x):
        return self.agemodel(x).squeeze(-1)
          # could add 116*torch.sigmoid


# In[ ]:


model = AgeModel()


# In[ ]:


learn = Learner(data_wiki_small, model, model_dir = "/temp/model/", opt_func=opt_func, bn_wd=False, metrics=root_mean_squared_error,
               callback_fns=[ShowGraph]).mixup(stack_y=False, alpha=0.2)

learn.loss_func = L1LossFlat()


# In[ ]:


learn.split([model.agemodel[4],model.agemodel[6],model.agemodel[8]])


# In[ ]:


#learn = cnn_learner(data, resnet50, pretrained = True, model_dir = "/temp/model/",callback_fns=[ShowGraph])
#learn.loss_func = L1LossFlat()


# In[ ]:


import fastai
learn.freeze_to(-1)
learn.lr_find()
learn.recorder.plot(suggestion = True)


# In[ ]:


lr = 2e-2


# In[ ]:


learn.fit_one_cycle(5, max_lr=slice(lr), wd=(1e-6, 1e-4, 1e-2, 1e-1), pct_start=0.5, callbacks=[SaveModelCallback(learn)])


# In[ ]:


learn.save('first_head_resnet34')
learn.load('first_head_resnet34')


# In[ ]:


learn.unfreeze()
learn.lr_find()
learn.recorder.plot(suggestion = True)


# In[ ]:


learn.fit_one_cycle(5, max_lr=slice(1e-6, lr/5), wd=(1e-6, 1e-4, 1e-2, 1e-1), 
                    callbacks=[SaveModelCallback(learn)], pct_start=0.5)


# In[ ]:


x,y = next(iter(learn.data.train_dl))
x.shape


# In[ ]:


learn.save('first_body_resnet34')
learn.load('first_body_resnet34')


# In[ ]:


learn.show_results()


# In[ ]:


img = open_image('../input/picture4/PM_Modi_2015.jpg')
img


# In[ ]:


x = learn.predict(img)
x


# ## Image Resizing

# In[ ]:


data_wiki_big = (data_wiki_small_src.transform(tfms, size=256)
        .databunch(num_workers=0).normalize(imagenet_stats))

learn.data = data_wiki_big
data_wiki_big.train_ds[0][0].shape


# In[ ]:


learn.freeze_to(-1)
learn.lr_find()
learn.recorder.plot(suggestion = True)


# In[ ]:


lr = 5e-4


# In[ ]:


learn.fit_one_cycle(5, max_lr=slice(lr), wd=(1e-6, 1e-4, 1e-2, 1e-1), pct_start=0.5, callbacks=[SaveModelCallback(learn)])


# In[ ]:


learn.recorder.plot_lr()


# In[ ]:


learn.recorder.plot_losses()


# In[ ]:


learn.save('second_head_resnet34')
learn.load('second_head_resnet34')


# In[ ]:


img = open_image('../input/picture4/PM_Modi_2015.jpg')
img


# In[ ]:


learn.predict(img)


# In[ ]:


x,y = next(iter(learn.data.train_dl))
x.shape


# In[ ]:


learn.unfreeze()
learn.lr_find()
learn.recorder.plot(suggestion = True)


# In[ ]:


learn.fit_one_cycle(5, max_lr=slice(1e-6,1e-4), wd=(1e-6, 1e-4, 1e-2, 1e-1), callbacks=[SaveModelCallback(learn)], pct_start=0.5)


# In[ ]:


learn.save('second_body_resnet34')
learn.load('second_body_resnet34')


# In[ ]:


learn.show_results()


# In[ ]:


img = open_image('../input/picture4/PM_Modi_2015.jpg')
img


# In[ ]:


x = learn.predict(img)
x


# In[ ]:


learn.save('third_body_resnet34')
learn.load('third_body_resnet34')


# # Fastai Hooks

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
learn.model.agemodel[16]


# In[ ]:


len(data_wiki_big.train_ds.items), len(data_wiki_big.valid_ds.items)


# In[ ]:


sf = SaveFeatures(learn.model.agemodel[16])


# In[ ]:


_= learn.get_preds(data_wiki_big.train_ds)
_= learn.get_preds(DatasetType.Valid)


# In[ ]:


len(sf.features)


# In[ ]:


img_path = [str(x) for x in (list(data_wiki_big.train_ds.items) +list(data_wiki_big.valid_ds.items))]
label = [x for x in (list(data_wiki_big.train_ds.y.items) +list(data_wiki_big.valid_ds.y.items))]
label_id = [x for x in (list(data_wiki_big.train_ds.y.items) +list(data_wiki_big.valid_ds.y.items))]


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


# # Annoy

# In[ ]:


from annoy import AnnoyIndex

f = len(df_new['img_repr'][0])
t = AnnoyIndex(f, metric='euclidean')


# In[ ]:


ntree = 500

for i, vector in enumerate(df_new['img_repr']):
    t.add_item(i, vector)
_  = t.build(ntree)


# In[ ]:


import time
def get_similar_images_annoy(img_index, num):
    start = time.time()
    base_img_id, base_vector, base_label  = df_new.iloc[img_index, [0,3,1]]
    similar_img_ids = t.get_nns_by_item(img_index, num)
    end = time.time()
    print(f'{(end - start) * 1000} ms')
    return base_img_id, base_label, df_new.iloc[similar_img_ids]


# In[ ]:


num = 89
base_image, base_label, similar_images_df = get_similar_images_annoy(890, num)


# In[ ]:


print(base_label)
open_image(base_image)


# In[ ]:


similar_images_df


# In[ ]:


data_src = (ImageList.from_df(similar_images_df, path=".",cols=['img_path'], folder='.' )
        .split_none()
        .label_from_df(cols=['label'],label_cls=FloatList))


# In[ ]:


data = (data_src.transform(tfms, resize_method=ResizeMethod.CROP, padding_mode='border', size=256)
                   .databunch(bs=num,num_workers=0).normalize(imagenet_stats))


# In[ ]:


data.show_batch(5)

