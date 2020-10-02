#!/usr/bin/env python
# coding: utf-8

# > **Note** : Code is taken from https://www.kaggle.com/abhikjha/fastai-hooks-and-image-similarity-search and applied to this dataset for understanding the functionality.
# 
# 

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        os.path.join(dirname, filename)

# Any results you write to the current directory are saved as output.


# In[ ]:


get_ipython().run_line_magic('reload_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')
get_ipython().run_line_magic('matplotlib', 'inline')

get_ipython().system('pip install fastai')
import fastai

from fastai import *
from fastai.vision import *

# from torchvision.models import *
# import pretrainedmodels

from utils import *
import sys

from fastai.callbacks.hooks import *

from fastai.callbacks.tracker import EarlyStoppingCallback
from fastai.callbacks.tracker import SaveModelCallback


# In[ ]:


path = Path('../input/best-artworks-of-all-time/images/images')


# In[ ]:


tfms = get_transforms(max_rotate= 10.,max_zoom=1., max_lighting=0.20, do_flip=False,
                      max_warp=0., xtra_tfms=[flip_lr(), brightness(change=(0.3, 0.60), p=0.7), contrast(scale=(0.5, 2), p=0.7),
                                              crop_pad(size=600, padding_mode='border', row_pct=0.,col_pct=0.),
                                              rand_zoom(scale=(1.,1.5)), rand_crop(),
                                              perspective_warp(magnitude=(-0.1,0.1)),
                                              symmetric_warp(magnitude=(-0.1,0.1)) ])

src = (ImageList.from_folder(path)
        .split_by_rand_pct(0.2, seed=42)
        .label_from_folder())


# In[ ]:


data = (src.transform(tfms, resize_method=ResizeMethod.CROP, padding_mode='border', size=128)
        .databunch(bs=64, num_workers=0)
        .normalize(imagenet_stats))


# In[ ]:


data.show_batch()


# In[ ]:


print(f'Classes: \n {data.classes}')


# In[ ]:


learn = cnn_learner(data, models.resnet50, metrics=accuracy, model_dir="/temp/model" ).mixup()


# In[ ]:


learn.freeze()
learn.lr_find()
learn.recorder.plot(suggestion=True)


# In[ ]:


lr=1e-3
learn.fit_one_cycle(2, max_lr=slice(1e-2), wd = (1e-6, 1e-4, 1e-2), pct_start=0.5)


# In[ ]:


learn.unfreeze()
learn.lr_find()
learn.recorder.plot(suggestion=True)


# In[ ]:


learn.fit_one_cycle(2, max_lr = slice(5e-6,lr/5), wd=(1e-6, 1e-4, 1e-2), pct_start=0.5)


# In[ ]:


data_big = (src.transform(tfms, resize_method=ResizeMethod.CROP, padding_mode='border', size=256)
        .databunch(bs=64, num_workers=0)
        .normalize(imagenet_stats))

learn.data = data_big


# In[ ]:


learn.freeze()
learn.lr_find()
learn.recorder.plot(suggestion=True)


# In[ ]:


lr = 1e-5
learn.fit_one_cycle(2, max_lr=slice(lr), wd=(1e-6, 1e-4, 1e-2), pct_start=0.5)


# In[ ]:


learn.unfreeze()
learn.lr_find()
learn.recorder.plot(suggestion=True)


# In[ ]:


learn.fit_one_cycle(2, max_lr=slice(1e-6, 1e-4), wd=(1e-6, 1e-4, 1e-2), pct_start=0.5)


# In[ ]:


interp = ClassificationInterpretation.from_learner(learn)

losses,idxs = interp.top_losses()

len(data_big.valid_ds)==len(losses)==len(idxs)


# In[ ]:


interp.plot_top_losses(9, figsize=(15,11))


# In[ ]:


interp.plot_confusion_matrix(figsize=(12,12), dpi=60)


# In[ ]:


interp.most_confused(min_val=2)


# In[ ]:


learn.save('final_model')


# In[ ]:


learn.model


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


# Second last layer of the model
learn.model[1][4]


# In[ ]:


sf = SaveFeatures(learn.model[1][4])


# In[ ]:


_= learn.get_preds(data_big.train_ds)
_= learn.get_preds(DatasetType.Valid)


# In[ ]:


len(sf.features)


# In[ ]:


img_path = [str(x) for x in (list(data_big.train_ds.items) +list(data_big.valid_ds.items))]
label = [data_big.classes[x] for x in (list(data_big.train_ds.y.items) +list(data_big.valid_ds.y.items))]
label_id = [x for x in (list(data_big.train_ds.y.items) +list(data_big.valid_ds.y.items))]


# In[ ]:


len(img_path), len(label), len(label_id)


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


df_new.head()


# In[ ]:


df_new.shape


# In[ ]:


from annoy import AnnoyIndex


# In[ ]:


f = len(df_new['img_repr'][0])
t = AnnoyIndex(f, metric='euclidean')


# In[ ]:


f


# In[ ]:


t


# In[ ]:


ntree = 50

for i, vector in enumerate(df_new['img_repr']):
    t.add_item(i, vector)
_  = t.build(ntree)


# In[ ]:


import time
def get_similar_images_annoy(img_index):
    start = time.time()
    base_img_id, base_vector, base_label  = df_new.iloc[img_index, [0, 3, 1]]
    similar_img_ids = t.get_nns_by_item(img_index, 8)
    end = time.time()
    print(f'{(end - start) * 1000} ms')
    return base_img_id, base_label, df_new.iloc[similar_img_ids]


# In[ ]:


base_image, base_label, similar_images_df = get_similar_images_annoy(1943)


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




