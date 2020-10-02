#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from pathlib import Path
from fastai import *
from fastai.vision import *
from fastai.callbacks import *
import torch


# In[ ]:


# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[ ]:


data_folder = Path("../input")
#data_folder.joinpath('train').ls()


# In[ ]:


train_df = pd.read_csv("../input/train.csv")
test_df = pd.read_csv("../input/sample_submission.csv")


# In[ ]:


train_df.head(), train_df.info()


# In[ ]:


test_img = ImageList.from_df(test_df, path=data_folder/'test', folder='test')
trfm = get_transforms(do_flip=True, flip_vert=True, max_rotate=10.0, max_zoom=1.2, max_lighting=0.2, max_warp=0.2, p_affine=0.75, p_lighting=0.75)
train_il = ImageList.from_df(train_df, path=data_folder/'train', folder='train')
train_img = (train_il.split_by_rand_pct(0.01)
            .label_from_df()
            .add_test(test_img)
            .transform(trfm, size=336)
            .databunch(path='.', bs=32, device= torch.device('cuda:0'))
            .normalize(imagenet_stats)
           )


# In[ ]:


train_il


# In[ ]:


# train_img.show_batch(rows=3, figsize=(7,6))


# In[ ]:


callbacks = [partial(EarlyStoppingCallback, monitor='accuracy', min_delta=0.01, patience=3)]


# In[ ]:


learn = cnn_learner(train_img, models.densenet161, metrics=[error_rate, accuracy], callback_fns=callbacks).mixup().to_fp16()


# In[ ]:


learn.lr_find()
learn.recorder.plot(suggestion=True)


# In[ ]:


lr = 1e-02
learn.fit_one_cycle(3, slice(lr), callbacks=[SaveModelCallback(learn, every='improvement', monitor='quadratic_kappa', name='bestmodel')])


# In[ ]:


learn.unfreeze()
learn.lr_find()
learn.recorder.plot(suggestion=True)


# In[ ]:


learn.fit_one_cycle(16, slice(1e-05, 1e-03), callbacks=[SaveModelCallback(learn, every='improvement', monitor='quadratic_kappa', name='bestmodel')])


# In[ ]:


#interp = ClassificationInterpretation.from_learner(learn)
#interp.plot_top_losses(9, figsize=(7,6))


# In[ ]:


learn.to_fp32()


# In[ ]:


preds,_ = learn.TTA(ds_type=DatasetType.Test)


# In[ ]:


test_df.has_cactus = preds.numpy()[:, 0]


# In[ ]:


test_df.to_csv('submission.csv', index=False)


# # Pseudo Label

# In[ ]:


test_df.head()


# In[ ]:


pseudo_df = test_df.copy()


# In[ ]:


pseudo_df.loc[pseudo_df['has_cactus'] > 0.99, 'has_cactus'] = 1
pseudo_df.loc[pseudo_df['has_cactus'] < 0.01, 'has_cactus'] = 0


# In[ ]:


pseudo_label_df = pseudo_df[pseudo_df['has_cactus'] > 0.99]
pseudo_label_df.append(pseudo_df[pseudo_df['has_cactus'] < 0.01])
pseudo_label_df.shape


# In[ ]:


pseudo_label_df['has_cactus'] = pseudo_label_df['has_cactus'].astype(np.int64)


# In[ ]:


pseudo_label_df.head()


# In[ ]:


label_src = ImageList.from_df(pseudo_label_df, path = data_folder/'test', folder='test', cols='id')


# In[ ]:


label_src


# In[ ]:


train_il.add(label_src)


# In[ ]:


train_img = (train_il.split_by_rand_pct(0.01)
            .label_from_df()
            .add_test(test_img)
            .transform(trfm, size=336)
            .databunch(path='.', bs=32, device= torch.device('cuda:0'))
            .normalize(imagenet_stats)
           )


# In[ ]:


learn = cnn_learner(train_img, models.densenet161, metrics=[error_rate, accuracy], callback_fns=callbacks).mixup().to_fp16()


# In[ ]:


lr = 1e-02
learn.fit_one_cycle(3, slice(lr), callbacks=[SaveModelCallback(learn, every='improvement', monitor='quadratic_kappa', name='bestmodel')])


# In[ ]:


learn.unfreeze()
learn.fit_one_cycle(16, slice(1e-05, 1e-03), callbacks=[SaveModelCallback(learn, every='improvement', monitor='quadratic_kappa', name='bestmodel')])


# In[ ]:


learn.to_fp32()


# In[ ]:


preds,_ = learn.TTA(ds_type=DatasetType.Test)
test_df.has_cactus = preds.numpy()[:, 0]
test_df.to_csv('submission.csv', index=False)

