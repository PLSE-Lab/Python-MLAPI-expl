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

# Any results you write to the current directory are saved as output.


# In[ ]:


# train_csv = pd.read_csv('../input/aptos2019-blindness-detection/train.csv')
# test_csv = pd.read_csv('../input/aptos2019-blindness-detection/test.csv')
# train_csv.head()


# In[ ]:


# df=train_csv.copy()
# df['id_code'] = train_csv.id_code.apply(lambda x:x+'.png')


# In[ ]:


from fastai.vision import *
from fastai import *
from fastai.callbacks import *


# In[ ]:


PATH = Path('../input/aptos2019-blindness-detection')


# In[ ]:


df_train = pd.read_csv(PATH/'train.csv')
df_test = pd.read_csv(PATH/'test.csv')


# In[ ]:


#aptos19_stats = ([0.42, 0.22, 0.075], [0.27, 0.15, 0.081])
data = ImageDataBunch.from_df(df=df_train,
                              path=PATH, folder='train_images', suffix='.png',
                              valid_pct=0.1,
                              ds_tfms=get_transforms(flip_vert=False, max_warp=0),
                              size=224,
                              bs=32,
                              seed=37,
                              num_workers=os.cpu_count()
                             ).normalize(imagenet_stats)


# In[ ]:


# data.show_batch(rows=3, figsize=(7,6))


# In[ ]:


# !rm -rf model
# !rm -rf ../model


# In[ ]:


get_ipython().system('mkdir /tmp')
get_ipython().system('mkdir /tmp/.cache/')
get_ipython().system('mkdir /tmp/.cache/torch/')
get_ipython().system('mkdir /tmp/.cache/torch/checkpoints')
get_ipython().system('cp ../input/resnet34fastai/resnet34.pth /tmp/.cache/torch/checkpoints/resnet34-333f7ec4.pth')


# In[ ]:


os.mkdir('../model')
os.listdir('..')


# In[ ]:


learn = cnn_learner(data, models.resnet34, metrics=[accuracy,KappaScore(weights="quadratic")],model_dir = '/kaggle/model')
#learn.callbacks.append(SaveModelCallback(learn,monitor='kappa_score', name='best_kappa'))


# In[ ]:


from fastai.callbacks import ReduceLROnPlateauCallback, EarlyStoppingCallback, SaveModelCallback
ES = EarlyStoppingCallback(learn, monitor='kappa_score',patience = 5)
RLR = ReduceLROnPlateauCallback(learn, monitor='valid_loss',patience = 2)
SAVEML = SaveModelCallback(learn, every='improvement', monitor='kappa_score', name='best_kappa')

learn.callbacks.extend([ES,RLR,SAVEML])


# In[ ]:


learn.callbacks[-1]


# In[ ]:


learn.freeze()
learn.lr_find();learn.recorder.plot(suggestion=True)


# In[ ]:


learn.fit_one_cycle(10, max_lr=6.31e-3)


# In[ ]:


learn.unfreeze()
learn.lr_find(); learn.recorder.plot(suggestion=True)


# In[ ]:


learn.fit_one_cycle(10, max_lr=slice(3.31e-6, 1e-4))


# In[ ]:


tta_params = {'beta':0.12, 'scale':1.0}
sample_df = pd.read_csv('../input/aptos2019-blindness-detection/sample_submission.csv')
sample_df.head()


# In[ ]:


learn = learn.load('best_kappa')


# In[ ]:


learn.data.add_test(ImageList.from_df(
    sample_df, PATH,
    folder='test_images',
    suffix='.png'
))


# In[ ]:


# learn.data.test_ds


# In[ ]:


preds,y = learn.TTA(ds_type=DatasetType.Test, **tta_params)
# num_batch = len(learn.data.test_dl)
# preds,target= learn.get_preds(DatasetType.Test,n_batch=num_batch)


# In[ ]:


# len(preds)


# In[ ]:


sample_df.diagnosis = preds.argmax(1)


# In[ ]:


sample_df.head()


# In[ ]:


sample_df.to_csv('submission.csv',index=False)

