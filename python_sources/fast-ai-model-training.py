#!/usr/bin/env python
# coding: utf-8

# In[ ]:


get_ipython().run_line_magic('reload_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


from fastai import *
from fastai.vision import *
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import cv2
from sklearn.metrics import cohen_kappa_score
import torch


# In[ ]:


Path = '../input/aptos2019-blindness-detection/'


# In[ ]:


train_df = pd.read_csv(Path+'train.csv')
test_df = pd.read_csv(Path+'test.csv')


# In[ ]:


train_df.head()


# In[ ]:


train_df['id_code'] = train_df['id_code'].apply(lambda x : x + '.png')
test_df['id_code'] = test_df['id_code'].apply(lambda x : x + '.png')


# In[ ]:


train_df.head()


# In[ ]:


bs = 32
SIZE = 224

tfms = get_transforms(do_flip=True,flip_vert=True,max_warp=0.,max_rotate=360.0)


# In[ ]:


data = (ImageList.from_df(df=train_df,folder='train_images',path=Path)
       .split_by_rand_pct(0.2)
       .label_from_df(cols='diagnosis')
       .transform(tfms,size=SIZE)
       .databunch(bs=bs)
       .normalize(imagenet_stats))


# In[ ]:


data


# In[ ]:


data.show_batch(rows=5,fig_size=(5,5))


# In[ ]:


get_ipython().system("mkdir -p '/tmp/.cache/torch/checkpoints/'")
get_ipython().system('cp ../input/resnet101/resnet101.pth /tmp/.cache/torch/checkpoints/resnet101-5d3b4d8f.pth')


# In[ ]:


arch = models.resnet101


# In[ ]:


kappa = KappaScore()
kappa.weights = "quadratic"


# In[ ]:


learn = cnn_learner(data, arch, metrics=[kappa,error_rate,accuracy],pretrained=True,path='../working/')


# In[ ]:


learn.lr_find()


# In[ ]:


learn.recorder.plot(suggestion=True)


# In[ ]:


lr = learn.recorder.min_grad_lr
lr


# In[ ]:


learn.fit_one_cycle(5, lr)


# In[ ]:


learn.recorder.plot_losses()


# In[ ]:


learn.save('stage1')


# In[ ]:


learn.unfreeze()


# In[ ]:


learn.lr_find()


# In[ ]:


learn.recorder.plot(suggestion=True)


# In[ ]:


lr = learn.recorder.min_grad_lr
lr


# In[ ]:


learn.fit_one_cycle(4, max_lr=slice(lr,lr/10))


# In[ ]:


learn.save('stage2')


# In[ ]:


SIZE = 256

data = (ImageList.from_df(df=train_df,folder='train_images',path=Path)
       .split_by_rand_pct(0.2)
       .label_from_df(cols='diagnosis')
       .transform(tfms,size=SIZE)
       .databunch(bs=bs)
       .normalize(imagenet_stats))


# In[ ]:


data


# In[ ]:


learn.data = data


# In[ ]:


learn.freeze()


# In[ ]:


learn.lr_find()


# In[ ]:


learn.recorder.plot(suggestion=True)


# In[ ]:


lr = learn.recorder.min_grad_lr
lr


# In[ ]:


learn.fit_one_cycle(5, slice(lr,lr/10))


# In[ ]:


learn.save('stage3')

