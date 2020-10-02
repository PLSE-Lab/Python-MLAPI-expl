#!/usr/bin/env python
# coding: utf-8

# In[ ]:


get_ipython().system('pip3 install fastai==1.0.42')


# In[ ]:


#!pip3 install git+https://github.com/fastai/fastai.git
#!pip3 install git+https://github.com/pytorch/pytorch


# In[ ]:


import torch
print(torch.__version__)


# In[ ]:


from fastai.vision import * 
from fastai import *


# In[ ]:


#import fastai; 
#fastai.show_install(1)


# In[ ]:


planet = untar_data(URLs.PLANET_TINY)
planet_tfms = get_transforms(flip_vert=False, max_lighting=0.1, max_zoom=1.05, max_warp=0.)


# In[ ]:


data = (ImageItemList.from_csv(planet, 'labels.csv', folder='train', suffix='.jpg')
        .random_split_by_pct()
        .label_from_df(label_delim=' ')
        .transform(planet_tfms, size=128)
        .databunch()
        .normalize(imagenet_stats))


# In[ ]:


data.show_batch(rows=2, figsize=(9,7))


# In[ ]:


#??create_cnn()


# In[ ]:


#learn = create_cnn(data, models.resnet18, metrics=Fbeta(beta=2))
#learn.fit(5)


# In[ ]:


learn = create_cnn(data, models.resnet18)
learn.fit_one_cycle(5,1e-2)
learn.save('mini_train')


# In[ ]:


learn.show_results(rows=3, figsize=(12,15))


# In[ ]:


#from fastai.metrics import FBeta


# In[ ]:


learn = create_cnn(data, models.resnet50, metrics=[accuracy_thresh])


# In[ ]:


learn.lr_find()


# In[ ]:


learn.recorder.plot()


# In[ ]:


lr = 3e-2


# In[ ]:


#learn.fit_one_cycle(5, slice(lr))


# In[ ]:


learn.fit_one_cycle(5,1e-2)
learn.save('mini_train')


# In[ ]:


learn.show_results(rows=3, figsize=(24,30))


# In[ ]:




