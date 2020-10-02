#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import torch
print(torch.__version__)


# In[ ]:


import fastai
fastai.__version__


# In[ ]:


from fastai.vision import * 
from fastai import *


# In[ ]:


import fastai; 
fastai.show_install(1)


# In[ ]:


planet = untar_data(URLs.PLANET_TINY)
planet_tfms = get_transforms(flip_vert=False, max_lighting=0.1, max_zoom=1.05, max_warp=0.)


# In[ ]:


data = ImageDataBunch.from_csv(planet, folder='train', size=128, suffix='.jpg', sep = ' ', ds_tfms=planet_tfms)


# In[ ]:


data = (ImageItemList.from_csv(planet, 'labels.csv', folder='train', suffix='.jpg')
        #Where to find the data? -> in planet 'train' folder
        .random_split_by_pct()
        #How to split in train/valid? -> randomly with the default 20% in valid
        .label_from_df(sep=' ')
        #How to label? -> use the csv file
        .transform(planet_tfms, size=128)
        #Data augmentation? -> use tfms with a size of 128
        .databunch())                          
        #Finally -> use the defaults for conversion to databunch


# In[ ]:


data.show_batch(rows=2, figsize=(9,7))


# https://docs.fast.ai/metrics.html#accuracy_thresh

# In[ ]:


learn = create_cnn(data, models.resnet18,  metrics=[accuracy_thresh])
learn.fit(5)


# In[ ]:




