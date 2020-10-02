#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#Inspired by fast.ai Practical Deep Learning for Coders(v3) lession 1


# In[ ]:


get_ipython().run_line_magic('reload_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


from fastai.vision import *
from fastai.metrics import error_rate


# In[ ]:


bs = 64


# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os
print(os.listdir('../input/car_data/car_data'))


# In[ ]:


data_dir='../input/car_data/car_data'

list = os.listdir(data_dir) 
number_files = len(list)
print(number_files)


# In[ ]:


path=Path(data_dir)
path


# In[ ]:


data = ImageDataBunch.from_folder(path,  
                                  valid_pct=0.2,
                                  ds_tfms=get_transforms(do_flip=True,flip_vert=False, max_rotate=90),
                                  size=224,bs=64, 
                                  num_workers=0).normalize(imagenet_stats)


# In[ ]:


data.show_batch(rows=3, figsize=(40,40))


# In[ ]:


print(data.classes)
len(data.classes)


# In[ ]:


learn = create_cnn(data, models.resnet34, metrics=accuracy, model_dir="/tmp/model/")


# In[ ]:


learn.model


# In[ ]:


learn.fit_one_cycle(10)


# In[ ]:


learn.save('stage-1')


# In[ ]:


interp = ClassificationInterpretation.from_learner(learn)

losses,idxs = interp.top_losses()

len(data.valid_ds)==len(losses)==len(idxs)


# In[ ]:


interp.plot_top_losses(9, figsize=(30,25))


# In[ ]:


interp.most_confused(min_val=2)


# Unfreezing, fine-tuning, and learning rates

# In[ ]:


learn.unfreeze()


# In[ ]:


learn.fit_one_cycle(10)


# In[ ]:


learn.lr_find()


# In[ ]:



learn.recorder.plot()


# In[ ]:


learn.unfreeze() 
learn.fit_one_cycle(4, max_lr=slice(1e-6,1e-4))


# In[ ]:


learn.save('stage-2')


# In[ ]:


interp = ClassificationInterpretation.from_learner(learn)


# In[ ]:


losses,idxs = interp.top_losses()

len(data.valid_ds)==len(losses)==len(idxs)


# In[ ]:


interp.plot_top_losses(9, figsize=(30,25))


# In[ ]:




