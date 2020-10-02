#!/usr/bin/env python
# coding: utf-8

# # Men/Women Classification

# <img src="http://okeya-life.com/wp-content/uploads/2017/01/1846x1230xmanwomanlife3.pagespeed.ic_.o9vxJgwptY.jpg" width="700px">

# # Importing Libraries

# In[ ]:


# Import linraries

import os
print(os.listdir("../input"))

from sklearn.metrics import confusion_matrix
from fastai import *
from fastai.vision import *


# # Create ImageDataBunch

# In[ ]:


# define image data directory path
DATA_DIR='../input/data'


# In[ ]:


# The directory under the path is the label name.
os.listdir(f'{DATA_DIR}')


# In[ ]:


# Check if GPU is available
torch.cuda.is_available()


# In[ ]:


# create image data bunch
data = ImageDataBunch.from_folder(DATA_DIR, 
                                  train=".", 
                                  valid_pct=0.2,
                                  ds_tfms=get_transforms(flip_vert=True, max_warp=0),
                                  size=224,
                                  bs=64, 
                                  num_workers=0).normalize(imagenet_stats)


# In[ ]:


# check classes
print(f'Classes: \n {data.classes}')


# In[ ]:


# show some sample images

data.show_batch(rows=3, figsize=(7,6))


# # Model Build

# In[ ]:


# build model (use resnet34)
learn = create_cnn(data, models.resnet34, metrics=accuracy, model_dir="/tmp/model/")


# # Learning

# In[ ]:


# search appropriate learning rate
learn.lr_find()
learn.recorder.plot()


# In[ ]:


learn.fit_one_cycle(6,1e-2)


# In[ ]:


# save stage
learn.save('stage-1')


# In[ ]:


learn.unfreeze()


# In[ ]:


# search appropriate learning rate
learn.lr_find()
learn.recorder.plot()


# In[ ]:


learn.fit_one_cycle(4, max_lr=slice(1e-6,1e-5 ))


# In[ ]:


learn.fit_one_cycle(4, max_lr=slice(1e-6,1e-5 ))


# In[ ]:


# save stage
learn.save('stage-2')


# # Check Result

# In[ ]:


learn.recorder.plot_losses()


# In[ ]:


interp = ClassificationInterpretation.from_learner(learn)
interp.plot_top_losses(9, figsize=(15,11))


# In[ ]:


interp.plot_confusion_matrix(figsize=(8,8), dpi=60)

