#!/usr/bin/env python
# coding: utf-8

# In[ ]:


get_ipython().run_line_magic('reload_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


import os
print(os.listdir("../input/v2-plant-seedlings-dataset/"))


# In[ ]:


from fastai import *
from fastai.vision import *


# In[ ]:


PATH = "../input/v2-plant-seedlings-dataset/nonsegmentedv2/"


# In[ ]:


sz=128
bs=64


# In[ ]:


tfms = get_transforms()
data = ImageDataBunch.from_folder(PATH, valid_pct=0.2,
    ds_tfms=tfms, size=sz,bs=bs, num_workers=0).normalize(imagenet_stats)


# In[ ]:


print(f'We have {len(data.classes)} different types of seedlings\n')
print(f'Types: \n {data.classes}')


# In[ ]:


data.show_batch(8, figsize=(20,15))


# In[ ]:


from os.path import expanduser, join, exists
from os import makedirs
cache_dir = expanduser(join('~', '.torch'))
if not exists(cache_dir):
    makedirs(cache_dir)
models_dir = join(cache_dir, 'models')
if not exists(models_dir):
    makedirs(models_dir)

# copy time!
get_ipython().system('cp ../input/resnet34/resnet34.pth /tmp/.torch/models/resnet34-333f7ec4.pth')


# In[ ]:


learn = create_cnn(data, models.resnet34, metrics=accuracy, model_dir='/output/model/',callback_fns=ShowGraph)


# In[ ]:


lrf=learn.lr_find()
learn.recorder.plot()


# In[ ]:


lr=1e-2


# In[ ]:


learn.fit_one_cycle(2,lr)


# In[ ]:


learn.save('seedlings-stage-1')


# In[ ]:


learn.unfreeze()


# In[ ]:


learn.fit_one_cycle(5, max_lr=slice(5e-6, 5e-4))


# In[ ]:


learn.save('seedlings-stage-2')


# In[ ]:


interp = ClassificationInterpretation.from_learner(learn,tta=True)


# In[ ]:


interp.plot_top_losses(16, figsize=(20,14))


# In[ ]:


interp.plot_confusion_matrix(figsize=(12,12), dpi=100, normalize=True, norm_dec=2, cmap=plt.cm.YlGn)


# In[ ]:




