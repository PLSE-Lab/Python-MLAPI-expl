#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from fastai import *
from fastai.vision import *

get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


path = '/kaggle/input/simpsons_dataset/simpsons_dataset/'
bs = 256


# In[ ]:


fnames = get_image_files(path)


# In[ ]:


data = ImageDataBunch.from_folder(path, ds_tfms=get_transforms(), valid_pct=0.2, size=224, num_workers=4, bs=bs).normalize()


# In[ ]:


data.show_batch(3)


# In[ ]:


learn = cnn_learner(data, models.resnet34, metrics=accuracy)
learn.unfreeze()


# In[ ]:


learn.fit_one_cycle(5)


# In[ ]:


interp = ClassificationInterpretation.from_learner(learn)


# In[ ]:


interp.plot_confusion_matrix(figsize=(12, 12))

