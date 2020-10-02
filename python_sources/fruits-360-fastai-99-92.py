#!/usr/bin/env python
# coding: utf-8

# In[ ]:


get_ipython().run_line_magic('reload_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


from fastai.vision import *


# In[ ]:


path = Path('/kaggle/input/fruits/fruits-360')
path.ls()


# In[ ]:


# Data augmentation
tfms = get_transforms(flip_vert=True)


# In[ ]:


np.random.seed(42)

# First start with half the original size 50x50 to save time, you can retrain the model on full size images later on
src = (ImageList.from_folder(path)
                .split_by_rand_pct()
                .label_from_folder()
                .transform(tfms, size=50))


# In[ ]:


data = src.databunch(bs=64).normalize(imagenet_stats)


# In[ ]:


data.show_batch(rows=3, figsize=(9,9))


# In[ ]:


arch = models.resnet50
metrics = [error_rate, accuracy]


# In[ ]:


learn = cnn_learner(data, arch, metrics=metrics)


# In[ ]:


learn.model_dir = '/kaggle/working'


# In[ ]:


learn.lr_find()
learn.recorder.plot(suggestions=True)


# In[ ]:


lr = 5e-2
learn.fit_one_cycle(4, slice(5e-2))


# In[ ]:


learn.recorder.plot_losses()


# In[ ]:


interp = ClassificationInterpretation.from_learner(learn)
interp.plot_top_losses(4, figsize=(12,12))


# In[ ]:


interp.most_confused(min_val=2)

