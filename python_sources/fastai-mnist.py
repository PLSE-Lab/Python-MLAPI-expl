#!/usr/bin/env python
# coding: utf-8

# In[ ]:


get_ipython().run_line_magic('reload_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


from fastai import *
from fastai.vision import *


# In[ ]:


path = untar_data(URLs.MNIST_SAMPLE); path


# In[ ]:


path.ls()


# In[ ]:


tfms = get_transforms(do_flip=False)
data = ImageDataBunch.from_folder(path, ds_tfms=tfms, size=26, bs=64)


# In[ ]:


data.show_batch(rows=3, figsize=(7, 6))


# In[ ]:


learn = cnn_learner(data, models.resnet34, metrics=accuracy)


# In[ ]:


learn.lr_find()
learn.recorder.plot()


# In[ ]:


learn.fit_one_cycle(4, max_lr=slice(1e-04, 1e-02))


# In[ ]:


learn.save('stage-1')


# In[ ]:


learn.lr_find()
learn.recorder.plot()


# In[ ]:


interp = ClassificationInterpretation.from_learner(learn)


# In[ ]:


interp.plot_top_losses(9, figsize=(6,7))


# In[ ]:


interp.plot_confusion_matrix(figsize=(4, 4), dpi=100, title="Confusion Matrix")


# In[ ]:




