#!/usr/bin/env python
# coding: utf-8

# In[ ]:


get_ipython().run_line_magic('reload_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


from fastai.vision import *
from fastai.metrics import error_rate
import os


# In[ ]:


bs = 64


# In[ ]:


path = Path('../input/chest_xray/chest_xray')
path.ls()


# In[ ]:


img = open_image(path/'val'/'NORMAL'/'NORMAL2-IM-1440-0001.jpeg')
print(img.data.shape)
img.show()


# In[ ]:


tfms = get_transforms()


# In[ ]:


np.random.seed(7)
data = ImageDataBunch.from_folder(path, 
                                  valid='val',
                                  valid_pct=0.2,
                                  size=256, bs=bs,
                                  ds_tfms=tfms).normalize(imagenet_stats)


# In[ ]:


data.show_batch(3, figsize=(6,6))


# In[ ]:


data.classes, data.c, len(data.train_ds), len(data.valid_ds)


# > - we have a very small validation set so some augmentation is needed

# In[ ]:


learn = cnn_learner(data, models.resnet50, metrics=error_rate, model_dir="/tmp/model/")


# In[ ]:


learn.fit_one_cycle(4)


# In[ ]:


learn.save('stage-1')


# In[ ]:


learn.unfreeze()


# In[ ]:


learn.lr_find()


# In[ ]:


learn.recorder.plot()


# In[ ]:


learn.fit_one_cycle(10, max_lr=slice(3e-5, 3e-4))


# In[ ]:


learn.save('stage-2')


# In[ ]:


interp = ClassificationInterpretation.from_learner(learn)


# In[ ]:


interp.plot_confusion_matrix()


# In[ ]:




