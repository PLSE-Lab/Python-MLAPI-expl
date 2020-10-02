#!/usr/bin/env python
# coding: utf-8

# In[ ]:


get_ipython().run_line_magic('reload_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


from fastai.vision import *
import pandas as ps
import numpy as np


# In[ ]:


image_data = Path("../input/cell_images/cell_images")


# In[ ]:


image_data.ls()


# In[ ]:


np.random.seed(42)
data = ImageDataBunch.from_folder(image_data, train='.', valid_pct=0.2,ds_tfms=get_transforms(flip_vert=True, max_warp=0), size=128, bs=64, num_workers=0).normalize(imagenet_stats)


# In[ ]:


data.classes, data.c


# In[ ]:


data.train_ds[0][0].shape


# In[ ]:


data.show_batch(rows=3)


# In[ ]:


model_path = Path('/tmp/models/')
learn = cnn_learner(data, models.resnet50, metrics=accuracy, model_dir=model_path)


# In[ ]:


learn.lr_find()


# In[ ]:


learn.recorder.plot()


# In[ ]:


learn.fit_one_cycle(5, max_lr=slice(1e-02,1e-01))


# In[ ]:


learn.save("stage-1")


# In[ ]:


learn.unfreeze()


# In[ ]:


learn.lr_find()


# In[ ]:


learn.recorder.plot()


# In[ ]:


learn.fit_one_cycle(5, max_lr=slice(1e-05,1e-04))


# In[ ]:


learn.save("stage-2")


# In[ ]:


learn.load("stage-2")


# In[ ]:


np.random.seed(42)
data = ImageDataBunch.from_folder(image_data, train='.', valid_pct=0.2,ds_tfms=get_transforms(flip_vert=True, max_warp=0), size=224, bs=64, num_workers=0).normalize(imagenet_stats)


# In[ ]:


learn.data = data
data.train_ds[0][0].shape


# In[ ]:


learn.freeze()


# In[ ]:


learn.lr_find()


# In[ ]:


learn.recorder.plot()


# In[ ]:


learn.fit_one_cycle(3, max_lr=slice(1e-03,1e-02))


# In[ ]:


learn.save("stage-3")


# In[ ]:


learn.load("stage-3")


# In[ ]:


learn.unfreeze()


# In[ ]:


learn.lr_find()


# In[ ]:


learn.recorder.plot()


# In[ ]:


learn.fit_one_cycle(4, max_lr=slice(1e-5, 1e-4))


# In[ ]:


learn.save("stage-4")


# In[ ]:


interp = ClassificationInterpretation.from_learner(learn)
losses, idxs = interp.top_losses()
len(data.valid_ds)==len(losses)==len(idxs)


# In[ ]:


interp.plot_confusion_matrix()

