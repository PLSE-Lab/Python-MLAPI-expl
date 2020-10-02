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


path = Path('../input/shipsnet/shipsnet/')
fnames = get_image_files(path)
fnames[:2]


# > - we can see the labels are the first number of the filenames
# > - since posixpath doesn't support indexing we can use a regex

# In[ ]:


pat = r'^\D*(\d+)'


# In[ ]:


np.random.seed(7)
tfms = get_transforms(flip_vert=True, # we can flip satellite images vertically, ship may come in from any angle
                      max_warp=0)     # There should be no warp for top-down view images
data = ImageDataBunch.from_name_re(path, fnames, pat, ds_tfms=tfms, size=256, bs=bs).normalize(imagenet_stats)


# In[ ]:


data.show_batch(rows=3, figsize=(7,6))


# In[ ]:


print(data.classes)


# In[ ]:


learn = cnn_learner(data, models.resnet50, metrics=error_rate, model_dir="/tmp/model/")


# In[ ]:


learn.model


# In[ ]:


lr_find(learn)
learn.recorder.plot()


# In[ ]:


lr = 3e-3


# In[ ]:


learn.fit_one_cycle(5, slice(lr))


# In[ ]:


learn.save('stage-1')


# In[ ]:


interp = ClassificationInterpretation.from_learner(learn)
losses, idxs = interp.top_losses()
interp.plot_top_losses(9, figsize=(7, 6))


# > - nighttime images are harder to recognize

# In[ ]:


interp.plot_confusion_matrix()


# > - we already have quite a decent model

# In[ ]:


learn.unfreeze()


# In[ ]:


lrs = slice(lr/400, lr/4)
learn.fit_one_cycle(8, lrs)


# In[ ]:


learn.save('stage-2')


# In[ ]:


interp = ClassificationInterpretation.from_learner(learn)
losses, idxs = interp.top_losses()
interp.plot_top_losses(9, figsize=(7, 6))


# In[ ]:


interp.plot_confusion_matrix()


# In[ ]:


learn.show_results(rows=3, figsize=(10,10))

