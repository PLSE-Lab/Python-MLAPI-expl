#!/usr/bin/env python
# coding: utf-8

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


help(untar_data)


# In[ ]:


path = untar_data(URLs.PETS)
path


# In[ ]:


path.ls()


# In[ ]:


path_anno = path/'annotations'
path_img = path/'images'


# In[ ]:


fnames = get_image_files(path_img)
fnames[:5]


# In[ ]:


np.random.seed(2)
pat = r'/([^/]+)_\d+.jpg$'


# In[ ]:


data = ImageDataBunch.from_name_re(path_img, fnames, pat, ds_tfms=get_transforms(),
                                   size=224, bs=bs).normalize(imagenet_stats)


# In[ ]:


data.show_batch(rows=3, figsize=(7,6))


# In[ ]:


print(data.classes)
len(data.classes),data.c


# In[ ]:


learn = cnn_learner(data, models.resnet34, metrics=error_rate)


# In[ ]:


learn.model


# In[ ]:


learn.fit_one_cycle(4)


# In[ ]:


learn.save('stage-1')


# In[ ]:


interp = ClassificationInterpretation.from_learner(learn)


# In[ ]:


losses,idxs = interp.top_losses()


# In[ ]:


len(data.valid_ds)==len(losses)==len(idxs)


# In[ ]:


interp.plot_top_losses(9, figsize=(15,11))


# In[ ]:


doc(interp.plot_top_losses)


# In[ ]:


interp.plot_confusion_matrix(figsize=(12,12), dpi=60)


# In[ ]:


interp.most_confused(min_val=2)


# In[ ]:


learn.unfreeze()


# In[ ]:


learn.fit_one_cycle(1)


# In[ ]:


learn.load('stage-1');


# In[ ]:


learn.lr_find()


# In[ ]:


learn.recorder.plot()


# In[ ]:


learn.unfreeze()
learn.fit_one_cycle(2, max_lr=slice(1e-6,1e-4))


# * 1.Here is Resnet50 model

# In[ ]:


data = ImageDataBunch.from_name_re(path_img, fnames, pat, ds_tfms=get_transforms(),
                                   size=299, bs=bs//2).normalize(imagenet_stats)


# In[ ]:


learn = cnn_learner(data, models.resnet50, metrics=error_rate)


# In[ ]:


learn.model


# In[ ]:


learn.fit_one_cycle(5)


# In[ ]:


learn.save('stage50-1')


# In[ ]:


interp = ClassificationInterpretation.from_learner(learn)


# In[ ]:


losses,idxs = interp.top_losses()


# In[ ]:


interp.most_confused(min_val=2)


# In[ ]:


learn.unfreeze()


# In[ ]:


learn.load('stage50-1');


# In[ ]:


learn.lr_find()


# In[ ]:


learn.recorder.plot()


# In[ ]:


learn.unfreeze()
learn.fit_one_cycle(2, max_lr=slice(1e-6,1e-4))

