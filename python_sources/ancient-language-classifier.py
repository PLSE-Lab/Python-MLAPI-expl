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


bs=64
np.random.seed(2)


# In[ ]:


data_path = "../input/ancient_language_dataset"
data = ImageDataBunch.from_folder(data_path,ds_tfms=get_transforms(),size=224,bs=bs,num_workers=0).normalize(imagenet_stats)


# In[ ]:


data.show_batch(rows=3, figsize=(7,6))


# In[ ]:


data.classes


# In[ ]:


learn = create_cnn(data, models.resnet34, metrics=accuracy, path=".")


# In[ ]:


learn.fit_one_cycle(4)


# In[ ]:


learn.save("stage-1")


# In[ ]:


interp = ClassificationInterpretation.from_learner(learn)

losses,idxs=interp.top_losses()
len(data.valid_ds)==len(losses)==len(idxs)


# In[ ]:


interp.plot_top_losses(9,figsize=(15,11))


# In[ ]:


interp.plot_confusion_matrix(figsize=(12,12),dpi=60)


# In[ ]:


interp.most_confused(min_val=1)


# In[ ]:


learn.unfreeze()


# In[ ]:


learn.fit_one_cycle(1)


# In[ ]:


learn.load("stage-1");


# In[ ]:


learn.lr_find()


# In[ ]:


learn.recorder.plot()


# In[ ]:


learn.fit_one_cycle(2,max_lr=slice(7e-4,1e-3))


# ## Resnet 50

# In[ ]:


data = ImageDataBunch.from_folder(data_path,ds_tfms=get_transforms(),size=299,bs=bs//2,num_workers=0).normalize(imagenet_stats)


# In[ ]:


learn = create_cnn(data, models.resnet50, path=".", metrics=accuracy)


# In[ ]:


learn.lr_find()


# In[ ]:


learn.recorder.plot()


# In[ ]:


learn.fit_one_cycle(8)


# In[ ]:


learn.save('stage-1-50')


# In[ ]:


learn.unfreeze()
learn.fit_one_cycle(6,max_lr=slice(1e-5,1e-4))


# In[ ]:


interp = ClassificationInterpretation.from_learner(learn)


# In[ ]:


interp.plot_confusion_matrix()


# In[ ]:


interp.most_confused(min_val=0)

