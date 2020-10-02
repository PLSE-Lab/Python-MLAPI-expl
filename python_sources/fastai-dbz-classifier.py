#!/usr/bin/env python
# coding: utf-8

# In[ ]:


get_ipython().run_line_magic('reload_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


get_ipython().system('pip install --upgrade fastai2 > /dev/null')
get_ipython().system('pip install --upgrade fastcore > /dev/null')


# In[ ]:


from fastai2 import *
from fastai2.vision import *


# In[ ]:


bs = 64


# In[ ]:


path=Path('../input/goku-vegeta-dataset/DB dataset/')
path


# In[ ]:


path.ls()


# In[ ]:


np.random.seed(2)


# In[ ]:


tfms = get_transforms(flip_vert=True, max_lighting=0.1, max_zoom=1.05, max_warp=0.)


# In[ ]:


data = ImageDataBunch.from_folder(path, valid_pct=0.2, size=256 ,ds_tfms=tfms, bs=bs).normalize(imagenet_stats)
data.c


# In[ ]:


data.show_batch(rows=3)


# In[ ]:


get_ipython().run_line_magic('pinfo2', 'cnn_learner')


# In[ ]:


learn = cnn_learner(data,models.resnet50, metrics=[error_rate,accuracy])


# In[ ]:


learn.model_dir='/kaggle/working'


# In[ ]:


learn.lr_find()
learn.recorder.plot()


# In[ ]:


learn.fit_one_cycle(8, 1e-3)


# In[ ]:


learn.save('stage-1')


# In[ ]:


learn.lr_find()
learn.recorder.plot()


# In[ ]:


learn.unfreeze()
learn.fit_one_cycle(5, slice(1e-5,1e-4))


# In[ ]:


learn.save('stage-2')


# In[ ]:


learn.load('stage-1')
learn.unfreeze()
learn.fit_one_cycle(3, slice(1e-5,1e-4))


# In[ ]:


learn.save('stage-3')


# In[ ]:


learn.lr_find()
learn.recorder.plot()


# In[ ]:


learn.load('stage-3')
learn.unfreeze()
learn.fit_one_cycle(6, slice(1e-4,5e-4))


# In[ ]:


learn.save('stage-4')
learn.freeze_to(-2)
learn.fit_one_cycle(2,1e-7)


# In[ ]:


learn.load('stage-4')
learn.export('/kaggle/working/model.pkl')


# In[ ]:


interp = ClassificationInterpretation.from_learner(learn)


# In[ ]:


interp.plot_confusion_matrix()


# In[ ]:


interp.plot_top_losses(12, figsize=(15,11))


# In[ ]:




