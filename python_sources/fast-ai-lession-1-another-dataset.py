#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from fastai import *
from fastai.vision import *


# In[ ]:


ds = 64


# In[ ]:


path = untar_data(URLs.MNIST); path


# In[ ]:


path.ls()


# In[ ]:


path_training = path/'training'
path_testing = path/'testing'


# In[ ]:


data = ImageDataBunch.from_folder(path, tain='training', test='testing', valid_pct= 0.2, size=16)


# In[ ]:


data.show_batch(3, figsize=(5,5))


# In[ ]:


learn = create_cnn(data, models.resnet50, metrics=error_rate)


# In[ ]:


learn.fit_one_cycle(4)


# In[ ]:


learn.save('stage-1')


# In[ ]:


interp = ClassificationInterpretation.from_learner(learn)
losses, idxs = interp.top_losses()
len(data.valid_ds)==len(losses)==len(idxs)


# In[ ]:


interp.plot_top_losses(9, figsize=(15,11))


# In[ ]:


interp.plot_confusion_matrix(figsize=(10,10), dpi=60)


# In[ ]:


interp.most_confused(min_val=2)


# In[ ]:


learn.unfreeze()


# In[ ]:


learn.fit_one_cycle(1)


# In[ ]:


learn.save('stage-2')


# In[ ]:


learn.lr_find()


# In[ ]:


learn.recorder.plot()


# In[ ]:


learn.unfreeze()


# In[ ]:


learn.fit_one_cycle(1, max_lr=slice(1e-6,1e-5))


# In[ ]:


learn.save('stage-2')


# In[ ]:




