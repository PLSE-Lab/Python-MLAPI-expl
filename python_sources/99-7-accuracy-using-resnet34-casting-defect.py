#!/usr/bin/env python
# coding: utf-8

# In[ ]:


get_ipython().run_line_magic('reload_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


from fastai.vision import *
from fastai.metrics import *


# **Data loading**

# In[ ]:


get_ipython().system('pwd')


# In[ ]:


print(os.listdir('../input'))


# In[ ]:


path = Path('../input/real-life-industrial-dataset-of-casting-product/casting_data'); path


# In[ ]:


path.ls()


# In[ ]:


path_train = path/'train'
path_test = path/'test'


# # Classification

# In[ ]:


tfs = get_transforms()


# In[ ]:


data = ImageDataBunch.from_folder(path, valid='test', size=224, bs=48)


# In[ ]:


data.normalize(imagenet_stats)


# In[ ]:


data.show_batch(rows=2, figsize=(7, 6))


# In[ ]:


# see data length
print(data.classes)
len(data.classes), data.c


# **Using a Pre-Trained Model**

# In[ ]:


learn = cnn_learner(data, models.resnet34, metrics= [error_rate, accuracy])


# In[ ]:


# applying a learning technique
learn.fit_one_cycle(4)


# **Fine-Tuning Model**

# In[ ]:


learn.unfreeze()


# In[ ]:


learn.fit_one_cycle(3, max_lr=slice(1e-5, 1e-4))


# **Interpretation**

# In[ ]:


interp = learn.interpret()


# In[ ]:


losses, idx = interp.top_losses()


# In[ ]:


len(data.valid_ds) == len(losses) == len(idx) 


# In[ ]:


interp.plot_top_losses(9, figsize=(15,6))


# In[ ]:


interp.plot_confusion_matrix()


# In[ ]:




