#!/usr/bin/env python
# coding: utf-8

# # Setup

# In[ ]:


get_ipython().run_line_magic('reload_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


from fastai import *
from fastai.vision import *


# In[ ]:


bs = 64
np.random.seed(2)


# # Load data

# In[ ]:


data_path = "../input/poisonous_plants_dataset/"
data = ImageDataBunch.from_folder(data_path, bs=bs//2, size=299, ds_tfms=get_transforms(),num_workers=0).normalize(imagenet_stats)


# In[ ]:


data.show_batch(rows=4,figsize=(8,8))


# # Create and train model

# In[ ]:


learn = create_cnn(data, models.resnet50, metrics=accuracy, path=".")


# In[ ]:


learn.fit_one_cycle(4)


# # Interpret results

# In[ ]:


interp = ClassificationInterpretation.from_learner(learn)


# In[ ]:


interp.plot_confusion_matrix(figsize=(8,8))


# In[ ]:


interp.plot_top_losses(9,figsize=(14,14))


# In[ ]:


interp.most_confused(min_val=0)


# In[ ]:


learn.save('stage-1')


# # Fine tuning

# In[ ]:


learn.lr_find()


# In[ ]:


learn.recorder.plot()


# In[ ]:


learn.unfreeze()
learn.fit_one_cycle(4,max_lr=slice(4e-6,1e-5))


# # Interpret Fine tuning results

# In[ ]:


interp = ClassificationInterpretation.from_learner(learn)


# In[ ]:


interp.plot_confusion_matrix(figsize=(8,8))


# In[ ]:


interp.plot_top_losses(9,figsize=(14,14))


# # Predict on test data

# In[ ]:


test_data = ImageDataBunch.from_folder(data_path, bs=bs//2, size=299, ds_tfms=get_transforms(),num_workers=0,valid='test').normalize(imagenet_stats)
loss, acc = learn.validate(test_data.valid_dl)


# In[ ]:


print(f'Loss: {loss}, Accuracy: {acc*100} %')

