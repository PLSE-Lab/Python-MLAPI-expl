#!/usr/bin/env python
# coding: utf-8

# In[ ]:


get_ipython().run_line_magic('reload_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


from fastai.vision import *
from fastai.metrics import error_rate


# The folder names for training and validation data is differnt from the IIT Pet Dataset. Fortunately it is easy to override with `train=` and `valid=` arguments. So we can load using from_folder method as in the course.

# In[ ]:


monkey_images_path = '../input/10-monkey-species/'
tfms = get_transforms()
data = ImageDataBunch.from_folder(monkey_images_path, train='training', valid='validation', ds_tfms=tfms, size=128)


# In[ ]:


print(data.classes)
len(data.classes),data.c


# In[ ]:


data.show_batch()


# The Resnet34 and Resnet50 models are added as input datasets. So we don't need to download them from internet. Copy them to cache folder. 

# In[ ]:


get_ipython().system('mkdir -p /tmp/.torch/models/')
get_ipython().system('cp ../input/resnet34/resnet34.pth /tmp/.torch/models/resnet34-333f7ec4.pth')
get_ipython().system('cp ../input/resnet50/resnet50.pth  /tmp/.torch/models/resnet50-19c8e357.pth')


# Model directory defaults to the dataset directory which is read-only in Kaggle. Change it to the working directory (/kaggle/working)

# In[ ]:


path_model='/kaggle/working/'
path_input='/kaggle/input/'
learn_resnet34 = cnn_learner(data, models.resnet34, metrics=error_rate, model_dir=f'{path_model}')


# In[ ]:


learn_resnet34.fit_one_cycle(4)


# In[ ]:


learn_resnet34.lr_find()
learn_resnet34.recorder.plot()


# For some reason, the model works poorly wheen the learning rate is fine-tuned. Need to workout the optimal working rate (the default one seems to be good enough right now

# In[ ]:


learn.unfreeze()
learn.fit_one_cycle(2, max_lr=slice(1e-03,1e-04))


# Try the same with resnet50. The results are even better.

# In[ ]:


learn_resnet50 = cnn_learner(data, models.resnet50, metrics=error_rate, model_dir=f'{path_model}')


# In[ ]:


learn_resnet50.fit_one_cycle(8)


# In[ ]:


interp_resnet50 = ClassificationInterpretation.from_learner(learn_resnet50)
interp_resnet50.most_confused(min_val=2)


# In[ ]:


interp_resnet38 = ClassificationInterpretation.from_learner(learn_resnet34)
interp_resnet38.most_confused(min_val=2)


# In[ ]:


interp_resnet38.plot_confusion_matrix(figsize=(12,12), dpi=60)


# In[ ]:


interp_resnet50.plot_confusion_matrix(figsize=(12,12), dpi=60)

