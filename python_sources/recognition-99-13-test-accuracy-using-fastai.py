#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from fastai import *
from fastai.vision import *


# ## **Path of dataset**

# In[ ]:


path='../input/flowers-recognition'


# # **Preparing Data**
# 1. Splitting data into 80:20 ratio
# 2. Resized input image to 224 * 224 pixels
# 3. Normalizing data

# In[ ]:


data=ImageDataBunch.from_folder(path,valid_pct=0.20,ds_tfms=get_transforms(),size=224).normalize(imagenet_stats)
data


# ## **Display of Random Images**

# In[ ]:


data.show_batch(rows=3,fig=(5,5))


# # **Creaing Densenet201 Model Architecture**

# In[ ]:


model=cnn_learner(data,models.densenet201,metrics=accuracy)


# ## Summary of Densenet201 Model

# In[ ]:


model.summary()


# # Training Model with 5 epochs

# In[ ]:


model.fit(5)


# ## Loss Graph

# In[ ]:


model.recorder.plot_losses()


# ## Testset Accuracy plot

# In[ ]:


model.recorder.plot_metrics()


# ## Learning rate plot
# 
# lr is made constant on this training i.e 0.003

# In[ ]:


model.recorder.plot_lr()


# ## Confusion Matrix on a test-set

# In[ ]:


interp=ClassificationInterpretation.from_learner(model)
interp.plot_confusion_matrix(title='Confusion matrix')


# ### **Don't forget to give upvote if you like my work**
