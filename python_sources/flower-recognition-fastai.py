#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from fastai.vision import *
from fastai.metrics import error_rate
import os


# In[ ]:


bs = 64


# In[ ]:


path = '/kaggle/input/flower-image-dataset/flowers'


# In[ ]:


os.listdir(path)


# In[ ]:


fnames = get_image_files(path)  #extract file names
fnames[:5] 


# In[ ]:


np.random.seed(2)
pat = r'/([^/]+)_\d+.jpg$' ## regex which will extract the labels from the filenames


# In[ ]:


data = ImageDataBunch.from_name_re(path, fnames, pat, ds_tfms = get_transforms(), size = 224, bs = bs
                                  ).normalize(imagenet_stats)  


# The ImageDataBunch operator makes a databunch which contains the training, testing, validation data. It also performs specific data augmentation tasks and image normalization.

# In[ ]:


print(data.classes) # different classes of flowers


# # Model for Image Recognition 

# In[ ]:


learn = cnn_learner(data, models.resnet34, metrics=error_rate)


# In[ ]:


learn.fit_one_cycle(4) ### train for 4 epochs 


# With an accuracy of around **92%**, we're doing pretty well!!

# # Interpreting our model

# In[ ]:


interp = ClassificationInterpretation.from_learner(learn)
losses, idxs = interp.top_losses()


# In[ ]:


interp.plot_top_losses(8, figsize = (12,8)) ## these are the top losses - predicted vs actual


# In[ ]:


interp.plot_confusion_matrix(figsize=(12,12), dpi=60) # confusion matrix for the losses

