#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from fastai.vision import *
from fastai.metrics import error_rate


# In[ ]:


bs = 64


# In[ ]:


help(untar_data)


# In[ ]:


path = untar_data(URLs.PETS)
path_anno = path/'annotations'
path_img = path/'images'
fnames = get_image_files(path_img)


# In[ ]:


print("Total number of images",len(fnames))


# In[ ]:


np.random.seed(2)
pat = r'/([^/]+)_\d+.jpg$'


# In[ ]:


fnames[:5]


# In[ ]:


data = ImageDataBunch.from_name_re(path_img,fnames,pat,ds_tfms=get_transforms(),size = 224,bs=bs)
data.normalize(imagenet_stats)


# In[ ]:


data.show_batch(rows=3,figsize=(7,6))


# In[ ]:


print(data.classes)


# In[ ]:


len(data.classes)


# In[ ]:


data.c


# In[ ]:


learn  = cnn_learner(data,models.resnet34,metrics=error_rate)
learn.model


# In[ ]:


learn.fit_one_cycle(4)


# In[ ]:


learn.save('stage-1')

