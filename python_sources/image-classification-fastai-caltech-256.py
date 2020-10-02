#!/usr/bin/env python
# coding: utf-8

# **Introduction**

# **Caltech 256**
# 
# We introduce a challenging set of 256 object categories containing a total of 30607 images. The original Caltech-101 [1] was collected by choosing a set of object categories, downloading examples from Google Images and then manually screening out all images that did not fit the category. Caltech-256 is collected in a similar manner with several improvements: a) the number of categories is more than doubled, b) the minimum number of images in any category is increased from 31 to 80, c) artifacts due to image rotation are avoided and d) a new and larger clutter category is introduced for testing background rejection. We suggest several testing paradigms to measure classification performance, then benchmark the dataset using two simple metrics as well as a state-of-the-art spatial pyramid matching [2] algorithm. Finally we use the clutter category to train an interest detector which rejects uninformative background regions. 
# 
# ![caltechdataset.jpg](attachment:caltechdataset.jpg)
# 
# [Citation](https://authors.library.caltech.edu/7694/)

# In[ ]:


get_ipython().run_line_magic('reload_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')
get_ipython().run_line_magic('matplotlib', 'inline')


# ### Libraries used
# 
# **FAST.ai**: Fast.ai is amazing Deep Learning library to make Deep neural networks uncool again. Their fast.ai Massive Open Online Course (MOOC) on Practical Deep Learning for Coders is simply amazing.

# In[ ]:


from fastai.vision import *


# In[ ]:


# Printing 256 object classes
path = Path('/kaggle/input/256_objectcategories/256_ObjectCategories')
path.ls()


# In[ ]:


tfms = get_transforms(do_flip=False,flip_vert=False, max_rotate=0,max_lighting=0.3, max_zoom=1.01)


# **Data**
# 

# We use a small trick told by Jeremy Howards to train our Neural network faster.
# `size=128` to train our network faster 2x times than usual 224px used, which was used in [Caltech 256 Dataset](https://www.kaggle.com/paultimothymooney/caltech-256-dataset-with-fastai-v1).

# In[ ]:


data = ImageDataBunch.from_folder(path, train=".", 
                                  valid_pct=0.2,
                                  ds_tfms=tfms,
                                  size=128,bs=64, 
                                  num_workers=0).normalize(imagenet_stats)


# In[ ]:


data.show_batch(rows=3, figsize=(12,9))


# In[ ]:


arch = models.resnet50


# In[ ]:



learn = cnn_learner(data, arch, metrics=accuracy, model_dir="/tmp/model/")


# In[ ]:


learn.lr_find()


learn.recorder.plot()


# In[ ]:


lr = 1e-01/2


# In[ ]:


learn.fit_one_cycle(5, slice(lr))


# In[ ]:


learn.save('stage1')


# We freeze our initial model and train it with normal image size, which takes a bit more duration. This trip 
# helps us in getting better accuracy for our trained model.

# In[ ]:


data = ImageDataBunch.from_folder(path, train=".", 
                                  valid_pct=0.2,
                                  ds_tfms=tfms,
                                  size=224,bs=64, 
                                  num_workers=0).normalize(imagenet_stats)


# In[ ]:


learn.data = data
data.train_ds[0][0].shape


# In[ ]:


learn.freeze()


# In[ ]:


learn.lr_find()
learn.recorder.plot()


# In[ ]:


lr = 1e-3/3


# In[ ]:


learn.fit_one_cycle(5, slice(lr))


# ## Output predictions for a given img
# 

# In[ ]:


img = data.train_ds[0][0]


# In[ ]:


learn.predict(img)


# **Conclusion**
# 
# 85% accuracy is pretty good given 256 categories and so few lines of code! That too in shorter time duration.
# 
# The training time of this model is approximately about 1 hour 30 minutes.
# 

# ## Things to try
# 
# - Use a bigger Resnet model(Resnet 101 exists)
# - Train for more epochs
# - Use [Image augmentation in Kaggle](https://www.kaggle.com/init27/introduction-to-image-augmentation-using-fastai)
