#!/usr/bin/env python
# coding: utf-8

# # Fruit 360 Dataset Classification with fast.ai

# ## The Dataset
# 
# The dataset contains 65.000+ images of 95 fruits. Every type of fruit is associated with a folder with the same name.
# 
# Source:
# - Horea Muresan, Mihai Oltean, [Fruit recognition from images using deep learning](https://www.researchgate.net/publication/321475443_Fruit_recognition_from_images_using_deep_learning), Acta Univ. Sapientiae, Informatica Vol. 10, Issue 1, pp. 26-42, 2018.

# Every notebook starts with the following three lines; they ensure that any edits to libraries you make are reloaded here automatically, and also that any charts or images displayed are shown in this notebook.

# In[ ]:


get_ipython().run_line_magic('reload_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')
get_ipython().run_line_magic('matplotlib', 'inline')


# We are going to work with the fastai V1. The fastai library provides many useful functions that enable us to quickly and easily build neural networks and train our models.

# In[ ]:


from fastai.vision import *
from fastai.metrics import error_rate


# In[ ]:


bs = 16 # batch size
sz = 224 # image size


# We are going to use the [ImageDataBunch.from_folder](https://docs.fast.ai/vision.data.html#ImageDataBunch.from_folder) function to which we must pass a path as an argument and which will extract the data.

# In[ ]:


base_path = Path('../input/fruits-360_dataset/fruits-360/')
data = ImageDataBunch.from_folder(path=base_path, train='Training', valid='Test', size=sz, bs=bs, num_workers=0).normalize(imagenet_stats)


# In[ ]:


data.show_batch(rows=3, figsize=(6, 6))


# In[ ]:


print(data.classes)
len(data.classes), data.c


# ## Training: resnet34
# 
# ResNet34 is a 34 layer residual network, described in Kaiming He,Xiangyu Zhang,[Deep Residual Learning for Image Recognition](https://arxiv.org/pdf/1512.03385.pdf)

# In[ ]:


model_dir=Path('/tmp/models/')
learn = cnn_learner(data, models.resnet34, metrics=error_rate, model_dir=model_dir)
learn.model


# In[ ]:


learn.lr_find()
learn.recorder.plot()


# In[ ]:


learn.fit_one_cycle(4)


# ## Results
# 
# Let's see what results we have got.
# 
# We will first see which were the categories that the model most confused with one another. We will try to see if what the model predicted was reasonable or not.

# In[ ]:


interp = ClassificationInterpretation.from_learner(learn)
losses,idxs = interp.top_losses()
len(data.valid_ds) == len(losses) == len(idxs)


# In[ ]:


interp.plot_top_losses(9, figsize=(15, 11))


# In[ ]:


interp.plot_confusion_matrix(figsize=(32, 32), dpi=60)


# In[ ]:


interp.most_confused(min_val=2)


# ## Acknowledgements
# Thanks to the fast.ai instructors for the inspiring [lesson 1](https://course.fast.ai/videos/?lesson=1).
