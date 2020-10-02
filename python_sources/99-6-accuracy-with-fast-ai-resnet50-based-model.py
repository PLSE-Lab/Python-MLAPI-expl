#!/usr/bin/env python
# coding: utf-8

# Copying the data from zip file to a different location. Mainly because the input directory given by Kaggle is a read-only folder.

# In[ ]:


get_ipython().system(' mkdir ../data')
get_ipython().system(' mkdir ../data/train')
get_ipython().system(' mkdir ../data/validation')
get_ipython().system(' cp -r ../input/training/training/* ../data/train/')
get_ipython().system(' cp -r ../input/validation/validation/* ../data/validation/')


# Import statements

# In[ ]:


from fastai import *
from fastai.vision import *
from pathlib import Path


# Loading the data

# In[ ]:


data = ImageDataBunch.from_folder(path=Path('../data').resolve(), train='train', valid='validation', dl_tfms=get_transforms(), num_workers=0, bs=64, size=224).normalize(imagenet_stats)


# In[ ]:


# import pandas as pd

# labels_df = pd.read_csv('../input/monkey_labels.txt', delimiter=' *, *', engine='python')
# labels = dict(zip(labels_df['Label'].tolist(), labels_df['Common Name'].tolist()))


# In[ ]:


# data = ImageImageList.from_folder(path=Path('../data').resolve()).split_by_folder(train='train', valid='validation').label_from_func(func=lambda x: labels[str(x.parts[-2])]).transform(get_transforms(), size=224).databunch(num_workers=0, bs=64).normalize(imagenet_stats)


# Visualizing some of the data loaded

# In[ ]:


data.show_batch(rows=3, figsize=(5,5))


# Creating a ResNet34 based CNN classifier and fitting for 3 cycles

# In[ ]:


learn_34 = create_cnn(data, models.resnet34, metrics=[accuracy, error_rate])
learn_34.fit_one_cycle(3)


# Creating a plot to check the predictions

# In[ ]:


interp_34 = ClassificationInterpretation.from_learner(learn_34)
interp_34.plot_top_losses(9, figsize=(15,11))


# We can see that the classifier misclassifies to 3 images from the validation set. We can also observe that the model has learnt that the face of the monkey is the most important feature to classify the breed.

# Let us check out the confusion matrix to see where the wrong classifications are happening

# In[ ]:


interp_34.plot_confusion_matrix()


# Using Learning Rate finder to find the LR sweet spot to improve the model

# In[ ]:


learn_34.lr_find()
learn_34.recorder.plot()


# The learning rate between 2e-4 and 9e-2 has the longest slope of decreasing loss. Using that range to train the network further

# In[ ]:


learn_34.fit_one_cycle(1, slice(2e-4, 9e-2))


# The validation loss and the accuracy has saturated for the model based on ResNet34. Trying a larger model.

# In[ ]:


learn_50 = create_cnn(data, models.resnet50, metrics=[accuracy, error_rate])
learn_50.fit_one_cycle(3)


# In[ ]:


interp_50 = ClassificationInterpretation.from_learner(learn_50)
interp_50.plot_confusion_matrix()


# In[ ]:


interp_50.plot_top_losses(4, figsize=(15,11))


# Just one image is being predicted wrong. And we can see the probability of this prediction is very low compared to the others.

# In[ ]:


learn_50.lr_find()
learn_50.recorder.plot()


# In[ ]:


learn_50.fit_one_cycle(2, slice(8e-06, 1.2e-06))


# Looks like we are able to get close 100% accuracy. Just 1 mistake by the model. For a model trained in under 10 minutes, an accuracy of this level is astounding. With a little more tweaks, we should be able to get 100% accuracy.
# 
# EDIT: I was able to get 100% by running the kernel over and over a couple of times. I guess the randomness helps boost this accuracy. But since Kaggle runs the kernel again when making the commit, the results will be slightly different.

# In[ ]:




