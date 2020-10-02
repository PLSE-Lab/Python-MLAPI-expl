#!/usr/bin/env python
# coding: utf-8

# ### The goal of this notebook is to train a learning algorithm to classify COVID vs. non-COVID lung X-rays using fastai for PyTorch. It is an adaptation of [Lesson 1](https://github.com/fastai/course-v3/blob/master/nbs/dl1/lesson1-pets.ipynb) in the fast.ai course.  

# In[ ]:


#preamble
get_ipython().run_line_magic('reload_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


#load modules
from fastai.vision import *
from fastai.metrics import error_rate
import os
import pandas as pd


# In[ ]:


#let's start with bs = 16
 bs = 16   


# In[ ]:


#file paths
print('files in folder',os.listdir("../input/covid-chest-xray/"))
path="../input/covid-chest-xray/"
path_anno=path + 'annotations/'
path_img=path + 'images/'
print('annotation directory:',path_anno,'\nimage directory',path_img)


# In[ ]:


fnames=get_image_files(path_img)
fnames[:5]


# ### Data Description
# Browsing through the [COVID Chest Xray data preview on the Kaggle page](https://www.kaggle.com/bachrr/covid-chest-xray), I noticed there are coronal and axial scans in this dataset. 
# 
# See: https://en.wikipedia.org/wiki/Anatomical_plane
# 
# Let's just look at the coronal X ray images. 
# 
# First, we'll look at the metadata file to filter out the images we don't want. 
# By the end of this next few blocks, we'll have a list of only the image files of interest. 

# In[ ]:


#explore the data and filter the images we want
metadf=pd.read_csv(path + 'metadata.csv' )
print(metadf.columns)
metadf.head()


# In[ ]:


#let's see how many of each different type of scan we have 
metadf.groupby(['modality','view']).describe()


# ### To give a consistent view to our machine learning algorithm, let's do just X-rays with a PA view.
# 
# PA stands for back (Posterior) to front (Anterior) 
# 
# This next cell gives us all the file names, which is what we set out to do with this data filtering step.
# 
# Also, let's get the labels associated with these xrays, and count up how many we have of each case.

# In[ ]:


xray_PA_fnames=path_img + metadf[(metadf.modality=='X-ray')&(metadf.view=='PA')].filename
labels=metadf[(metadf.modality=='X-ray')&(metadf.view=='PA')].finding
metadf[(metadf.modality=='X-ray')&(metadf.view=='PA')].groupby('finding').describe()


# ### The majority of images in this database are of COVID-19 patients. 
# 
# Plus, we can expect there could be an issue with distinguishing COVID-19 from Acute Respiratory Disease Syndrome (ARDS). There is some debate in the medical community currently whether COVID-19 presents as ARDS or if the symptoms currently being classified as ARDS in association with COVID-19 is a unique illness. 
# 
# With that in mind, let's train the neural network to distinguish between {ARDS, COVID-19, COVID-19,ARDS} vs everything else. 
# 
# As I make my databunch, I'm going to change the labels so that it reflects the two classes that I want to train: "COVID_ARDS" and "Other"

# In[ ]:


class_labels=labels.copy()
class_labels.replace(to_replace=['COVID-19','ARDS','COVID-19, ARDS'], value='COVID_ARDS',inplace=True)
class_labels.replace(class_labels[class_labels!='COVID_ARDS'],value='Other',inplace=True)
class_labels.unique()


# ### Make the databunch and visualize

# In[ ]:


data = ImageDataBunch.from_lists(path_img, xray_PA_fnames, labels=class_labels, ds_tfms=get_transforms(), 
                                 size=224, bs=bs).normalize(imagenet_stats)

data.show_batch(rows=4, figsize=(15,11))


# In[ ]:


print(data.classes)
len(data.classes),data.c


# In[ ]:


data


# ### Train the model

# In[ ]:


learn=cnn_learner(data, models.resnet34, metrics=accuracy)
learn.model


# In[ ]:


learn.fit_one_cycle(4)


# In[ ]:


learn.model_dir='/kaggle/working/'
learn.save('stage-1')


# ### Results

# In[ ]:


interp = ClassificationInterpretation.from_learner(learn)

losses,idxs = interp.top_losses()

len(data.valid_ds)==len(losses)==len(idxs)


# In[ ]:


interp.plot_top_losses(9, figsize=(15,11))


# In[ ]:


interp.plot_confusion_matrix(figsize=(10,10), dpi=100)


# In[ ]:


interp.most_confused(min_val=2)


# ### Unfreezing, fine-tuning, and learning rates
# Our model is somewhat working as we expect. Let's train the model some more. 

# In[ ]:


learn.unfreeze()


# In[ ]:


learn.fit_one_cycle(1)


# In[ ]:


learn.load('stage-1')


# In[ ]:


learn.lr_find()


# In[ ]:


learn.recorder.plot()


# In[ ]:


learn.fit_one_cycle(2, max_lr=slice(3e-5,3e-4))


# ### Let's try training with resnet50
# to achieve better performance

# In[ ]:


data = ImageDataBunch.from_lists(path_img, xray_PA_fnames, labels=class_labels, ds_tfms=get_transforms(), 
                                 size=224, bs=bs).normalize(imagenet_stats)
learn = cnn_learner(data, models.resnet50, metrics=accuracy)


# In[ ]:


learn.model_dir='/kaggle/working/'
learn.lr_find()
learn.recorder.plot()


# In[ ]:


learn.fit_one_cycle(10)


# The error rate has stabilized. Given this network, these parameters, and dataset, perhaps that is the best it will do. 

# In[ ]:


learn.save('stage-1-50')


# In[ ]:


learn.load('stage-1-50')


# In[ ]:


learn.unfreeze()
learn.fit_one_cycle(3, max_lr=slice(3e-5,3e-4))


# In[ ]:


learn.load('stage-1-50')


# In[ ]:


interp = ClassificationInterpretation.from_learner(learn)


# In[ ]:


interp.plot_confusion_matrix(figsize=(10,10), dpi=100)


# ### Conclusion
# There is a little bit less confusion with resnet50. But overall, the network at this stage can classify COVID_ARDS the majority of the time, but not to a level that would be satisfactory for clinical implementation.
# 
# Possible areas of improvement: 
# * supplement this with more images in the "Other" category to balance the representatives in the training and validation sets
# * at least one image is mislabeled (not a PA view); check for others 
# * subject matter expertise would be helpful to understand what features are unique to COVID_ARDS vs. the other categories. in other words, is it reasonable for any network to do better than this, and what are the features we would expect it to find 
