#!/usr/bin/env python
# coding: utf-8

# ## Fastai CNN learner
# 
# Short and readable code inorder to create a fastai CNN learner that preforms with 99%+ success rate

# In[ ]:


import os
import numpy as np
import pandas as pd

from pathlib import *
from fastai.vision import *


# In[ ]:


# Create dataset
path = '/kaggle/input/working-potholes/kaggle/working/pothole-detection-dataset'

fastai_data = ImageDataBunch.from_folder(path, train=".",
                                         valid_pct=0.2, ds_tfms=get_transforms(), 
                                         size=300, num_workers=4, 
                                         bs=32).normalize(imagenet_stats)


# In[ ]:


fastai_data.show_batch(figsize=(7,8))
fastai_data.classes, fastai_data.c, len(fastai_data.train_ds), len(fastai_data.valid_ds)


# ## Transfer Learning
# 
# Wikipedia [Transfer learning](https://en.wikipedia.org/wiki/Transfer_learning) - (TL) is a research problem in machine learning (ML) that focuses on storing knowledge gained while solving one problem and applying it to a different but related problem.
# 
# I'm using the famous [resnet34](https://www.kaggle.com/pytorch/resnet34) pretrained model in order to get better results.
# 
# I want my kaggle notebook to work without the internet option so I uploaded resnet .pth file with the + Add Data option.

# In[ ]:


# Tricking torch not to download resnet
# Torch did not ran yet so cache directory does not exist
get_ipython().system('mkdir -p /tmp/.cache/torch/checkpoints')

# Copy resnet34.pth into torch cache
get_ipython().system('cp /kaggle/input/resnet34/resnet34.pth /tmp/.cache/torch/checkpoints/resnet34-333f7ec4.pth')


# In[ ]:


learn = cnn_learner(fastai_data, models.resnet34, metrics=error_rate, model_dir='/tmp/models')


# ## Fit One Cycle
# One of the most important things in training a deep neural netowrk is the learning rate (LR). The 1 cycle policy is a way do adopt the learning rate during the train of the model. The cycle contains two steps of equal lengths, one going from a low LR to a higher one and than back to the minimum.
# 
# For further reading
# [One Cycle policy](https://sgugger.github.io/the-1cycle-policy.html)

# In[ ]:


learn.fit_one_cycle(4)


# In[ ]:


# Save model before unfreezing
learn.save('stage1')


# In[ ]:


def plot_confusion_matrix(learner):
    """
    Ploting a confusion matrix using a fastai cnn learner
    """
    interp = ClassificationInterpretation.from_learner(learner)
    interp.plot_confusion_matrix()


# In[ ]:


plot_confusion_matrix(learn)


# In[ ]:


unfreezed_learner = learn.load('stage1')


# ## Freezing and Unfreezing
# When training using a pretrained model all the layers except the last one are **freezed**. That means they are going to keep the same values and training is not going to effect them.
# Unfreezing the model lets you train those layers.

# In[ ]:


unfreezed_learner.unfreeze()


# In[ ]:


unfreezed_learner.fit_one_cycle(3, max_lr=slice(3e-5,3e-4))


# In[ ]:


plot_confusion_matrix(unfreezed_learner)


# # Solving corrupted images problem
# 
# * Copied all images into working dir so I can delete the unrelevant images.
# * Used fastai library verify_images in order to find out which images are valid.
# * Created a zip file that contains all the valid images.
# * Uploaded it as data to the notebook.
# 
# Need to do this once so I left it in comments

# In[ ]:


# Move data to a read/write dir
# input_path = "/kaggle/input/pothole-detection-dataset/"
# !rm -rf /kaggle/working/pothole-detection-dataset
# !cp -r  {input_path} /kaggle/working


# In[ ]:


# Create paths
# path = Path("/kaggle/working/pothole-detection-dataset")
# normal_path = path/"normal"
# pothole_path = path/"potholes"


# In[ ]:


# Verify and delete images it can not open
# verify_images(path=normal_path, delete=True)
# verify_images(path=pothole_path, delete=True)


# In[ ]:


# from zipfile import ZipFile

# Create zip file
# dirName = "/kaggle/working/pothole-detection-dataset"

# with ZipFile('/tmp/b.zip', 'w') as zipObj:
#    # Iterate over all the files in directory
#    for folderName, subfolders, filenames in os.walk(dirName):
#        for filename in filenames:
#            #create complete filepath of file in directory
#            filePath = os.path.join(folderName, filename)
#            # Add file to zip
#            zipObj.write(filePath)
# Move zip file to a place I can download from
# !cp  /tmp/b.zip /kaggle/working/b.zip

