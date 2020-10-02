#!/usr/bin/env python
# coding: utf-8

# ## Fast.ai Learning through Cats & Dogs Image Classification
# 
# #### Sunil Kumar

# This kernel can be used to submit to Cats & Dogs Kaggle competition. One can try 1st option for training just the last layer, i,e., Dense FC layer while re-using pretrained ResNet* layers. Other option is to try SGDR. Then another approach given below is to try SGDR with Differential Learning Rate. Observe the validation loss, validation accuracy and Kaggle score. NOTE that try this with GPU ON and Internet CONNECTED in your Kaggle kernel. One can try fast.ai Images Augmentation while preparing the ConvNet model before model learning and it is one of the ways of better generalizing our deep learning - refer to https://becominghuman.ai/data-augmentation-using-fastai-aefa88ca03f1 .
# 
# Even before moving to lectures beyond Lesson #1, I got curious to explore learning performance with various options given in fast.ai Lesson #1 notebook and explored if all fast.ai tricks can be implemented in Keras & TensorFlow. Refer to tutorial on [Tutorial Keras: Transfer Learning with ResNet50 for image classification on Cats & Dogs dataset](https://www.kaggle.com/suniliitb96/tutorial-keras-transfer-learning-with-resnet50).
# 
# Notes about special observations noticed while practicing with fast.ai on Cats & Dogs dataset from an old Kaggle competition with reference to fast.ai Lesson#1: - 
# * Refer to http://ruder.io/deep-learning-optimization-2017/ for a comprehensive practical background in Deep Learning Optimizations with respect to Optimizers, Learning Rate schedule, etc.
# * Refer to SGDR in https://abdel.me/2018/01/05/fastai-part1/ for a better understanding as what fast.ai is doing under the hood.
# * SGDR
#     * Refer to the below plot of learning schedule for cycle_len=1 (notice iteration count as ~313, i.e., 10k/32)
#     * It is based on the concept of Cyclical Learning, a.k.a., Learning Rate Annealing with Warm Restart. SGD with Learning Rate Annealing schedule is a common practice to compete with adaptive Adam optimizer.
#     * Learning Rate decays as per half cosine with each Epoch Step (ref model.py >> step fn) and it restarts with specified Learning Rate value
# *  Differential Learning Rate for Fine Tuning PreTrained ResNet*    
#     * It is not enough to just train and re-train the Dense FC last layer while keeping ResNet layers frozen.
#     * We idenity best Learning Rate for Dense FC last layer and apply sclaed down Learning Rate for ResNet* grouped layers (Fast.ai seem to have treated ResNet* in 2 grouped layers & I'm yet to figure out as how this grouping has been done).
#     * SGDR + DLR + cyclt_mult affects in stretching Learning Rate Annealing to span across epochs. Refer to below plot for DLR.    
#     

# In[ ]:


import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

import cv2

import os


# In[ ]:


os.listdir("../input")


# In[ ]:


from fastai.imports import *
from fastai.transforms import *
from fastai.conv_learner import *
from fastai.model import *
from fastai.dataset import *
from fastai.sgdr import *
from fastai.plots import *


# In[ ]:


PATH = "../input/"
TMP_PATH = "/tmp/tmp"
MODEL_PATH = "/tmp/model/"
sz=224


# In[ ]:


fnames = np.array([f'train/{f}' for f in sorted(os.listdir(f'{PATH}train'))])
labels = np.array([(0 if 'cat' in fname else 1) for fname in fnames])


# In[ ]:


arch=resnet50
data = ImageClassifierData.from_names_and_array(
    path=PATH, 
    fnames=fnames, 
    y=labels, 
    classes=['dogs', 'cats'], 
    test_name='test', 
    tfms=tfms_from_model(arch, sz)
)
learn = ConvLearner.pretrained(arch, data, precompute=True, tmp_name=TMP_PATH, models_name=MODEL_PATH)


# In[ ]:


###
### Search for suitable, i.e., best Learning Rate for our-newly-added-Last Layer (as we have used 'precompute=True', i.e., ResNet50-minus-its-last-layer weights are being re-used as is)
###
#lrf=learn.lr_find()
#learn.sched.plot_lr()
#learn.sched.plot()

###
### Use the identified best Learning Rate for our-newly-added-Last Layer
### Note that even without running above 3 lines of Learning Rate Finder, it is well known that best learning rate is 0.01 for Cats & Dogs images with 224x224 size
### Kaggle Score obtained is 0.38683 (v7)
###
#learn.fit(0.01, 2)


# In[ ]:


###
### SGDR (SGD with warm Resrart): fast.ai uses half Cosine shape decay (start with 0.01 & decay till 0) of LR during each epoch and then it restarts with 1e-02
### Kaggle score obtained is 0.37578 (v8)
###
learn.fit(1e-2, 10, cycle_len=1)
learn.sched.plot_lr()


# In[ ]:


###
### Continue from Last Layer learned model with PreCompute=TRUE
### Unfreeze all layers (all weights learned so far are retained) => it sets PreCompute=FALSE making all layers learnable
### Effectively, the network weights are intialized as (ResNet-minus-last-layer with its original pre-trained weight & Last Layer as per above model learning while keeping ResNet as frozen)
### Now, all layers are FURTHER learnable
### Kaggle score obtained is 0.34815 (v9)
###
learn.unfreeze()

# Differential LR (above identified best LR for last layer, x0.1 to middle layer, x0.01 to inner layer)
lr=np.array([1e-4,1e-3,1e-2])

learn.fit(lr, 3, cycle_len=1, cycle_mult=2)

learn.sched.plot_lr()


# In[ ]:


temp = learn.predict(is_test=True)
pred = np.argmax(temp, axis=1)


# In[ ]:


import cv2

# learn.predict works on unsorted os.listdir, hence listing filenames without sorting
fnames_test = np.array([f'test/{f}' for f in os.listdir(f'{PATH}test')])

f, ax = plt.subplots(5, 5, figsize = (15, 15))

for i in range(0,25):
    imgBGR = cv2.imread(f'{PATH}{fnames_test[i]}')
    imgRGB = cv2.cvtColor(imgBGR, cv2.COLOR_BGR2RGB)
    
    # a if condition else b
    predicted_class = "Dog" if pred[i] else "Cat"

    ax[i//5, i%5].imshow(imgRGB)
    ax[i//5, i%5].axis('off')
    ax[i//5, i%5].set_title("Predicted:{}".format(predicted_class))    

plt.show()


# In[ ]:


results_df = pd.DataFrame(
    {
        'id': pd.Series(fnames_test), 
        'label': pd.Series(pred)
    })
results_df['id'] = results_df.id.str.extract('(\d+)')
results_df['id'] = pd.to_numeric(results_df['id'], errors = 'coerce')
results_df.sort_values(by='id', inplace = True)

results_df.to_csv('submission.csv', index=False)
results_df.head()


# ### References
# 
# 1. [Kaggle kernel about implementing Cats & Dogs images classification in Keras](https://www.kaggle.com/suniliitb96/tutorial-keras-transfer-learning-with-resnet50)
