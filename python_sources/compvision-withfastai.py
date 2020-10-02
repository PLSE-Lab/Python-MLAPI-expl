#!/usr/bin/env python
# coding: utf-8

# # Computer Vision Hackathon AV with fast-ai

# ### Load neccessary libraries

# In[ ]:


get_ipython().run_line_magic('reload_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


from albumentations import *
import cv2

import os
print(os.listdir("../input"))

#!pip install pretrainedmodels
from tqdm import tqdm_notebook as tqdm
from torchvision.models import *
#import pretrainedmodels

from fastai.vision import *
from fastai.vision.models import *
from fastai.vision.learner import model_meta
from fastai.callbacks import * 

#from utils import *
import sys


# In[ ]:


path = "../input/emergency-vs-nonemergency-vehicle-classification/dataset"


# In[ ]:


import numpy as np
import pandas as pd


# trainKg = pd.read_csv("../input/emergency-vs-nonemergency-vehicle-classification/dataset/train.csv")
# testKg = pd.read_csv("../input/emergency-vs-nonemergency-vehicle-classification/dataset/test_vc2kHdQ.csv")

# Clean_tr is file with updated mislabelled images

# In[ ]:


#trainAV = pd.read_csv("../input/emergency-vs-nonemergency-vehicle-classification/dataset/train_SOaYf6m/train.csv")
trainAV = pd.read_csv("../input/cleantrain/clean_tr.csv")
testAV = pd.read_csv("../input/emergency-vs-nonemergency-vehicle-classification/dataset/test_vc2kHdQ.csv")


# In[ ]:


trainAV.shape


# In[ ]:


trainAV.info()


# In[ ]:


trainAV.head()


# In[ ]:


tfms = get_transforms(max_rotate=90.0, max_zoom=1.3, max_lighting=0.4, max_warp=0.4,
                      p_affine=1.0, p_lighting=1.)


# In[ ]:


bs = 32 #with image size 299, bs=48 & above will allocate more memory
sz = 299


# ### Trained model with various validation sets & finally for 90% train & 10% test, this will give highest model accuracy

# In[ ]:


np.random.seed(42)
data = ImageDataBunch.from_csv('../input', folder = 'emergency-vs-nonemergency-vehicle-classification/dataset/train_SOaYf6m/images', csv_labels = 'cleantrain/clean_tr.csv',
                               valid_pct=0.10,size = sz, ds_tfms = tfms,bs=bs)
data.normalize(imagenet_stats)


# In[ ]:


data.show_batch(rows = 3)


# In[ ]:


data.train_ds


# In[ ]:


def _plot(i,j,ax):
    x,y = data.train_ds[3]
    x.show(ax, y=y)

plot_multi(_plot, 3, 3, figsize=(8,8))


# In[ ]:


print(data.classes); data.c


# ## Apply different Fast-AI ResNet Models & train it

# ## Resnet-34

# In[ ]:


gc.collect()
learnResnet34 = cnn_learner(data, models.resnet34, metrics=error_rate, bn_final=True)


# In[ ]:


learnResnet34.fit_one_cycle(1)


# We get 93.9% accuracy with resnet34 model

# ## Resnet-50

# In[ ]:


gc.collect()
learnResnet50 = cnn_learner(data, models.resnet50, metrics=error_rate, bn_final=True)


# In[ ]:


learnResnet50.fit_one_cycle(1)


# We get 92.7% accuracy with resnet50 model

# ## Resnet-152

# In[ ]:


gc.collect()
learnResnet152 = cnn_learner(data, models.resnet152, metrics=accuracy, bn_final=True)#error_rate


# In[ ]:


learnResnet152.fit_one_cycle(1)


# With pre-trained model we get 90.24% accuracy

# In[ ]:


learnResnet152.model_dir = "/kaggle/working/models"
learnResnet152.save("stage-152-1")


# ### Let's unload pre-trained set, get learning rate & train model for training data we have

# In[ ]:


learnResnet152.unfreeze()
learnResnet152.fit_one_cycle(1)


# In[ ]:


learnResnet152.model_dir = "/kaggle/working/models"
learnResnet152.save("stage-152-2")


# In[ ]:


learnResnet152.load('stage-152-2');
learnResnet152.lr_find()


# In[ ]:


learnResnet152.recorder.plot()


# In[ ]:


learnResnet152.load('stage-152-2');


# In[ ]:


learnResnet152.fit_one_cycle(3, max_lr=slice(3e-7,1e-3))


# In[ ]:


learnResnet152.fit_one_cycle(3, max_lr=slice(3e-7,1e-3))


# In[ ]:


learnResnet152.fit_one_cycle(3, max_lr=slice(3e-7,1e-3))


# In[ ]:


learnResnet152.fit_one_cycle(3, max_lr=slice(3e-7,1e-3))


# In[ ]:


learnResnet152.fit_one_cycle(3, max_lr=slice(3e-7,1e-3))


# In[ ]:


learnResnet152.fit_one_cycle(3, max_lr=slice(3e-7,1e-3))


# In[ ]:


learnResnet152.save('stage-152-fin1');


# In[ ]:


learnResnet152.fit_one_cycle(3, max_lr=slice(3e-7,1e-3))


# In[ ]:


learnResnet152.fit_one_cycle(1, max_lr=slice(3e-7,1e-3))


# In[ ]:


learnResnet152.fit_one_cycle(1, max_lr=slice(3e-7,1e-3))


# In[ ]:


learnResnet152.load('stage-152-fin1');


# In[ ]:


learnResnet152.freeze()


# In[ ]:


learnResnet152.fit_one_cycle(3, max_lr=slice(1e-3))


# In[ ]:


learnResnet152.fit_one_cycle(3, max_lr=slice(1e-3))


# In[ ]:


learnResnet152.fit_one_cycle(3, max_lr=slice(3e-7,1e-3))


# In[ ]:


learnResnet152.fit_one_cycle(1, max_lr=slice(1e-3))


# differential learning rates to alter the lower layers as well. The lower layers want to be altered less, so it is good practice to set each learning rate to be 10 times lower than the last:

# In[ ]:


learnResnet152.unfreeze()


# In[ ]:


learnResnet152.fit_one_cycle(3, max_lr=slice(1e-4))


# In[ ]:


learnResnet152.fit_one_cycle(1, max_lr=slice(1e-4))


# In[ ]:


learnResnet152.fit_one_cycle(1, max_lr=slice(1e-5))


# In[ ]:


learnResnet152.fit_one_cycle(1, max_lr=slice(1e-5))


# In[ ]:


learnResnet152.fit_one_cycle(1, max_lr=slice(1e-5))


# In[ ]:


learnResnet152.fit_one_cycle(5, max_lr=slice(1e-5))


# In[ ]:


learnResnet152.load('stage-152-fin1');


# ## Let's plot & see

# ### Results
# Let's see what results we have got.
# 
# We will first see which were the categories that the model most confused with one another. We will try to see if what the model predicted was reasonable or not. In this case the mistakes look reasonable (none of the mistakes seems obviously naive). This is an indicator that our classifier is working correctly.
# 
# Furthermore, when we plot the confusion matrix, we can see that the distribution is heavily skewed: the model makes the same mistakes over and over again but it rarely confuses other categories. This suggests that it just finds it difficult to distinguish some specific categories between each other; this is normal behaviour.

# In[ ]:


interp = ClassificationInterpretation.from_learner(learnResnet152)
losses,idxs = interp.top_losses()

len(data.valid_ds)==len(losses)==len(idxs)
interp.plot_top_losses(4, figsize=(15,11))


# In[ ]:


interp.plot_confusion_matrix()


# ## Apply vgg19_bn models

# In[ ]:


learnVGG = cnn_learner(data, models.vgg19_bn, metrics=accuracy, bn_final=True)#error_rate
#learnVGG =  VGG16()
#ConvLearner.pretrained


# In[ ]:


learnVGG.fit_one_cycle(3)


# In[ ]:


learnVGG.model_dir = "/kaggle/working/models"
learnVGG.save("stage-vgg-1")


# In[ ]:


learnVGG.unfreeze()
learnVGG.fit_one_cycle(1)


# In[ ]:


learnVGG.save("stage-vgg-2")


# In[ ]:


learnVGG.lr_find()
learnVGG.recorder.plot()


# In[ ]:


learnVGG.fit_one_cycle(6,slice(1e-6,1e-5))


# In[ ]:





# ### Results
# Let's see what results we have got.
# 
# We will first see which were the categories that the model most confused with one another. We will try to see if what the model predicted was reasonable or not. In this case the mistakes look reasonable (none of the mistakes seems obviously naive). This is an indicator that our classifier is working correctly.
# 
# Furthermore, when we plot the confusion matrix, we can see that the distribution is heavily skewed: the model makes the same mistakes over and over again but it rarely confuses other categories. This suggests that it just finds it difficult to distinguish some specific categories between each other; this is normal behaviour.

# In[ ]:


interpvg = ClassificationInterpretation.from_learner(learnVGG)
losses,idxs = interpvg.top_losses()

len(data.valid_ds)==len(losses)==len(idxs)
interpvg.plot_top_losses(9, figsize=(15,11))


# In[ ]:


interpvg.plot_confusion_matrix()


# VGG19 modle gives 0.9513 score on AV

# ## Predict for given Test data

# In[ ]:


dft = pd.read_csv('../input/emergency-vs-nonemergency-vehicle-classification/dataset/test_vc2kHdQ.csv')
dft.head()


# In[ ]:


dft.shape


# In[ ]:


dt_test = ImageList.from_df(dft, '../input/emergency-vs-nonemergency-vehicle-classification/dataset/train_SOaYf6m/images')


# In[ ]:


# To get image from images folder based on name from test.csv
# str('../input/emergency-vs-nonemergency-vehicle-classification/dataset/train_SOaYf6m/images/'+dft["image_names"][1])


# In[ ]:


img = open_image(str('../input/emergency-vs-nonemergency-vehicle-classification/dataset/train_SOaYf6m/images/'+dft["image_names"][1]))
pred_class,pred_idx,outputs = learnResnet152.predict(img)
pred_class


# ### Predict for target images
# ### from Resnet152

# In[ ]:


defaults.device = torch.device('cpu')
labl =[]
for i in range(dft.shape[0]): #
    img = open_image(str('../input/emergency-vs-nonemergency-vehicle-classification/dataset/train_SOaYf6m/images/'+dft["image_names"][i]))
    #img = img.normalize(imagenet_stats)
    pred_class,pred_idx,outputs = learnResnet152.predict(img)
    labl.append(pred_class)


# ### Predict from vgg19

# In[ ]:


defaults.device = torch.device('cpu')
labl =[]
for i in range(dft.shape[0]): #
    img = open_image(str('../input/emergency-vs-nonemergency-vehicle-classification/dataset/train_SOaYf6m/images/'+dft["image_names"][i]))
    #img = img.normalize(imagenet_stats)
    pred_class,pred_idx,outputs = learnVGG.predict(img)
    labl.append(pred_class)


# In[ ]:


len(labl)


# ### Export submission file

# In[ ]:


sample = pd.read_csv('../input/emergency-vs-nonemergency-vehicle-classification/dataset/sample_submission_yxjOnvz.csv')


# In[ ]:


sample.head()


# In[ ]:


#create datafarme for submission & export
sample['image_names'] = testAV['image_names']
sample['emergency_or_not'] = labl
sample.to_csv('submit141.csv', index=False)


# # Thank you
