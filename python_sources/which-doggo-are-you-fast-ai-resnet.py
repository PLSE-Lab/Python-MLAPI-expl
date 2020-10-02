#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)


from fastai.vision import *
from fastai.metrics import error_rate,accuracy



# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory
import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[ ]:


import gc
gc.collect()


# Image size = 224, Batch Size = 64, path for model and random seed

# In[ ]:


size = 224
bs = 64
path = "../input/"
np.random.seed(1)


# path_img is where we will find all the image folders with labels as name of these folders

# In[ ]:


path_anno = path+'annotations/Annotation/'
path_img = path+'images/Images'


# **Inspecting the data**

# In[ ]:


labels = os.listdir(path_img)
print("No. of labels: {}".format(len(labels)))
print("-----------------")

for label in labels:
    print("{}, {} files".format(label, len(os.listdir(path_img+'/'+label))))


# Randomly displaying 12 Images

# In[ ]:


import matplotlib.pyplot as plt
from PIL import Image

fig, ax = plt.subplots(nrows=3, ncols=4,figsize=(20, 10))
fig.tight_layout()
cnt = 0
for row in ax:
    for col in row:
        image_name = np.random.choice(os.listdir(path_img+ '/' + labels[cnt]))
        im = Image.open(path_img+"/{}/{}".format(labels[cnt],image_name))
        col.imshow(im)
        col.set_title(labels[cnt])
        col.axis('off')
        cnt += 1
plt.show();


# Since our images are placed in folders whose names correspond to the image labels, we will use the ImageDataBunch.from_folder() function to create an object that contains our image data. 
# 
# A function argument called get_transforms() which returns a list of available image transformations upon call.
# 
# Fast.ai can automatically split our data into train and validation sets using valid_pct = 0.2(20%), so we don't even need to create these on our own.
# 
# Size refers to the Image size and bs to batch size.
# 
# To normalize the data in our object, we simply call normalize() on the object. Since we will be using a ResNet architecture for our model which was trained on ImageNet, we will be using the ImageNet stats.

# In[ ]:


data = ImageDataBunch.from_folder(path_img, 
                                  ds_tfms=get_transforms(),
                                  valid_pct=0.2, 
                                  size=size, 
                                  bs=bs).normalize(imagenet_stats)


# In[ ]:


data.show_batch(rows=3, figsize=(20,10))


# In[ ]:


print(data.classes)
len(data.classes),data.c


# Now we can create a learner object, we are using resnet34 as our base model and using metrics accuracy and error rate. 
# Defining the callback function ShowGraph simply tells the learner that it should return a graph for whatever it does, which seems very useful for seeing whether the model is still improving.
# We are assigning a model directory as in kaggle the "..input/" is read only and thus creating this temporary folder will allow us to change the location of the learner. (Ignore this parameter if not using Kaggle)

# In[ ]:


learn = cnn_learner(data, models.resnet34, metrics=[accuracy,error_rate], callback_fns=ShowGraph ,model_dir="/tmp/model/")


# In[ ]:


learn.model


# In[ ]:


learn.fit_one_cycle(4)


# In[ ]:


interp = ClassificationInterpretation.from_learner(learn)

losses,idxs = interp.top_losses()

len(data.valid_ds)==len(losses)==len(idxs)


# We can visualize the most incorrect predictions to check model performance.

# In[ ]:


interp.plot_top_losses(6, figsize=(25,20));


# Using most_confused we can find where the algorithm is making most mistakes. (minimum value of mistakes = 10)

# In[ ]:


interp = ClassificationInterpretation.from_learner(learn)
interp.most_confused(min_val=10)


# In[ ]:


interp.plot_confusion_matrix(figsize=(40,40), dpi=60)


# In[ ]:


learn.fit_one_cycle(1)


# In[ ]:


learn.lr_find()
learn.recorder.plot()


# In[ ]:


learn.unfreeze()
learn.fit_one_cycle(2, max_lr=slice(1e-6,1e-4))


# We will now try predicting with resnet50 as base model but we will need to increase image size, the more the image size the better the performance but it becomes more computational. We will use a smaller batch size due to this.

# In[ ]:


data = ImageDataBunch.from_folder(path_img, 
                                  ds_tfms=get_transforms(),
                                  valid_pct=0.2, 
                                  size=299, 
                                  bs=bs//2).normalize(imagenet_stats)


# We will try to make better predictions using ResNet50.

# In[ ]:


learn = cnn_learner(data, models.resnet50, metrics=[accuracy,error_rate], callback_fns=ShowGraph ,model_dir="/tmp/model/")


# In[ ]:


learn.lr_find()
learn.recorder.plot()


# In[ ]:


learn.fit_one_cycle(8)


# In[ ]:


interp = ClassificationInterpretation.from_learner(learn)
interp.most_confused(min_val=10)


# In[ ]:


interp.plot_confusion_matrix(figsize=(40,40), dpi=60)


# We can clearly see the difference between both ResNet models, now we will try to tune the model.
# If tuning does not improve in predicting we can load the previous model using learn.load('stage-1-50').

# In[ ]:


learn.save('stage-1-50')


# Tuning the learner.

# In[ ]:


learn.unfreeze()
learn.fit_one_cycle(3, max_lr=slice(1e-6,1e-4))


# In[ ]:


interp = ClassificationInterpretation.from_learner(learn)
interp.most_confused(min_val=10)


# In[ ]:


interp.plot_confusion_matrix(figsize=(40,40), dpi=60)

