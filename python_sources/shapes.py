#!/usr/bin/env python
# coding: utf-8

# **Kindergarten level ML**
# 
# I just recently stumbled across [fast.ai](http://) webpage and not long after I realised how their motto makes a lot of sense - "Making neural nets uncool again".
# 
# Being able to learn how to write code shown below doesn't make you special. It just proves that anyone can do it. Of course we're talking of the level mentioned in the header of this notebook. Eventhough I, like many of the beginners, tried playing with cats and dogs categorising and ASL dataset, I was particularly  happy to find Four Shapes data sets.
# 
# ![image.png](attachment:image.png)
# 
# Sorting shapes is exactly what kids do, don't they?

# In[ ]:


import numpy as np
import pandas as pd
import os

from fastai.vision import *
from sklearn.model_selection import train_test_split


# In[ ]:


os.listdir('../input/shapes/')


# In[ ]:


# Get the list of images with the full path to each file

img_list = []
for dirname, _, filenames in os.walk('../input'):
        for filename in filenames:
            img_list.append(os.path.join(dirname, filename))
len(img_list)
img_list = img_list[3:] #drop first three python script files


# In[ ]:


# Label fuction definition

def get_labels(img_list):
    if '/circle/' in str(img_list):
        label = 'circle'
    elif '/triangle/' in str(img_list):
        label = 'triangle'
    elif '/star/' in str(img_list):
        label = 'star'
    else:
        label = 'square'
    return label


# In[ ]:


#Splitting data into train and test datasets

training_images, testing_images = train_test_split(img_list, train_size=0.8, test_size=0.2, random_state=0)
df_testing_images = pd.DataFrame(testing_images) #test images list in DataFrame to be used for ImageList


# In[ ]:


#define the train data to fit a model on

data = ImageDataBunch.from_name_func('', training_images, label_func=get_labels, ds_tfms=get_transforms(), valid_pct=0.2, size=24)
test_data = ImageList.from_df(df_testing_images, '')
data.classes


# In[ ]:


data.show_batch(rows=3, figsize=(6,6))


# In[ ]:


#Simple cnn is enough to get you going

model = simple_cnn((3,16,16,4))
learn = Learner(data, model)
learn.metrics=[accuracy]
learn.fit_one_cycle(5)


# In[ ]:


#Find out which learning rate is the best one 

learn.lr_find()
learn.recorder.plot()


# In[ ]:


#fit once more with chosen lr

learn.metrics=[accuracy]
learn.fit_one_cycle(5, max_lr = 1e-02)


# In[ ]:


#and lets make predictions and analyse how well they were done

preds,y,losses = learn.get_preds(test_data, with_loss=True)
interp = ClassificationInterpretation(learn, preds, y, losses)
interp.plot_confusion_matrix()


# In[ ]:


interp1 = ClassificationInterpretation.from_learner(learn)
interp1.plot_top_losses(9, figsize=(10,10))


# That's it to it.
