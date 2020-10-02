#!/usr/bin/env python
# coding: utf-8

#  ## **Introduction**
# 
# ---
# 
# 

# #                            Diabetic Rectiopathy detection using Deep learning
# 
# ---
# 
# 
#                          
#                          
#   Diabetic rectiopathy is an disease of eye caused to an person due to diabetes,it is the condition in which person's eye is damaged due to mellitus present in human eye. It is basically occurs when the blood form several droplets on to the retina and it remain collected at that perticular place and this affect the person and if it occur in more extent then an person can lead towards blindliness due to the cause called Diabetic Rectiopathy.
#    present in human eye.
#   
# >![](http://www.koreabiomed.com/news/photo/201710/1687_1860_3515.gif) 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# here is an image of an normal person retina image and in front of it an image of an person having diabetic rectiopathy.
# 
# 
# 
# 
# > ![](https://afamilyoptician.co.uk/wp-content/uploads/2017/05/diabetic-retinopathy-v01.png)
# 
# 
# ---
# 
# 
# 
# # Diabetic rectiopathy has several stages
# 
# ---
# 
# 
# 
# 
# 0 - No DR
# 
# 1 - Mild
# 
# 2 - Moderate
# 
# 3 - Severe
# 
# 4 - Proliferative DR
# 
# 
# ---
# 
# 
# # Content
# 
# 
# ---
# 
# 
# 
# * Over 285 million peoples are suffering from Diabetic rectiopathy world wide out of which 31.7 million from india alone.
# 
# * Approximately 1 in 3 people living with diabetes have some degree of DR and 1 in 10 will develop a vision threatening form of the disease. DR is the leading cause of vision loss in working age adults (20- 65 years).
# 
# * In front of this there are very few doctors to analize the disease.
# 
# * So Deep learning can help to solve this problem
# here is an model that predicts whether an person has diabetic rectiopathy or not on the basis of there retina images.
# 

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
#for dirname, _, filenames in os.walk('/kaggle/input'):
     #   print(os.path.join(dirname, filename))
files = os.listdir("../input")
print(files)
print('trainlabels.csv' in files)
print(len(files))
# Any results you write to the current directory are saved as output.


# # **Importing required libraries**
# 
# 
# ---
# 
# 
# 
# First of all import the required libraries
# I am using fastai library for my project at this time as this an type of image classification problem I am using fatai.vision to solve the problem.

# In[ ]:


from fastai import *
from fastai.vision import *
import matplotlib as plt
import pandas as pd
from fastai.widgets import ClassConfusion
from fastai.widgets import *


# > Checking if an GPU is enabled or not
# Make sure that cuda is installed and is available.

# In[ ]:


print('Make sure cuda is installed:', torch.cuda.is_available())
print('Make sure cudnn is enabled:', torch.backends.cudnn.enabled)


# # **Reading the data**
# 
# 
# ---
# 
# 
# Read the data which is given in an csv 

# In[ ]:


train_df =  pd.read_csv("../input/aptos2019-blindness-detection/train.csv")
valid_df =  pd.read_csv("../input/aptos2019-blindness-detection/test.csv")


# Look at the data i.e dataframe of train images and also for the test images.

# In[ ]:


train_df.head(10)


# In[ ]:


valid_df.head(10)


# > Look through the training data

# In[ ]:


train_df['diagnosis'].hist(figsize = (10, 5))


# # **Transforming the images**
# 
# 
# ---
# 
# 
# 
# Transforming the images for bettor model training 
# 
# 
# Transform the images of same type
# Example all the images are initially not of the same size and shape so we have to transform them in specific order for the better trainig the neural network.
# Here it rotate the image either vertical or flip or zoom it according to its need.

# In[ ]:


#tfms = get_tranforms(do_flip=True,)
tfms=get_transforms(do_flip = True,flip_vert = True,max_rotate=360,max_zoom = 1.1)


# # **Creating Databunch**
# 
# 
# ---
# 
# 
# 1. Now the data is ready to Bunch them together to fit the model for Trainig.
# 2. Here I use ImageList to bunch them together.
# 3. Am also splitting them by and random.
# 4.  labbelling them according to the indexes.
# 5.  and finnaly normalize them here normalization means arranging the things in specific order 

# In[ ]:


data = (ImageList.from_df(train_df,"../input/aptos2019-blindness-detection/train_images",suffix='.png').
       split_by_rand_pct(0.1).
       label_from_df(1).
       transform(tfms,size=256).
       databunch(bs = 16).
       normalize(imagenet_stats))


# > Display the data
# look at the images after listing them together according to the batch size.
# * Example if I choose batch size of 16 then the first 16 images will be shown here.

# In[ ]:


data.show_batch()


# # **Creating an learner**
# 
# 
# ---
# 
# 
# making an Architecture
# create learn(A object for the neural network) here I am choosing Pretrained Model resnet101 for my problem to solve.

# In[ ]:


learn = cnn_learner(data, models.resnet101, metrics=accuracy,model_dir="/kaggle/working")


# # **Learning Rate finder**
# 
# 
# ---
# 
# 
# For finding appropriate learning rate we go through the data once and plot the graph of it.

# In[ ]:


learn.lr_find()


# plot the graph of learning rate finder.

# In[ ]:


learn.recorder.plot()


# # **Model Training**
# 
# 
# ---
# 
# 
# 
# Here learning rate is slice(2e-5,2e-3) which means first layer will have the learning rate as 2e-5 and last layer(not an output layer) will have learning rate of 2e-3 other all hidden layers will have learning rate of in between 2e-5 to 2e-3

# In[ ]:


learn.fit_one_cycle(5,slice(2e-5,2e-3),wd=0.1,moms=(0.8,0.9))


# Train a bit more

# In[ ]:


learn.fit_one_cycle(2,max_lr=slice(2.5e-3),wd=0.1,moms=(0.8,0.9))


# In[ ]:


learn.fit_one_cycle(2,slice(2.5e-5),wd=0.01,moms=(0.8,0.9))


# In[ ]:


learn.fit_one_cycle(6,max_lr=slice(1e-3,1e-4),wd=0.1,moms=0.9)


# # **Save the model**
# 
# 
# ---
# 
# 
# Now we got enough accurate results so save the Trained weights.

# In[ ]:


learn.save('stage-1')


# Loading the Trained Model named "stage-1"

# In[ ]:


learn.load('stage-1')


# Freeze the weights for not furthur modification

# In[ ]:


learn.freeze()


# # **Plot the losses**
# 
# 
# ---
# 
# 

# In[ ]:


learn.recorder.plot_losses()


# show results will display the result of any batch in this case first batch.

# In[ ]:


learn.show_results()


# show all predictions of Test data

# In[ ]:


learn.get_preds()


# > ClassificationInterpretation

# # **Classification Interpretation**
# 
# 
# ---
# 
# 

# In[ ]:


interp = ClassificationInterpretation.from_learner(learn)

losses,idxs = interp.top_losses()

len(data.valid_ds)==len(losses)==len(idxs)


# 

# In[ ]:


interp.plot_top_losses(9, figsize=(15,11))


# # **Confusion matrix**
# 
# 
# ---
# 
# 

# In[ ]:


interp.plot_confusion_matrix()


# # **Most confused**
# 
# 
# ---
# 
# 

# In[ ]:


interp.most_confused()


# > Transforming the images for prediction

# In[ ]:


Tf = partial(Image.apply_tfms,tfms=get_transforms(do_flip=True, flip_vert = True)[0][1:]+get_transforms(do_flip=True, flip_vert = True)[1],size = 512)  


# # **Reading the submission.csv file**
# 
# 
# ---
# 
# 

# In[ ]:


sub = pd.read_csv("../input/aptos2019-blindness-detection/sample_submission.csv")


# In[ ]:


print(sub)


# # **Testing on an single image**
# 
# 
# ---
# 
# 

# In[ ]:


img = open_image("../input/aptos2019-blindness-detection/test_images/020f6983114d.png")
pre = learn.predict(img)
x = pre[1]
x = int(x)
print(x)


# > Predicting the images and writting the prediction on submission.csv file for submission

# In[ ]:



for i in range(len(sub.id_code)):
    s=0
    id = sub.id_code[i]
    img=open_image("../input/aptos2019-blindness-detection/test_images/"+sub.id_code[i]+".png")
    """
    for i in range(10):
            Img = Tf(img)
            p = learn.predict(Img)
            p = p[1]
            p = int(p)
            #print(p) 
            s=s+p
    """
            
    Img = Tf(img)
    s = learn.predict(Img)
    s = s[1]
    s = int(s)
    print(s)
    sub.diagnosis[i]=s
    print(sub.diagnosis[i])


# In[ ]:



sub.to_csv("submission.csv",index=False)


# # **Submission.csv file**
# 
# 
# ---
# 
# 

# In[ ]:


print(sub) 


# In[ ]:




