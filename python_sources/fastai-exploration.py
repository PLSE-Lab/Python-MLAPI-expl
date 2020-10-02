#!/usr/bin/env python
# coding: utf-8

# Abhishek Lalwani
# Slack handle - @Abhishek Lalwani
# Email ID - abhisheklalwani96@gmail.com
# As the name of the Kernel suggests, with this Hackathon, I wanted to explore FastAI and the benefits which it proposes along with it's shortcomings.
# 1. Model Selection
# Given the ease of use FastAI provides, I decided to try out multiple pre-trained models (for a small number of epochs) before arriving at DenseNet as the clear winner in terms of the loss and the validation accuracy achieved. The other models which I tried out were Alexnet,Squeezenet and resnet152.
# 2. Hyperparameter
# I used lr_find to find the optimal learning rate
# 3. Transforms and Normalization
# I decided to try out the default transforms as well as some custom transforms better suited to this dataset such as random_resize_crop etc. Higher accuracy was achieved with default transforms. For Normalizattion purposes, I tried using imagenet_stats, but shifted to using batch values only for higher accuracy.
# 
# Since the split of training and validation was already provided and due to shortage of time, I wasn't able to experiment with that.
# I initially also planned to submit a kernel which involved another approach quite similar to what was taught in our first lesson (Intro to Deep Learning with PyTorch) but due to shortage of time wasn't able to work on it.

# In[ ]:


#Getting all the necessary imports
import numpy as np
import pandas as pd
from pathlib import Path
from fastai import *
from fastai.vision import *
import torch
import os


# In[ ]:


#Creating the data path variable and initializing the transforms
data_folder = Path("../input/flower_data/flower_data")
trfms = get_transforms(do_flip=True, flip_vert=True, max_rotate=10.0, max_zoom=1.1, max_lighting=0.2, max_warp=0.2, p_affine=0.75, p_lighting=0.75)


# In[ ]:


#creating the data loader
data = (ImageList.from_folder(data_folder)
        .split_by_folder()
        .label_from_folder()
        .add_test_folder("../input/test set/test set")
        .transform(trfms, size=128)
        .databunch(bs=64, device= torch.device('cuda:0'))
        .normalize())


# In[ ]:


#Testing the data loader
data.show_batch(3, figsize=(6,6), hide_axis=False)


# In[ ]:


#defining the learner
learn = cnn_learner(data, models.densenet161, metrics=[error_rate, accuracy], model_dir = "../../../working") #Using densenet as discussed above


# In[ ]:


#finding the learning rate
learn.lr_find(stop_div=False, num_it=200)


# In[ ]:


#plotting loss against learning rate
learn.recorder.plot(suggestion = True)
min_grad_lr = learn.recorder.min_grad_lr


# In[ ]:


#using the learning rate and starting the training
lr = min_grad_lr
learn.fit_one_cycle(60, slice(lr)) #For final model, keep number of epochs = 60


# In[ ]:


#Saving the model
learn.export(file = '../../../working/export.pkl')


# In[ ]:


#Reloading the model into the memory and using it over test data
learn = load_learner(os.getcwd(), test=ImageList.from_folder('../input/test set/test set')) #pointing the learner towards the test data


# In[ ]:


#Getting the labels from the JSON
import json
with open('../input/cat_to_name.json') as f:
  conversion_data = json.load(f)


# In[ ]:


#Creating a final list with file name, prediction category and the corresponding name
final_result = []
for i in range (len(learn.data.test_ds)):
    filename = str(learn.data.test_ds.items[i])[27:]
    pred_category  = int(learn.predict(learn.data.test_ds[i][0])[1])
    category_name = conversion_data[str(pred_category)]
    final_result.append((filename, pred_category, category_name))


# In[ ]:


#Sorting the list alphabetically
final_result = sorted(final_result,key=lambda x: x[0])


# In[ ]:


#Saving the Final Output to a CSV
final_output = pd.DataFrame(final_result, columns=["Filename", "Predicted_Category","Category_Name"])
final_output.to_csv('final_output.csv', index=False)


# In[ ]:


#Checking that the CSV is created properly
test_csv = pd.read_csv("final_output.csv")
test_csv


# In[ ]:




