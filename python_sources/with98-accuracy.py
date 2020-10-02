#!/usr/bin/env python
# coding: utf-8

# Importing Packages

# In[ ]:


import pandas as pd
import numpy as np
import torch
import fastai
from fastai.vision import *
import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


# **Function to Unzip Zip Files**

# In[ ]:


from zipfile import ZipFile 
def unZip(file_name):
  with ZipFile(file_name, 'r') as zip: 
      zip.extractall() 
      print('Done!') 


#  **Function to Copy Files**

# In[ ]:


import shutil
def copyfiles(filesName,dest):
  for file in filesName:
    shutil.copy(file, dest)
    print("Copied")
    


# **Getting Files Names from Normal and Pneumonia Folder**

# In[ ]:


import glob
path_data="../input/chest_xray/chest_xray/"
filesPos= sorted(glob.glob(path_data+'train/NORMAL/*.jpeg'))
filesNeg=sorted(glob.glob(path_data+'/train/PNEUMONIA/*.jpeg'))


# Making Equal number of Files Positive and Negitive

# In[ ]:


filesNeg=filesNeg[:1300]
filesPos=filesPos[:1300]


# Folder for training data

# In[ ]:


get_ipython().system('pwd')
get_ipython().system('mkdir /kaggle/working/dataset')
get_ipython().system('mkdir /kaggle/working/dataset/NORMAL')
get_ipython().system('mkdir /kaggle/working/dataset/PNEUMONIA')


# Coping Positive and Negitive Files

# In[ ]:


copyfiles(filesPos,'/kaggle/working/dataset/NORMAL')


# In[ ]:



copyfiles(filesNeg,'/kaggle/working/dataset/PNEUMONIA')


# In[ ]:


get_ipython().system('ls /kaggle/working/dataset/NORMAL')


# In[ ]:


len(filesPos),len(filesNeg)


# Creating Data Batch 20% Validation
# 1. Image Size: 224
# 2. Flip Images : Yes
# 3. Paralel Processor : 4
# 4. Batch Size : 32

# In[ ]:


import numpy as np
np.random.seed(42)
path='/kaggle/working/dataset'
data = ImageDataBunch.from_folder(path,
        ds_tfms=get_transforms(do_flip=True),
        valid_pct=0.2,
        size=224,
        num_workers=4,
        bs=32,
        test="test").normalize()


# Creating Learner Class which is a CNN Model . Architecture is ResNet101 
# Metrics are Error rate and Accuracy

# In[ ]:


learn1 = create_cnn(data, models.resnet101, model_dir="/tmp/model/", metrics=[error_rate, accuracy],callback_fns=ShowGraph)


# Delete Layers Weights

# In[ ]:


learn1.unfreeze()


# Fitting model to Our Dataset
# Number of Epochs: 5
# Weights: Random
# Learning Rate: Random

# In[ ]:



learn1.fit_one_cycle(7)


# In[ ]:


learn1.save('notPreTrainedResnet101.pth')


# In[ ]:


interp = ClassificationInterpretation.from_learner(learn1)
interp.plot_confusion_matrix()


# In[ ]:


learn2 = create_cnn(data, models.resnet101, model_dir="/tmp/model/", metrics=[error_rate, accuracy],callback_fns=ShowGraph)


# In[ ]:


learn2.fit_one_cycle(20)


# In[ ]:


interp = ClassificationInterpretation.from_learner(learn2)
interp.plot_confusion_matrix()

