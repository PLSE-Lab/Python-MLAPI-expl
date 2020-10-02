#!/usr/bin/env python
# coding: utf-8

# 
#  **Beginners In Kaggle &  Natural Language Processing** 
#  
# Those who are aspired to become data scincetists. And who want to paly with the data, Here is the tutuorial for reading the text data in kaggle.

# In[ ]:


#import the basic packages which is necessary for every model.

import numpy as np
import pandas as pd 


# 1. Choose the new Kernel and Add the dataset of your own on the top right side by clicking ADD DATASET
# 2. Type the data of your known and save file name as .txt extension, And upload it in the "ADD DATASET"

# In[ ]:


#On top right side select the filename of our data.
#Here my file name is "textfile". Copy the path of that file and follow the below for reading the dataset.

myfile= pd.read_csv('../input/textdata.txt') # load data from csv or txt, this processn is applicable for both train and test datasets

print(myfile) # Shows the data  


# In[ ]:


#shows the length of the dataset.
myfile.shape 


# In[ ]:


#explains about the dataset unique words, count of words etc.
myfile.describe()


# Hope something useful for beginners.
