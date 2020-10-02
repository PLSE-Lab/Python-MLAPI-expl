#!/usr/bin/env python
# coding: utf-8

# ## Introduction
# Greetings kaggler!!. Below code is to help you getting started with this dataset.

# ## Exploratory Analysis
# 

# In[ ]:


from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt # plotting
import numpy as np # linear algebra
import os # accessing directory structure
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from tqdm import tqdm, tqdm_notebook
import cv2
import sys


# In[ ]:


#beautiful library to get tree view of folders
get_ipython().system('apt-get install tree')


# There is 0 csv file in the current version of the dataset:
# 

# Take a peek at how files are arranged...

# In[ ]:


get_ipython().system("tree -L 1 '../input'")


# In[ ]:


#view 5 files from each folder
for dirname, _, filenames in os.walk('/kaggle/input'):
    count = 0
    for filename in filenames:
        if count > 5:
            break
        print(os.path.join(dirname, filename))
        count = count+1


# In[ ]:


#sample image
plt.imshow(cv2.imread("../input/training/0_101.jpg"))


# ### You will be working with evaluation, training and validation folders
# 
# File naming has been done in following fashion:- class_someImagenumber.jpg  (e.g. 3_49.jpg)

# So loading data would involve extraction of class from file name.
# Another major point:
#     1. Either you can load complete dataset at once
#     2. Or Loading a bach of images at a time to train and repeat the process.
# 
# Given that images would occupy large RAM space, it's better to go ahead with step 2. 

# I would demonstrate step 1 here and for training dataset.

# In[ ]:


folder_train = "../input/food-11/training/"
training_data = []
training_target = []
count = 0

#loop through all the files in train folder
for files in tqdm_notebook(os.listdir(folder_train)):
    img = cv2.imread(os.path.join(folder_train,files))
    #reshape all image to same size
    img = cv2.resize(img,(224,224) )
    #scale pixel values for training process to be efficient
    img = img/255
    #ignore images with different size
    if(img.shape != (224,224,3)):
        print("error case", img.shape)
        print("file name: ",files)
        continue
    #letter before "_" has the class number 
    target = int(files.split("_")[0])
    training_data.append(img)
    training_target.append(target)
    
    #we will just load 5k images
    count = count + 1
    if(count>5000):
        break

print("size:(in mb) ",sys.getsizeof(training_data)/1048576)
training_data = np.array(training_data)
print("size:(in mb) ",sys.getsizeof(training_data)/1048576)
training_data = training_data.reshape(-1,224,224,3)
training_target = np.array(training_target)
print("training data shape",training_data.shape)
print("target for training data shape", training_target.shape)


# ## Conclusion
# As you can see, on loading training dataset only, it's taking a reasonable space of RAM. As suggested, load data in batches. You can either write your own custom data generator or use keras ImageDataGenerator. 
# Happy Kaggling!

# In[ ]:




