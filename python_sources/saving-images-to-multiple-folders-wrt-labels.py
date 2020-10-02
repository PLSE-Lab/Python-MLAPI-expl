#!/usr/bin/env python
# coding: utf-8

# In[311]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os
print(os.listdir("../input/"))


# Check for contents in CSV File

# In[312]:


label_data = pd.read_csv('../input/train.csv')
label_data=label_data.loc[:30]
label_data.head()


# The file containes Image names and labels, lets extract the names for later use

# In[313]:


label_names=label_data.iloc[:,0]
label_names.head()


# Hot encoding labels to keep the problem general, as some data sets have multiple labels shared as multiples columns instead of one.

# In[314]:


expand_labels=pd.get_dummies(label_data.Id)
expand_labels.head()


# converting labels to boolean so thats its easier to find their respective index later, and merging them to image names again.

# In[315]:


expand_label_bool=expand_labels.astype('bool')
frames=[label_names,expand_label_bool]
dataset=pd.concat(frames,axis=1)
dataset.head()


# Extracting label names, as we will be using them later to name folders

# In[316]:


#Create separate folders for each label
#to get names for each folder
folder_names=label_data.Id.unique()
folder_names


# Specify path to 'compy from' and to 'copy to'. For this particular case its

# In[317]:


import shutil

copy_from='../input/train'
copy_to='../working/temp/'


# You can change the path according to your requirement.
# 
# If Temp folder is already created, delete and recreate it, to avoid false data. 

# In[318]:


if os.path.isdir(copy_to):
    shutil.rmtree(copy_to)
    os.makedirs(copy_to)
else:
    os.makedirs(copy_to)
os.listdir(copy_to)


# Adding Label Name Folders to Temp folder

# In[319]:


for name in folder_names:
    #create folder
    os.mkdir(copy_to+name)
os.listdir(copy_to)


# Save image files to their respective label folders

# In[320]:


import shutil

for name in folder_names:
    files=dataset.Image[dataset[name]==True]
    for file in files:
        path_from='../input/train/train/'+file
        path_to='../working/temp/'+name+'/'+file
        shutil.copyfile(path_from, path_to) 


# In[321]:


os.listdir('../working/temp/')


# In[322]:


os.listdir('../working/temp/new_whale//')


# In[ ]:





# In[ ]:




