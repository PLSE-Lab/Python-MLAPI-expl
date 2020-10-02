#!/usr/bin/env python
# coding: utf-8

# In[25]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 
import IPython
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input/train/audio/"))

# Any results you write to the current directory are saved as output.


# # Listen to a specific file

# In[46]:


files = os.listdir("../input/train/audio/two/")
files[:10] #change the number to see number of files..


# In[40]:


file = "../input/train/audio/two/7d8babdb_nohash_0.wav"# change the folder and file accordingly 
IPython.display.display(IPython.display.Audio(file))


# # Listen to no.of files in a folder or directory

# In[41]:


#defining a function 
def audio_files(dir, number_of_files):
    i=0
    path ='../input/train/audio/'+dir+'/'
    files = os.listdir(path)
    for file in files:
        file =path + file
        IPython.display.display(IPython.display.Audio(file))
        i=i+1
        if i==number_of_files:
            break

#calling the function            
audio_files('no', 5)


# In[42]:


audio_files('yes', 3)


# # Listen to a specific no.of files in all directories.

# In[43]:


dirs = os.listdir("../input/train/audio/")
for dir in dirs:
    print(dir)
    audio_files(dir,1)# change the number of file as per your need
    


# #### Thank you!

# In[ ]:




