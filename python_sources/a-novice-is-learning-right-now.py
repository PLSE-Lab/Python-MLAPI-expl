#!/usr/bin/env python
# coding: utf-8

# **Just Simple DATA Analysis**

# In[2]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 
#LEAVE IT ALONE
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# ****IMPORT tHE FILES. I KNOW Without bold YOu won't REad ME****

# In[4]:


df1=pd.read_csv('../input/X_train.csv')  #This happens when comptetion maker try to be smart give x_train &Y_train to help
df2=pd.read_csv('../input/y_train.csv')  #but it ends up confusing you. 
test=pd.read_csv('../input/X_test.csv')
sample=pd.read_csv('../input/sample_submission.csv')


# ***Lets Check that shape and size!!!!***

# In[6]:


print((df1.shape),(df2.shape),(test.shape),(sample.shape))


# ****Something is bad !! Why are X_train and Y_train not simliar ?check data and columns quick!****

# In[7]:


df1.head()


# In[10]:


df1.info()


# In[11]:


df2.head()


# In[12]:


df2.info()


# In[13]:


sample.head()


# In[14]:


test.head()


# **OK !!! SO X_train have data Y_train have index and target. Sample will be used for submission**

# **LEts SEE Unique types of surfaces we have???**

# In[15]:


df2.surface.nunique()


# **TO BE COntinued......**
