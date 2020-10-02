#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import matplotlib.pyplot as plt #Library for general visualizations
import seaborn as sns #For more beautiful visualizations
import numpy as np #Library that handles mathematical operations
import pandas as pd #Working with .csv files
import time #General Python time library

#Magic command to the jupyter notebook that we want all visualizations to stay within the file
get_ipython().run_line_magic('matplotlib', 'inline')

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))


# Any results you write to the current directory are saved as output.


# Assigning the above file to a variable:

# In[ ]:


df = pd.read_csv("../input/data.csv")
df.head(10)


# From the above table, I can clearly see that the name of the target column is: **diagnosis**. I will first change the values to integer and into a separate variable
# 
# Even though the dataset clearly says **Missing attribute values: none**, I can see a strange column, **Unnamed: 32** which only contained **NaN** values for the first 10 rows. I will also get the details of all the columns to see the extent of the missing data. 

# In[ ]:


print(df.shape)
print(df.info())


# From above, we can clearly see that Unnamed:32 column needs to be dropped. Now to assign the target values to a **target** variable and change the values to 1's and 0's.

# In[ ]:


targets = df.diagnosis
ax = sns.countplot(targets, label="Count", palette="Set3")
M,B = targets.value_counts()
print('Malignant:',M,"Percent:",int(M/(M+B)*100),"%")
print('Benign:',B,"   Percent:",int(B/(M+B)*100),"%")
targets = np.where(targets.values == 'M', 0, 1)


# From the above plot, we can see that we have a greater percent (62%) of malignant cases in our data. Now to explore how the different features correlate to whether the diagnosis is malignant or not.
