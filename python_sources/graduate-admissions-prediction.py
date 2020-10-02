#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[ ]:


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns


# In[ ]:


df = pd.read_csv("../input/Admission_Predict_Ver1.1.csv")


# In[ ]:


df.head()


# In[ ]:


df.info() #to check null values


# In[ ]:


df.describe() # to check outliers


# In[ ]:


# Correlation matrix - linear relation among independent attributes and with the Target attribute

sns.set(style="white")

# Compute the correlation matrix
correln = df.corr()

# Generate a mask for the upper triangle
#mask = np.zeros_like(correln, dtype=np.bool)
#mask[np.triu_indices_from(mask)] = True

# Set up the matplotlib figure
f, ax = plt.subplots(figsize=(11, 9))

# Generate a custom diverging colormap
cmap = sns.diverging_palette(220, 10, as_cmap=True)

# Draw the heatmap with the mask and correct aspect ratio
sns.heatmap(correln,  cmap=cmap, vmax=.3, #mask=mask,
            linewidths=.5, cbar_kws={"shrink": .7})


# In[ ]:


from sklearn.preprocessing import MinMaxScaler # for performing scaling
from sklearn.model_selection import train_test_split # to split the data to train and test


# In[ ]:


x = df.rename(index = str, columns = {"Serial No." : "serial_no", "GRE Score" : "gre", "TOEFL Score" : "toefl", "University Rating" : "university_rating", "Chance of Admit" : "ChanceofAdmit",}) # to rename data columns


# In[ ]:


x


# In[ ]:


cols = list(df.columns)
xnew = df[cols[0:8]] # sample


# In[ ]:


xnew


# In[ ]:


ynew = df[cols[-1]] # target


# In[ ]:


ynew.head()


# In[ ]:


x_train, x_test, y_train, y_test = train_test_split(xnew, ynew, random_state = 0) # splitting data to train and test data


# In[ ]:


scaler = MinMaxScaler() # scaler function
x_train = scaler.fit_transform(x_train) #  scaling x_train data
x_test = scaler.fit_transform(x_test) # scaling x_test data


# In[ ]:


from sklearn.linear_model import LinearRegression # importing linear regression 

lreg =LinearRegression()  # linear regression 
lreg.fit(x_train, y_train) # applying linear regression to x_train and y_train data


# In[ ]:


print(lreg.score(x_train, y_train)) # scoring  accuracy of the train data
print(lreg.score(x_test, y_test))   # scoring accuracy of the test data

