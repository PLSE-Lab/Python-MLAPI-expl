#!/usr/bin/env python
# coding: utf-8

# Feature engineering is one of the important aspects of machine learning.He we will be covering Featuring Scaling.We will be discussing about different types of scaling.Scaling is done because it helps to improve the performance of the model.If you like the kernel please do vote.

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# ### Importing the Data

# In[ ]:


df = pd.read_csv('../input/weight-height/weight-height.csv')
df.head()


# ### Converting the Categorical into Numerical Value

# In[ ]:


pd.get_dummies(df['Gender'],prefix = 'Gender').head()


# ### Feature Transformation

# **1.Rescale with fixed factor**

# In[ ]:


df['Height (feet)'] = df['Height']/12.0
df['Weight (100 lbs)'] = df['Weight']/100


# In[ ]:


df.describe().round(2)


# So now we can see that Height values are scaled in between 4.52 to 6.58.
# 
# The weight values are in the range 0.65 to 2.70 

# **2.MinMax normalization**

# In[ ]:


from sklearn.preprocessing import MinMaxScaler

mms = MinMaxScaler()
df['Weight_mms'] = mms.fit_transform(df[['Weight']])
df['Height_mms'] = mms.fit_transform(df[['Height']])
df.describe().round(2)


# So now we can see that we have scaled the Height and Weight values.So value of Height and Weight now is in the range of 0 to 1.

# **3.Standard Normalization**

# In[ ]:


from sklearn.preprocessing import StandardScaler

ss = StandardScaler()
df['Weight_ss'] = ss.fit_transform(df[['Weight']])
df['Height_ss'] = ss.fit_transform(df[['Height']])
df.describe().round(2)


# So after doing a standard scaling our values of Height and Weight is scaled between the Standard Deviation for the feature value.
# 

# ### Plotting the Scaled Featured

# In[ ]:


import matplotlib.pyplot as plt
plt.figure(figsize =(15,5))

for i, feature in enumerate(['Height','Height (feet)','Height_mms','Height_ss']):
    plt.subplot(1,4,i+1)
    df[feature].plot(kind='hist',title = feature)
    plt.xlabel(feature)


# If We use this scaled data dor making our Prediction our Results accuracy would be better.
