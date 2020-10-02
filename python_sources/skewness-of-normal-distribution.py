#!/usr/bin/env python
# coding: utf-8

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


# In[ ]:





# In[ ]:


import pandas as pd 
import numpy as np


# In[ ]:


data = pd.read_csv('/kaggle/input/usa-cers-dataset/USA_cars_datasets.csv')


# In[ ]:


data.head()


# In[ ]:


data.info


# In[ ]:


data.shape


# In[ ]:


data = data.drop(['Unnamed: 0'], axis = 1)


# In[ ]:


import seaborn as sns
data['price'].hist(grid='False')


# In[ ]:


data['price'].skew()


# Skewness is the measure of lack of symmetry.
# As, the value of skew is greater than 0.5, it doesn't lie in the range of -0.5 to 0.5 so it is Positive Skewness

# In[ ]:


# density plot
sns.distplot(data['price'], hist=True)


# In[ ]:


# checking the skewness of mileage column
data['mileage'].skew()


# In[ ]:


sns.distplot(data['mileage'], hist = True)


# From this , it is confirmed that it is Positive Skewness as value is also greater than 0.5 
# 

# There are some methods to transform the skewness :
# 

# For positive skewness 
# # 1. Log Transformations :

# In[ ]:


log_mileage = np.log(data['mileage'])
log_mileage.head()


# In[ ]:


log_mileage.skew()


# # 2. Root Transformation 

# This is the moderate effect on the distribution, and it is weaker than logarithm and cube root 

# In[ ]:


sqrt_mileage = np.sqrt(data['mileage'])
sqrt_mileage.head()


# In[ ]:


sqrt_mileage.skew()


# In[ ]:


sns.distplot(sqrt_mileage, hist = True)


# Skew value of mileage have been reduced from 7 to 1.66  

# In[ ]:


cbrt_mileage = np.cbrt(data['mileage'])
cbrt_mileage.head()


# In[ ]:


cbrt_mileage.skew()


# In[ ]:


sns.distplot(cbrt_mileage, hist= True)


# ## log > cube Root > square root 

# In[ ]:


log_mileage.skew(),cbrt_mileage.skew(), sqrt_mileage.skew()


# # Negative Skewness 

# In[ ]:


name = pd.Series(['Sameer', 'Pankaj', 'Sam' ,'Hemant' , 'Vivek','Ram',
                     'Suman', 'Anup', 'Mohit','Sandeep'])
marks = pd.Series([10,20,30,48,62,87,93,85,60,75])
data = 


# In[ ]:


cla = pd.DataFrame({'Names':name,'Marks': marks})


# In[ ]:


cla


# In[ ]:


sns.distplot(cla['Marks'], hist = True)


# In[ ]:


cla['Marks'].skew()


# As, skew value -ve  bt lie in the range  of -0.5 to 0.5 so it is symmetrical type

# For left skewness, there are some methods  like : 
# 1. Square Distribution
# 2. Cube Distribution
# 3. High Power

# In[ ]:




