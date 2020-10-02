#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import preprocessing
from sklearn import impute

from scipy.stats import skew, norm, boxcox
from scipy.special import boxcox1p
from scipy.stats import boxcox_normmax
from scipy.stats.mstats import winsorize

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# In[ ]:


data = pd.read_csv('/kaggle/input/insurance/insurance.csv')
data.head(8)


# In[ ]:


#Null values
data.isna().sum()


# No null values, that's good

# In[ ]:


data.info()


# ## Skewness

# In[ ]:


#skewness in charges
sns.distplot(data['charges'], hist = True)
plt.title('Charges Frequency plot')
plt.ylabel('Frequency')
plt.xlabel('Charges')


# Positively skewed

# In[ ]:


# log transformation
data['charges'] = np.log1p(data['charges'])


# In[ ]:


#After log transformation 
sns.distplot(data['charges'], hist = True)
plt.title('Charges Frequency plot')
plt.ylabel('Frequency')
plt.xlabel('Charges')


# Certainly better

# In[ ]:


#Skewness in other variables (before transformation)
skews = data.skew(axis = 0)
skews


# In[ ]:


#skewness in other features
#age
sns.distplot(data['age'], hist = True)
plt.title('age Frequency plot')
plt.ylabel('Frequency')
plt.xlabel('age')


# In[ ]:


#transforming age
data['age'] = boxcox1p(data['age'], boxcox_normmax(data['age'] + 1))


# In[ ]:


#after transformation
sns.distplot(data['age'], hist = True)
plt.title('age Frequency plot')
plt.ylabel('Frequency')
plt.xlabel('age')


# In[ ]:


#bmi
sns.distplot(data['bmi'], hist = True)
plt.title('bmi Frequency plot')
plt.ylabel('Frequency')
plt.xlabel('bmi')


# Normal

# In[ ]:


#skewness in other features
sns.distplot(data['children'], hist = True)
plt.title('children Frequency plot')
plt.ylabel('Frequency')
plt.xlabel('childrem')


# In[ ]:


#transforming children
data['children'] = np.log1p(data['children'])


# In[ ]:


#after transformation
sns.distplot(data['children'], hist = True)
plt.title('children Frequency plot')
plt.ylabel('Frequency')
plt.xlabel('childrem')


# In[ ]:


#Skewness after transformation
skews = data.skew(axis = 0)
skews


# ## Outliers

# In[ ]:


f, ax = plt.subplots(figsize=(9, 8))
sns.boxplot(data = data)


# Outliers in BMI

# In[ ]:


#the clip function (replaces upper outliers with 95th quantile and lower with 5th quantiles)
data['bmi'] = data['bmi'].clip(lower=data['bmi'].quantile(0.05), upper=data['bmi'].quantile(0.95))


# In[ ]:


f, ax = plt.subplots(figsize=(9, 8))
sns.boxplot(data = data)


# In[ ]:


#Encoding categoricals
data = pd.get_dummies(data, drop_first = True)
data.head()


# ## Feature Scaling

# In[ ]:




