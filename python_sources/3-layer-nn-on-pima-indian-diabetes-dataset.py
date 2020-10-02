#!/usr/bin/env python
# coding: utf-8

# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')
import pandas as pd
import numpy as np
import seaborn as sns

import matplotlib.pyplot as plt


# In[ ]:


df = pd.read_csv('../input/diabetes.csv')


# In[ ]:


df.head()


# In[ ]:


#Check null values & how many numerical and categorical features
df.info()


# No Null Values and All are numerical values

# In[ ]:


#the min and max values
df.describe()


# In[ ]:



sns.pairplot(df,hue= 'Outcome')


# In[ ]:


#Build a histogram of all KPIs together
_=df.hist(figsize=(12,10))


# In[ ]:


#Find the correation between the variables
sns.heatmap(df.corr(),annot=True)


# Age and Number of pregnancies are correlated (0.54); Skin Thickness and Insulin are correlated

# Glucose is correlated with Outcome

# In[ ]:


#Since variables have varied scale, we would rescale all of them, using Standard Scalar
#StandardScalar subtracts mean from all the values and divide by SD, so we scale all the values to mean = 0 and sd = 1

from sklearn.preprocessing import StandardScaler
from keras.utils import to_categorical


# In[ ]:


sc= StandardScaler()
x = sc.fit_transform(df.drop('Outcome',axis =1))


# In[ ]:


Y = df['Outcome'].values


# In[ ]:


y_cat = to_categorical(Y)


# In[ ]:


x


# In[ ]:


x.shape


# In[ ]:


y_cat


# In[ ]:


y_cat.shape


# In[ ]:




