#!/usr/bin/env python
# coding: utf-8

# In[ ]:



import numpy as np
import pandas as pd
from matplotlib import pyplot as p


# In[ ]:


data = pd.read_csv("../input/pima-indians-diabetes.csv",header=None)


# In[ ]:


x = data.drop(8,axis=1).copy()
y = data[8]


# In[ ]:


x.columns = {'Pregnancies','Glucose','BloodPressure','SkinThickness','Insulin','BMI','DiabetesPedigreeFunction','Age'}
y.columns = {'y'}


# In[ ]:


x.head()


# In[ ]:


from sklearn.preprocessing import MinMaxScaler


# In[ ]:


min_max_scaler = MinMaxScaler() 
x_scaled = min_max_scaler.fit_transform(x)


# In[ ]:


x_scaled_df = pd.DataFrame(x_scaled)


# In[ ]:


x_scaled_df.columns = x.columns
x_scaled_df.head()


# In[ ]:


import seaborn as sns
from scipy import stats

x_normalized = np.log(x_scaled_df)

f , (ax1 , ax2 ) = p.subplots(1,2,figsize=(8,3))


x_normalized = x_normalized.replace(-np.inf,0)

p.sca(ax1)
sns.distplot(x_scaled_df['Glucose'])
p.sca(ax2)
sns.distplot(x_normalized['Glucose'])


# In[ ]:


from sklearn.preprocessing import normalize

x_norm = pd.DataFrame(normalize(x_scaled_df,axis=1))
x_norm.columns = x.columns 


# In[ ]:


from scipy import stats

f ,(ax1,ax2) = p.subplots(1,2,figsize=(8,5))
p.sca(ax1)
sns.distplot(x_scaled_df['Glucose'])
p.sca(ax2)
sns.distplot(x_norm['Glucose'])


# In[ ]:


from sklearn.preprocessing import StandardScaler

sscaler = StandardScaler()

x_stand = sscaler.fit_transform(x)
x_stand = pd.DataFrame(x_stand)
x_stand.columns = x.columns


# In[ ]:


f ,(ax1,ax2) = p.subplots(1,2,figsize=(8,5))
p.sca(ax1)
sns.lineplot(x_scaled_df['Glucose'],x_scaled_df['Age'])
p.sca(ax2)
sns.lineplot(x_stand['Glucose'],x_stand['Age'])


# In[ ]:


from sklearn.preprocessing import binarize

x_bin = pd.DataFrame(binarize(x))
x_bin.columns = x.columns
x_bin.head()

