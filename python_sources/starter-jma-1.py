#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np 
import os 
import pandas as pd


# In[ ]:


df = pd.read_csv("../input/dataset.csv")
df.head()


# In[ ]:


df.info()


# In[ ]:


df.describe()


# In[ ]:


df['Data_Value_Type'].value_counts()


# In[ ]:


df['Data_Value_Type'].value_counts(normalize=True)


# In[ ]:


df['Break_out'].value_counts()


# In[ ]:


df['Break_out'].value_counts(normalize=True)


# In[ ]:


df['Confidence_Limit_Low'].hist();


# In[ ]:


df['Confidence_Limit_High'].hist();


# In[ ]:


sns.boxplot(df['Confidence_Limit_Low'])


# In[ ]:


sns.boxplot(df['Confidence_Limit_High'])


# In[ ]:


plt.scatter(df['Confidence_Limit_High'], df['Confidence_Limit_Low']);

