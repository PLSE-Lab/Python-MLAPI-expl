#!/usr/bin/env python
# coding: utf-8

# In[ ]:




import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory
import os
import warnings
warnings.filterwarnings("ignore")


# In[ ]:


df=pd.read_csv("../input/cholera-dataset/data.csv")


# In[ ]:


df.head()


# In[ ]:


df.columns=["country","year","case","death","fatality_rate","who_region"]


# In[ ]:


df.head()


# In[ ]:


df.tail()


# In[ ]:


df.describe()


# In[ ]:


df.info()


# In[ ]:


df.isna().sum()


# In[ ]:


df[df.case=="3 5"]


# In[ ]:


df["case"]=df["case"].apply(lambda x:str(x).replace('3 5','35')if '3 5' in str(x) else str(x))


# In[ ]:


df=df.dropna()


# In[ ]:


df.head()


# In[ ]:


df.tail()


# In[ ]:


df.isna().sum()


# In[ ]:


df.death.unique()


# there are unknown value...

# In[ ]:


df[df.death=="Unknown"]


# In[ ]:


df.drop([761],inplace=True)


# In[ ]:


df[df.death=="0 0"]


# In[ ]:


df.drop([1059],inplace=True)


# In[ ]:


df.info()


# In[ ]:


df["case"]=df["case"].apply(lambda x: int(x))
df["death"]=df["death"].apply(lambda x: int(x))
df["fatality_rate"]=df["fatality_rate"].apply(lambda x: float(x))


# In[ ]:


#visualize the correlation
plt.figure(figsize=(15,10))
sns.heatmap(df.iloc[:,0:15].corr(), annot=True,fmt=".0%")
plt.show()


# case and death %44
