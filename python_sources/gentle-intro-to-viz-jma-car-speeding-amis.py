#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# In[ ]:


nRowsRead = 1000 # specify 'None' if want to read whole file
df = pd.read_csv('../input/car-speeding-and-warning-signs/amis.csv')
nRow, nCol = df.shape
print(f'There are {nRow} rows and {nCol} columns')


# In[ ]:


df.head()


# In[ ]:


df.info()


# In[ ]:


categorical_cols = [cname for cname in df.columns if
                    df[cname].nunique() < 10 and 
                    df[cname].dtype == "object"]


# Select numerical columns
numerical_cols = [cname for cname in df.columns if 
                df[cname].dtype in ['int64', 'float64']]


# In[ ]:


print(categorical_cols)


# In[ ]:


print(numerical_cols)


# In[ ]:


total = df.isnull().sum().sort_values(ascending = False)
percent = (df.isnull().sum()/df.isnull().count()*100).sort_values(ascending = False)
missing_data  = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
missing_data.head(8)


# In[ ]:


df.dtypes.value_counts()


# In[ ]:


corrs = df.corr()
corrs


# In[ ]:


import matplotlib.pyplot as plt
import seaborn as sns


# In[ ]:


plt.figure(figsize = (20, 8))
sns.heatmap(corrs, cmap = plt.cm.RdYlBu_r, vmin = -0.25, annot = True, vmax = 0.6)
plt.title('Correlation Heatmap')


# In[ ]:


sns.distplot(df["speed"])


# In[ ]:


sns.countplot(df["speed"])


# In[ ]:


print ("Skew is:", df.speed.skew())
plt.hist(df.speed, color='pink')
plt.show()


# In[ ]:


target = np.log(df.speed)
print ("Skew is:", target.skew())
plt.hist(target, color='green')
plt.show()

