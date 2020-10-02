#!/usr/bin/env python
# coding: utf-8

# In this kernel we will cover how to explore a data set using statistics and vizualization.This kernel is a work in process.I will be updating the kernel in coming days.If you like my work please do vote.

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# **Importing Python Modules**

# In[ ]:


import numpy as np
import matplotlib.pyplot as plt 
import pandas as pd 
import warnings
import seaborn as sns
warnings.filterwarnings('ignore') 


# In[ ]:


df=pd.read_csv('../input/wine-quality/winequalityN.csv')
df.head()


# **Summary of Dataset**

# In[ ]:


print('Rows     :',df.shape[0])
print('Columns  :',df.shape[1])
print('\nFeatures :\n     :',df.columns.tolist())
print('\nMissing values    :',df.isnull().values.sum())
print('\nUnique values :  \n',df.nunique())


# In[ ]:


df.columns


# **Finding out which columns are catogerical**

# In[ ]:


df.select_dtypes(exclude=['int','float']).columns


# Only the column type is of catogerical type

# In[ ]:


print(df['type'].unique())


# In[ ]:


df.info()


# **Describe Data**

# In[ ]:


df.describe().T


# We can see that the mean Alcohol content is 10.49 % .
# 
# Average Quality of the win is 5.81

# **Histograms**

# In[ ]:


df.hist(column='alcohol',bins=15,grid=False,figsize=(10,6),color='r')
plt.ioff()


# **Histogram with Seaborn**

# In[ ]:


sns.distplot(df['alcohol'],bins=25,kde=False,color='r')
plt.ioff()


# **KDE Histogram**

# In[ ]:


sns.distplot(df['alcohol'],bins=25,kde=True,color='r')
plt.ioff()


# In[ ]:


df['alcohol'].value_counts().head()


# Top five alcohol content perentage is 9.5,9.4,9.2,10,10.5

# **Styling and Axis Labels**

# In[ ]:


import matplotlib.pyplot as plt
sns.distplot(df.alcohol)
plt.xlabel('Alcohol Percentage')
plt.ylabel('Count')
plt.title('Alcohol Content')
plt.ioff()


# **Use Seaborn Style**

# In[ ]:


sns.set_style('dark')
sns.distplot(df.alcohol,bins=15)
plt.ioff()


# **All Histograms Together**

# In[ ]:


df[['fixed acidity','volatile acidity','citric acid','residual sugar','chlorides','free sulfur dioxide','density','pH','sulphates','alcohol','quality']].hist(figsize=(10,8),bins=40,color='r',linewidth='1.5',edgecolor='k')
plt.tight_layout()
plt.show()


# **Scatter plot**

# In[ ]:


sns.lmplot(x='alcohol',y='fixed acidity',data=df)
plt.ioff()


# In[ ]:


sns.lmplot(x='alcohol',y='density',data=df,fit_reg=False)
plt.ioff()


# **Segregation using HUE**

# In[ ]:


sns.lmplot(x='alcohol',y='chlorides',data=df,fit_reg=False,hue='quality')
plt.ioff()


# In[ ]:


sns.lmplot(x='alcohol',y='chlorides',data=df,fit_reg=False,hue='type')
plt.ioff()


# **Understanding Percentile** 

# In[ ]:


print(df['alcohol'].quantile(0.1))
print(df['alcohol'].quantile(0.5))
print(df['alcohol'].quantile(0.9))
print(df['alcohol'].quantile(0.99))


# 10 % Wine has 9.1 % Alcohol 
# 
# 10.3 % Wine has 10.3 % Alcohol 
# 
# 12.3% Wine has 12.3 % Alcohol
# 
# 13.4% Wine has 13.4 % Alcohol

# **What is maximum Value of alcohol?**

# In[ ]:


df['alcohol'].max()


# In[ ]:


df['alcohol'].quantile(([0.05,0.95]))


# **Box Plots and outliers**

# In[ ]:


import matplotlib.pyplot as plt
from PIL import Image
get_ipython().run_line_magic('matplotlib', 'inline')
import numpy as np
img=np.array(Image.open('../input/box-plot-1/BOX_PLOT.PNG'))
fig=plt.figure(figsize=(10,10))
plt.imshow(img,interpolation='bilinear')
plt.axis('off')
plt.show()


# maximum:Q3 + 1.5 * IQR
# minimum:Q1-1.5 * IQR

# In[ ]:


print(df['alcohol'].quantile(([0.25,0.75])))
sns.boxplot(data=df['alcohol'])
plt.ioff()


# **Box plot of all Features**

# In[ ]:


plt.figure(figsize=(20,10))
sns.boxplot(data=df,palette='Set3')
plt.ioff()


# **Violin Plots**

# In[ ]:


sns.violinplot(data=df['alcohol'])
plt.ioff()


# **Bar Plot**

# In[ ]:


plt.figure(figsize=(12,8))
sns.countplot(x='quality',data=df,hue='type')
plt.ioff()


# **Density Plot**

# In[ ]:


sns.kdeplot(df.alcohol,df.density)
plt.ioff()


# **Joint Distribution plot**

# In[ ]:


sns.jointplot(x='alcohol',y='density',data=df)
plt.ioff()


# **Factor Plots and Bee Swarm Plots**

# In[ ]:


#g=sns.factorplot(x='free sulfur dioxide',y='alcohol',data=df,col='quality',hue='quality',kind='point')
#g.set_xticklabels(rotation=-45)
#plt.ioff()


# **Emprical Cumulative Distribution function (ECDF)**

# In[ ]:


import numpy
x=np.sort(df['alcohol'])
y=np.arange(1,len(x)+1)/len(x)
plt.plot(x,y,marker='.',linestyle='none')
plt.margins(0.05)
plt.xlabel('Percent of Alcohol in Wine')
plt.ylabel('ECDF')
plt.grid(True)
plt.show()


# In[ ]:


print(df['alcohol'].quantile(([0.2,0.8])))


# In[ ]:




