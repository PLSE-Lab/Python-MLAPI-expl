#!/usr/bin/env python
# coding: utf-8

# <h1><center>An Explanatory Data Analysis on the world's GDP in lockdown </center></h1>

# # Data Preprocessing

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import xlrd
# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# <h3>Display top five rows of the dataframe</h3>

# In[ ]:


df = pd.read_excel('../input/steroid-dataset/wbdata.xlsx')
df.head()


# <h3>Get a summary of the data from the dataframe</h3>

# In[ ]:


df.describe()


# In[ ]:


print("The dimension of the dataframe is :",df.shape)


# In[ ]:


print("The list of datatypes of the values in dataframe :\n")
print(df.dtypes)


# In[ ]:


print("Detailed information of the dataframe :\n")
print(df.info())


# <h3>Checking for null values</h3>

# In[ ]:


print("Do we have any null values in this dataframe ?",df.isnull().values.any())


# In[ ]:


print("The no. of null data elements in the dataframe :",df.isnull().sum().sum())


# In[ ]:


print("A summary of the null values in dataset :\n")
print(df.isnull().sum())


# In[ ]:


print("Drop NaN values to purify the dataset :\n")
df1 = df.copy()
df1.dropna(inplace=True)
df1.info()


# In[ ]:


print("After dropping missing values, the dimension of the dataset :",df1.shape)


# # Data Visualization

# <h3> 1. Pairplot of all columns </h3>

# In[ ]:


sns.pairplot(df1)


# <h3> 2. Scatterplot of GDP </h3>

# In[ ]:


sns.scatterplot(data=df1['gdp']).set_title("GDP Values")


# <h3> 3.Scatterplot of GDP per capita vs population </h3>

# In[ ]:


sns.scatterplot(x='gdp.cap', y='population', data=df1, hue='population')


# <h3> 4. Skewed graph of population density </h3>

# In[ ]:


sns.kdeplot(df1['population'], shade=True, color='orangered')


# <h3> 5. Boxplot of all parameters </h3>

# In[ ]:


df1.plot(kind='box', subplots=True, layout=(3,3), sharex=False, sharey=False, figsize=(20, 20), color='deeppink')


# <h3> 6. Boxplot of GDP </h3>

# In[ ]:


sns.boxplot(data=df1['gdp'], color='darkturquoise')


# <h3> 7. Desnsity plot of all parameters </h3>

# In[ ]:


df1.plot(kind='density', subplots=True, layout=(3,3), sharex=False, figsize=(20, 20))


# <h3> 8. Correlation Plot of GDP parameters per capita, per population and against poulation densities </h3>

# In[ ]:


mask = np.tril(df1.corr())
sns.heatmap(df1.corr(), fmt='.1g', annot = True, cmap= 'cool', mask=mask)


# In[ ]:




