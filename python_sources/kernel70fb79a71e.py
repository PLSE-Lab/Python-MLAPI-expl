#!/usr/bin/env python
# coding: utf-8

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


# In[ ]:


#importing the required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')


# In[ ]:


#reading the file
df = pd.read_excel('../input/Train.xlsx')


# In[ ]:


#checking the shape of the dataframe
df.shape


# In[ ]:


#Summary of the dataset
df.describe()


# In[ ]:


#displaying first 5 rows
df.head(5)


# In[ ]:


#displaying sumary of the dataset
df.describe()


# In[ ]:


#setting options to display only two points after float
pd.options.display.float_format = '{:.2f}'.format


# In[ ]:


#converting all the column names to lower case
df.columns = [x.lower().replace(' ','_') for x in df.columns]


# In[ ]:


#checking if columns names have been converted
df.head()


# In[ ]:


#writing a function to get the null values and unique values in the dataset
def levels(df):
    return (pd.DataFrame({'dtype':df.dtypes, 
                          'levels':df.nunique(), 
                          'levels':[df[x].unique() for x in df.columns],
                         'null_values':df.isna().sum(),
                         'unique':df.nunique()}))


# In[ ]:


levels(df)


# In[ ]:


#Checking evaluationvalue distribution
df_above_threshold = df.copy()
import seaborn as sns
plt.figure(figsize = (15,6))
sns.distplot(df.propertyevaluationvalue)
plt.show()


# Since the dat is sparse setting the limit of threshold to be 1250000

# In[ ]:


df = df[df.propertyevaluationvalue<=1250000]
df.reset_index(drop = True, inplace = True)


# In[ ]:


#checking distribution after setting threshold
plt.figure(figsize = (15,6))
sns.distplot(df.propertyevaluationvalue)
plt.show()


# In[ ]:


df_above_threshold = df_above_threshold[df_above_threshold.propertyevaluationvalue>1250000]
plt.figure(figsize = (15,6))
sns.distplot(df_above_threshold.propertyevaluationvalue)
plt.show()


# In[ ]:


#checking for log distribution
plt.figure(figsize = (15,6))
sns.distplot(np.log(df_above_threshold.propertyevaluationvalue))
plt.show()


# In[ ]:


#checking the shape of the dataset
df.shape


# In[ ]:


#mean and median are almost same hence no outliers in the propery evaluation column
df.propertyevaluationvalue.describe()


# In[ ]:


#Checking for unique values in the columns
len(df.propertyid.unique())


# Since all the values in the propertyid are unique dropping the column

# In[ ]:


# df.drop('propertyid', axis = 1, inplace = True)


# In[ ]:


#checking unique values in borugh
df.borough.unique()


# In[ ]:


df.borough = df.borough.astype('category')


# In[ ]:


#price variation per borough
df.groupby('borough')['propertyevaluationvalue'].mean().plot(kind = 'bar')


# In[ ]:


df.groupby('borough')['propertyevaluationvalue'].median().plot(kind = 'bar')


# In[ ]:


plt.figure(figsize = (15,6))
sns.boxplot(x = df.borough, y = df.propertyevaluationvalue)
plt.show()


# In[ ]:




