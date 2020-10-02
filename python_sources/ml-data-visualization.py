#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
from sklearn import linear_model
from sklearn.model_selection import train_test_split
import statsmodels.api as sm

from matplotlib import rcParams#Size of plots 
import plotly as py
import cufflinks
from tqdm import tqdm_notebook as tqdm



# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# In[ ]:


path = "../input/novel-corona-virus-2019-dataset/covid_19_data.csv"
data = pd.read_csv(path)


# In[ ]:


data.head()


# In[ ]:


data.isnull().sum()


# In[ ]:


data.describe(include = "all")


# # MEDIAN: The median is a simple measure of central tendency. If there is an odd number of observations, the median is the middle value. If there is an even number of observations, the median is the average of the two middle values.

# # mean and median(50%) are huge gap in Confirmed, Deaths, and Recovered cases. As we can see that Confirmed cases mean is 6144.887002 and median is 245.Here, median is the Confirmed cases of the average data and mean is average Confirmed cases of the data.

# # Here mean and median are not equal. When mean is little more than the median it says that there is some data that has been pushed to the right. that is right skewed. 

# # Standard deviation : Standard deviation is a measure of how spared a typical observation is from the average.where the deviation is how far from the average you are...In this data we have average Confirmed cases is 252 but the standard deviation is 6237.239754. if the SD and average is same value so there is no issue, but here compare to confirmed average cases we have a huge SD cases.

# # Range= max - min
# # Confirmed cases = 362764.000000 - 0 = 362764 . We can say that the range of the confirmed cases is 362764.
# # The range of the deaths cases is 36914
# # the range of the Recovered cases is 379157

# In[ ]:


data.info()


# In[ ]:


data.hist(figsize=(20,30))


# # Confirmed and deaths cases have more left skewed than the right skewed.

# In[ ]:


sns.boxplot(x="Confirmed" , y="Deaths" , data=data)


# In[ ]:


sns.boxplot(x="Confirmed" , y="Recovered" , data=data)


# In[ ]:


sns.boxplot(x="Deaths" , y="Recovered" , data=data)


# In[ ]:


pd.crosstab(data['Confirmed'] , data['Deaths'])


# In[ ]:


pd.crosstab(data['Recovered'],data['Deaths'])


# In[ ]:


pd.crosstab(data['Confirmed'],data['Recovered'])


# In[ ]:


sns.pairplot(data)


# In[ ]:


data.mean()


# In[ ]:


data['Confirmed'].mean()


# In[ ]:


data['Confirmed'].std()


# In[ ]:


data['Deaths'].mean()


# In[ ]:


data['Recovered'].mean()


# In[ ]:


data['Recovered'].std()


# In[ ]:


data['Recovered'].var()


# In[ ]:


data['Confirmed'].median()


# In[ ]:


data['Confirmed'].var()


# In[ ]:


data['Deaths'].median()


# In[ ]:


data['Deaths'].var()


# In[ ]:


data['Deaths'].std()


# In[ ]:


data.cov()


# In[ ]:


data['Deaths'].std()


# In[ ]:


data['Recovered'].std()


# In[ ]:


data['Recovered'].median()


# In[ ]:


sns.distplot(data['Confirmed'],bins=1)


# In[ ]:


sns.distplot(data['Deaths'],bins=5)


# In[ ]:


sns.distplot(data['Recovered'],bins=1)


# In[ ]:


sns.countplot(x='Confirmed' , data=data)


# In[ ]:


sns.countplot(x='Deaths' , data=data)


# In[ ]:


sns.countplot(x='Recovered' , data=data)


# In[ ]:


sns.boxplot(x=data['Confirmed'])


# In[ ]:


sns.boxplot(x=data['Confirmed'] , y=data['Deaths'])


# In[ ]:


sns.boxplot(x=data['Deaths'])


# In[ ]:


sns.boxplot(x=data['Recovered'])


# In[ ]:


sns.boxplot(x=data['Deaths'] , y=data['Recovered'])


# In[ ]:


sns.boxplot(x=data['Confirmed'] , y=data['Recovered'])


# In[ ]:


corr = data.corr()
corr


# In[ ]:


sns.heatmap(corr , annot=True)


# In[ ]:


data.dtypes


# In[ ]:


data.drop(['ObservationDate','Province/State' , 'Country/Region' , 'Last Update'], axis='columns', inplace=True)

# Examine the shape of the DataFrame (again)
print(data.shape)


# In[ ]:


# split the dataset into train and test
# --------------------------------------
train, test = train_test_split(data, test_size = 0.3)
print(train.shape)
print(test.shape)


# In[ ]:


# split the train and test into X and Y variables
# ------------------------------------------------
train_x = train.iloc[:,0:1]; train_y = train.iloc[:,1]
test_x  = test.iloc[:,0:1];  test_y = test.iloc[:,1]
print(train_x)
print(test_x)


# In[ ]:


print(train_x.shape)
print(train_y.shape)
print(test_x.shape)
print(test_y.shape)


# In[ ]:


train_y.head()


# In[ ]:


train_x.head()


# In[ ]:


train.head()


# In[ ]:


train.tail()


# In[ ]:


train.dtypes


# In[ ]:


lm1 = sm.OLS(train_y, train_x).fit()
pdct1 = lm1.predict(test_x)
print(pdct1)


# In[ ]:


actual = list(test_y.head(5))
type(actual)
predicted = np.round(np.array(list(pdct1.head(5))),2)
print(predicted)
type(predicted)
data_results = pd.DataFrame({'actual':actual, 'predicted':predicted})
print(data_results)


# In[ ]:


from sklearn import metrics  
print('Mean Absolute Error:', metrics.mean_absolute_error(test_y, pdct1))  
print('Mean Squared Error:', metrics.mean_squared_error(test_y, pdct1))  
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(test_y, pdct1)))  

