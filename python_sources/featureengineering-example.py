#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import Imputer, LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[ ]:


df = pd.read_csv("../input/UntidyDataSet.csv",sep=",")


# In[ ]:


df.head()


# In[ ]:


df.gender.unique()


# In[ ]:


df.describe()


# In[ ]:


df.info()


# In[ ]:


Age = df.age.values.reshape((-1,1))
imputer = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0)
Age = imputer.fit_transform(Age)


# In[ ]:


Age


# In[ ]:


Country = df.country.values.reshape((-1,1))
labelEncoder = LabelEncoder()
CountryLE = labelEncoder.fit_transform(Country)
CountryLE


# In[ ]:


oneHotEncoder = OneHotEncoder(categorical_features = 'all')
CountryOHE = oneHotEncoder.fit_transform(Country).toarray()
CountryOHE


# In[ ]:


Gender = df.gender.values.reshape((-1,1))
CountryDF = pd.DataFrame(data = CountryOHE, index = range(22), columns = ['FR','TR','US'])
HeightWeightDF = pd.DataFrame(data = df.iloc[:,1:3].values, index = range(22), columns = ['Height', 'Weight'])
Age = pd.DataFrame(data=Age, index = range(22), columns = ['Age'])


# In[ ]:


CountryDF.head()


# In[ ]:


HeightWeightAgeDF = pd.concat([HeightWeightDF, Age], axis = 1)
CountryHeightWeightAgeDF = pd.concat([HeightWeightAgeDF, CountryDF], axis=1)


# In[ ]:


Gender = pd.DataFrame(data=Gender, columns=['Gender'], index = range(22))
Gender.head()


# In[ ]:


x_train, x_test, y_train, y_test = train_test_split(CountryHeightWeightAgeDF, Gender, test_size = 0.33, random_state=0)


# In[ ]:


standardScaler = StandardScaler()
X_train = standardScaler.fit_transform(x_train)
X_test = standardScaler.fit_transform(x_test)


# In[ ]:




