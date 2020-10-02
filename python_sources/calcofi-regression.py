#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from matplotlib import pyplot as plt

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# Any results you write to the current directory are saved as output.


# In[ ]:


dataBottle = pd.read_csv('../input/calcofi/bottle.csv',encoding='latin1')


# **CalCOFI**: Over 60 years of oceanographic data: *Is there a relationship between water salinity & water temperature? Can you predict the water temperature based on salinity?*

# * **T_degCWater** : temperature in degree Celsius
# * **Salnty** : Salinity in g of salt per kg of water (g/kg)

# In[ ]:


salinity = dataBottle[['Salnty']]
temperature = dataBottle[['T_degC']]


# * Now we're adding the module that let us divide the data into test and training processes
# * **Cross validation** allows us to compare different machine learning methods and get a sense of how well they will work in practice.

# Imputation fills in the missing value with some number. The imputed value won't be exactly right in most cases, but it usually gives more accurate models than dropping the column entirely.

# The default behavior fills in the mean value for imputation. Statisticians have researched more complex strategies, but those complex strategies typically give no benefit once you plug the results into sophisticated machine learning models.
# 
# One (of many) nice things about Imputation is that it can be included in a scikit-learn Pipeline. Pipelines simplify model building, model validation and model deployment.

# In[ ]:


from sklearn.impute import SimpleImputer
my_imputer = SimpleImputer()
salinityLast = my_imputer.fit_transform(salinity)
temperatureLast = my_imputer.fit_transform(temperature)
# Convert structured or record ndarray to DataFrame
tempFrame = pd.DataFrame.from_records(temperatureLast, index=None, exclude=None, columns=None, coerce_float=False, nrows=None)
salinityFrame = pd.DataFrame.from_records(salinityLast, index=None, exclude=None, columns=None, coerce_float=False, nrows=None)


# In[ ]:


plt.scatter(salinityLast, temperatureLast, edgecolors='r')
plt.xlabel('salinity')
plt.ylabel('temperature')
plt.show()


# In[ ]:


# salinityFrame.corr(tempFrame)


# In[ ]:


from sklearn.model_selection  import train_test_split


# In[ ]:


x_train, x_test, y_train, y_test = train_test_split(salinityLast,temperatureLast,test_size=0.33,random_state=0)


# In[ ]:


from sklearn.linear_model import LinearRegression
lr = LinearRegression()


# We're giving the training values to make the machine learn

# In[ ]:


lr.fit(x_train,y_train)


# In[ ]:


# By giving the test values of Salinity Data, predictions will be made for temperature.
prediction = lr.predict(x_test)


# In[ ]:


# We're sorting the data by their indexes to see the graphs more clear.
# x_train = x_train.sort_index()
# y_train = y_train.sort_index()
plt.plot(x_train,y_train)
plt.plot(x_test,prediction)
plt.rcParams['agg.path.chunksize'] = 10000
plt.show()


# In[ ]:




