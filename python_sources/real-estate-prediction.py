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


import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


#Reading Data from file
filename = '/kaggle/input/real-estate-price-prediction/Real estate.csv'
df_realestate = pd.read_csv(filename) 


# In[ ]:


#Data summary
df_realestate.shape


# Rows - 414
# Columns - 8

# In[ ]:


df_realestate.info()


# No null values in dataset

# In[ ]:


df_realestate.head()


# In[ ]:


#Data Cleaning
#We can drop 'No' column as we have index column
#we can change column names
df_realestate.drop(['No'],inplace = True , axis=1)


# In[ ]:


#Rename
df_realestate.rename(columns = {
                                'X1 transaction date': 'Date',
                                'X2 house age' : 'House_age',
                                'X3 distance to the nearest MRT station' : 'MRT_distance',
                                'X4 number of convenience stores' : 'Conv_store_count',
                                'X5 latitude' : 'Latitude',
                                'X6 longitude' : 'Longitude',
                                'Y house price of unit area' : 'Price_per_unit'
}, inplace = True)


# In[ ]:


df_realestate.head()


# In[ ]:


#Date needs to be cleaned lets separate year
df_realestate['Year'] = df_realestate['Date'].astype(str).apply(lambda x: x[:4])
df_realestate['Year'].astype(int)


# In[ ]:


df_realestate.describe()


# In[ ]:


df_realestate.corrwith(df_realestate['Price_per_unit'])


# In[ ]:


df_realestate['Year'].nunique()


# In[ ]:


#we can see the distribution by price 
sns.catplot(x = 'Year' , y = 'Price_per_unit' , data =df_realestate)


# There is not much difference in years.

# In[ ]:


sns.boxplot(x = 'Year' , y = 'Price_per_unit' , data =df_realestate)


# In[ ]:


df_realestate.columns


# In[ ]:


#Modelling
X = df_realestate.drop(['Date','Year','Price_per_unit'],axis = 1)
Y = df_realestate['Price_per_unit']


# In[ ]:


X_train , X_test ,Y_train , Y_test = train_test_split(X,Y,test_size =0.2 , random_state = 42)


# In[ ]:


LM = LinearRegression()
LM.fit(X_train,Y_train)
LM.score(X_test ,Y_test)


# In[ ]:


Y_predict = LM.predict(X)


# In[ ]:


df_realestate['Predicted_price_per_unit'] =  LM.predict(X)


# In[ ]:


df_realestate.head()


# We used basic linear regression for training the model with using 20% as test data .Score is 0.67
# Next version will try polynomial features and get a better score.

# In[ ]:


#Using Polynomial features
pf = PolynomialFeatures(degree=2)
X_poly = pf.fit_transform(X)
X_poly_train , X_poly_test ,Y_poly_train , Y_poly_test = train_test_split(X_poly,Y,test_size =0.2 , random_state = 42)
LM_poly = LinearRegression()
LM_poly.fit(X_poly_train,Y_poly_train)
LM_poly.score(X_poly_test ,Y_poly_test)


# We can see we got a better score for polynomial regression with degree 2.
# Lets try for degree 2-10 and see the score values.

# In[ ]:


polynomial_degree1 = range(2,10)
score = []
for i in polynomial_degree1:
    X1 = df_realestate.drop(['Date','Year','Price_per_unit'],axis = 1)
    Y1 = df_realestate['Price_per_unit']
    pf1 = PolynomialFeatures(degree=i)
    X1_poly = pf1.fit_transform(X1)
    X1_poly_train , X1_poly_test ,Y1_poly_train , Y1_poly_test = train_test_split(X1_poly,Y1,test_size =0.2 , random_state = 42)
    LM1_poly = LinearRegression()
    LM1_poly.fit(X1_poly_train,Y1_poly_train)
    print(LM1_poly.score(X1_poly_test ,Y1_poly_test))


# In[ ]:


Since values are going negative degree 2 polynomial is a better fit so far.


# In[ ]:


df_realestate['Predicted_polynomial_price_per_unit'] =  LM_poly.predict(X_poly)
df_realestate


# Polynomial regression with degree two gives a better score.

# In[ ]:




