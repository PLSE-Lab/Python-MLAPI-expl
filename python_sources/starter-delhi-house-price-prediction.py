#!/usr/bin/env python
# coding: utf-8

# ## Exploratory Analysis
# To begin this exploratory analysis, first import libraries and define functions for plotting the data using `matplotlib`. Depending on the data, not all plots will be made.

# In[ ]:


from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt # plotting
import numpy as np # linear algebra
import os # accessing directory structure
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)


# In[ ]:


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression 
from sklearn.tree import DecisionTreeRegressor 
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder 
from sklearn.metrics import mean_squared_error as mse


# In[ ]:


import seaborn as sns


# There is 1 csv file in the current version of the dataset:
# 

# In[ ]:


for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# ### Let's check 1st file: /kaggle/input/MagicBricks.csv

# In[ ]:


nRowsRead = 1259 # specify 'None' if want to read whole file
# MagicBricks.csv may have more rows in reality, but we are only loading/previewing the first 1000 rows
df1 = pd.read_csv('/kaggle/input/MagicBricks.csv', delimiter=',', nrows = nRowsRead)
df1.dataframeName = 'MagicBricks.csv'
nRow, nCol = df1.shape
print(f'There are {nRow} rows and {nCol} columns')


# Let's take a quick look at what the data looks like:

# In[ ]:


df1.head(5)


# In[ ]:


df1.describe()


# In[ ]:


sns.countplot(df1['BHK'])


# In[ ]:


df1['BHK'].value_counts()


# As, we can see,in the countplot and count_values there is only 1 property where 'BHK' is 7 and 10. And 6 rows have 6 'BHK'. These are some rare values that i don't want my model to train on. so the best option is to remove them from dataset.

# In[ ]:


df1[df1['BHK']==6]


# In[ ]:


df1.drop([721,345,163,164,261,352,353,585],inplace=True)  ##### these are indexes of rows, being removed from dataset


# In[ ]:


sns.countplot(df1['Bathroom'])


# similarly, we are removing Bathrooms, where Bathrooms are 6 and 7

# In[ ]:


df1[df1['Bathroom']==6]


# In[ ]:


df1[df1['Bathroom']==7]


# In[ ]:


df1.drop([225,495,527,659,676,681,1211,248,1029],inplace=True)


# In[ ]:


df1.Furnishing.value_counts().plot.bar()


# This is pretty obvious, right?

# In[ ]:


plt.figure(figsize=(14,7))
sns.boxplot(x=df1.Furnishing,y=df1.Price)


# In[ ]:


df1.isnull().sum()


# 33 Null values in Parking, which means, No Parking available in the apartment. so we can impute NAN by 0
# 

# In[ ]:


df1.Parking.fillna(0,inplace=True)


# In[ ]:


sns.countplot(df1.Parking)


# wow, we've got 39 and 114 parking for some apartments, which is definitely an error. Now the problem is that we can not just remove all the rows we have , 'cause we already facing some serious shortage of data, removing them won't do any good. 
# 

# In[ ]:


df1[df1.Parking==39]


# These are DDA flats of Narela and price of each apartment is very low, and 39 must be the sum of all parkings available so we can safely assume every flat has one parking alloted. we are repalcing 39 as 1. similary 114 is the sum of all the parkings available,so we will replace 114 by 1.

# In[ ]:


df1['Parking'].replace([39,114],1,inplace=True)
df1['Parking'].replace([5,9,10],4,inplace=True)


# In[ ]:


sns.countplot(df1.Status)


# In[ ]:


sns.countplot(df1['Type'])


# In[ ]:


sns.boxplot(x=df1.Transaction,y=df1.Price)


# In[ ]:


plt.figure(figsize=(14,7))
sns.scatterplot(x=df1.Area,y=df1.Price)


# As, I anticipated there is a positive correlation between 'Area' and 'Price. 
# 

# And there is a definitely a relation between 'Area','Price' and 'Per_Sqft', which is
# 'Per_Sqft' = 'Price'/'Area'. 
# 
# **Multicolinearity : if two or more features(except target variable) are highly correlated on each other,this phenomena is known as multicolinearity. we don't want that to happen as it just create redundancy in dataset.**
# 
# so, we can eliminate one of the variable, as to make model less complex and that variable would be 'Per_Sqft' because it contains 240 NAN values.

# In[ ]:


df1.drop('Per_Sqft',axis=1,inplace=True)


# In[ ]:


df1.isnull().sum()


# Too many null values, we've got to impute them by mean, meadian or mode

# In[ ]:


df1.Bathroom.fillna(df1.Bathroom.median(),inplace=True)
df1.Type.fillna('Apartment',inplace=True)
df1.Furnishing.fillna('Semi-Furnished',inplace=True)


# **So far, all good. 
# Localities are essential for any property, infact after 'Area', 'locality' is the characteristic which matter most in property sale/purchase. for example:- if you are buying a house of area 200 sqft in locality like 'Seelampur' it may cost you about 2 Lac (say) 1000 Rs per square feet but on the contrary in 'Rohini' it may cost you 20 Lac, 10000 Rs per square feet.   **

# In[ ]:


df1.Locality.unique()


# This is too messy. For the time being,I would remove this from dataset. I can't do anything with this feature right now. Always open to suggestions/feedback.

# In[ ]:


df1.drop('Locality',axis=1,inplace=True)


# Before moving on, we need to convert all categorical(where data type is object) features into numerical features,because machine learning algorithm doesn't work on strings or other objects.
# Here i am using one hot encoding , since we don't have too many category in a feature for example :- in 'Type' we have 'Apartment' and 'Builder-Floor'. so one-hot-encoding would be right choice.

# In[ ]:


df1 = pd.get_dummies(df1)


# In[ ]:


Y = df1.Price
X = df1.drop('Price',axis=1)


# Dividing data in 80% training and 20% testing parts.usally i keep it 70 & 30.

# In[ ]:


x_train,x_test,y_train,y_test = train_test_split(X,Y,test_size = 0.2,random_state = 42)


# In[ ]:


x_train


# Using Linear Regression, my first choice for this project, Intiutive right?

# In[ ]:


lr = LinearRegression()
lr.fit(x_train,y_train)   ########### traing model
pred = lr.predict(x_test) ########### Getting predictions


# In[ ]:


pred


# In[ ]:


from math import sqrt


# In[ ]:


print(sqrt(mse(y_test,pred)))  ######## Root Mean Square Error 


# This is scary(8 digit no.) as an error

# Let's try Decision tree regressor

# In[ ]:


lr = DecisionTreeRegressor()
lr.fit(x_train,y_train)
pred = lr.predict(x_test)
print(sqrt(mse(y_test,pred)))


# 8 digit no. but less than the previous, we are making progress. i guess 

# In[ ]:


sns.lineplot(x=df1.Area,y=df1.Price)


# Here, I want to remove some outliers, we'll see how does it affect RMSE

# In[ ]:


plt.figure(figsize=(14,7))
sns.scatterplot(x=df1.Area,y=df1.Price)


# I'd like to get rid of all the 'Areas' greater than 5000. 

# In[ ]:


p = np.array(df1[df1.Area>5000].index)


# In[ ]:


df1.drop(p,inplace=True)  ##### these are indexes of rows, being removed from dataset


# **Regression Plot**

# In[ ]:


plt.figure(figsize=(14,7))
sns.regplot(x="Area", y="Price", data=df1)


# In[ ]:


Y = df1.Price
X = df1.drop('Price',axis=1)
x_train,x_test,y_train,y_test = train_test_split(X,Y,test_size = 0.2,random_state = 42)
lr = DecisionTreeRegressor()
lr.fit(x_train,y_train)
pred = lr.predict(x_test)
print(sqrt(mse(y_test,pred)))


# In[ ]:


Y = df1.Price
X = df1.drop('Price',axis=1)
x_train,x_test,y_train,y_test = train_test_split(X,Y,test_size = 0.2,random_state = 42)
lr = LinearRegression()
lr.fit(x_train,y_train)
pred = lr.predict(x_test)
print(sqrt(mse(y_test,pred)))


# ## Conclusion
# We are still getting 8 digit no. but we have minimized it a lot.
# I have few more things to try, but i don't want to make it too large for a starter code.
# I hope you like it and please share your feedbacks/suggestion to improve it. Thanks 
