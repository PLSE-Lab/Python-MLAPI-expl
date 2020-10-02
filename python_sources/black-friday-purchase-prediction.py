#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
pd.set_option('float_format', '{:.2f}'.format)
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
import sklearn.linear_model as model
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.svm import SVR
from sklearn.metrics import accuracy_score

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# **This notebook is a work in Progress !**

# In[ ]:


dataset = pd.read_csv('../input/BlackFriday.csv')
dataset.head()


# In[ ]:


print("The Number of Rows:",dataset.shape[0])
print("The Number of Columns:",dataset.shape[1])


# Since the age data we have is a range, we try to give it a discrete value which would help us in prediction.

# In[ ]:


dataset.Age.replace(('0-17','18-25','26-35','36-45','46-50','51-55','55+'),(0,1,2,3,4,5,6), inplace =True)


# In[ ]:


dataset.head()


# In[ ]:


dataset.describe(include='all')


# In[ ]:


age0 = dataset[dataset['Age'] ==0]['Purchase'].values.sum()
age1 = dataset[dataset['Age'] ==1]['Purchase'].values.sum()
age2 = dataset[dataset['Age'] ==2]['Purchase'].values.sum()
age3 = dataset[dataset['Age'] ==3]['Purchase'].values.sum()
age4 = dataset[dataset['Age'] ==4]['Purchase'].values.sum()
age5 = dataset[dataset['Age'] ==5]['Purchase'].values.sum()
age6 = dataset[dataset['Age'] ==6]['Purchase'].values.sum()


X = pd.DataFrame([age0,age1,age2,age3,age4,age5,age6])
X.index=['0-17','18-25', '26-35','36-45','46-50','50-54','55+']
X.plot(kind = 'bar',title="Total Sales based on the Age group")


# In[ ]:


dataset.Gender.replace(('M','F'),(0,1), inplace =True)
dataset.head()


# In[ ]:


Male = dataset[dataset['Gender'] ==0]['Purchase'].values.sum()
Female = dataset[dataset['Gender'] ==1]['Purchase'].values.sum()

print("Total Sales by Male: ", Male)
print("The Ratio of Male to Total: ", Male/(Male+Female))
print("Total Sales by Female:", Female)
print("The Ratio of Female to Total: ", Female/(Male+Female))


# In[ ]:


GenderPlot = pd.DataFrame([Male,Female])
GenderPlot.index=['Male','Female']
GenderPlot.plot(kind = 'bar',title="Total Sales based on the Gender")


# In[ ]:


corr = dataset.corr(method='pearson')
print("Correlation of the Dataset:",corr)


# In[ ]:


f,ax = plt.subplots(figsize=(18, 18))
print("Plotting correlation:")
sns.heatmap(corr,annot= True, linewidths=.5)


# In[ ]:


print("Data Based on Occupation:")
occupationStat = dataset['Occupation'].value_counts(dropna = False)
occupationStat.plot(kind='pie', figsize=(10,10))


# In[ ]:


print("Total Nan Values:")
totalnan = dataset.isnull().sum(axis = 0)
print(totalnan)


# In[ ]:


dataframe = dataset.drop(['User_ID'], axis=1)


# In[ ]:


labelEncoder_CityCat = LabelEncoder()
dataframe.City_Category = labelEncoder_CityCat.fit_transform(dataframe.City_Category)


# Since we can't have categorical data for Regression but if we notice our Stay_In_Current_City_Years has 4+ Years for people staying more than 4 years. since 4+ is the only range we have, here we will replace that with four and consider 4 as everything thats more than 3.

# In[ ]:


dataframe.Stay_In_Current_City_Years.replace(('4+'),(4), inplace =True)


# We have some missing data in our columns, we can handle it in multiple ways. For now, we are just going to use only the datas that doesn't contain N/A. In future versions of the notebook we will update this with other methods of handling missing data.

# In[ ]:


dataframe.Product_Category_1 = dataframe.Product_Category_1.fillna(0)
dataframe.Product_Category_2 = dataframe.Product_Category_2.fillna(0)
dataframe.Product_Category_3 = dataframe.Product_Category_3.fillna(0)


# In[ ]:


dataframe = dataframe[0:30000]
dataframe.head()
print("No. of Rows:", dataframe.shape[0])


# In[ ]:


# features = dataframe.drop(['Product_ID'],axis=1)
# X_train = features[:20000]
# X_test = features[20000:]
# y_train = features.Purchase[:20000]
# y_test = features.Purchase[20000:]

# # diabetes_X_train = diabetes_X[:-20]
# # diabetes_X_test = diabetes_X[-20:]

# # # Split the targets into training/testing sets
# # diabetes_y_train = diabetes.target[:-20]
# # diabetes_y_test = diabetes.target[-20:]

# # testtarget = features.Purchase[20000:]


# In[ ]:


testfeature = testfeature.drop(['Purchase'],axis=1)
# testfeature = testfeature.drop(['Product_ID'],axis=1)


# In[ ]:


# trainfeatures.drop(['Product_ID'],axis=1)


# In[ ]:


trainfeatures.head(10)


# In[ ]:


features_train, features_test, target_train, target_test = train_test_split(features, target, test_size=0.20, random_state=42)


# In[ ]:


print(trainfeatures.shape)
print(traintarget.shape)


# In[ ]:


regr = model.LinearRegression()
regr.fit(features_train,target_train)
prediction = regr.predict(features_test)

print('Coefficients: \n', regr.coef_)
print("Mean squared error: %.2f"
      % mean_squared_error(target_test, prediction))
print('Variance score: %.2f' % r2_score(target_test, prediction))


# In[ ]:


svr = SVR()
svr.fit(features_train,target_train)
prediction_svr = svr.predict(features_test)
score = r2_score(target_test, prediction_svr)
mae = mean_squared_error(prediction_svr, target_test)
print("Score:", score)
print("Mean Absolute Error:", mae)


# 
