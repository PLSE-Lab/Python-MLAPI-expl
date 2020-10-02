#!/usr/bin/env python
# coding: utf-8

# **EDA, VISUALIZATION AND PREDICTION USING LINEAR REGRESSION**

# Here I have made an attempt to implement my learnings of linear regression model and exloratory data analysis on Top 5000 youtube channels dataset.
# 
# DETAILS OF DATASET:
# COLUMNS:
# Rank
# Grade
# Channel Name
# Video uploads
# Subscribers
# Video Views
# 
# I have predicted the video views on the basis of Grade, Video Uplaods, Subscribers.

# Importing libraraies and load data set.
# 

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")
import os
os.listdir("../input")


# In[ ]:



dataset=pd.read_csv("../input/data.csv")
dataset.head()


# Before we move further and play around with data set its important to remove null values from dataset.
# following statement is used to get the number of null values in dataset but in this case python is not showing any null values even after they are present because it treats nan as null value by default but here null value is "--".

# In[ ]:


dataset.isnull().sum()


# Therefore to_numeric function of pandas library is used, whose errors argument converts "--" into nan. After that all the rows with nan values are dropped.

# In[ ]:



dataset["Subscribers"]=pd.to_numeric(dataset["Subscribers"],errors="coerce")

dataset["Video Uploads"]=pd.to_numeric(dataset["Video Uploads"],errors="coerce")

dataset=dataset.dropna()


# Dividing the dataset into feature matrix and prediction vector.

# In[ ]:



X=dataset.iloc[:,[1,3,4]].values
y=dataset.iloc[:,-1].values


# In[ ]:



from sklearn.preprocessing import Imputer
imp=Imputer()
X[:,[1,2]]=imp.fit_transform(X[:,[1,2]])


# LabelEncoder is used to assign integer value to each string value of the given attribute. 

# In[ ]:



from sklearn.preprocessing import LabelEncoder
lab_x=LabelEncoder()

X[:,0]=lab_x.fit_transform(X[:,0].astype(str))


# OneHotEncoder converts the above column into sparse matrix.

# In[ ]:



from sklearn.preprocessing import OneHotEncoder
one=OneHotEncoder(categorical_features=[0])
X=one.fit_transform(X)
X=X.toarray()


# StandarScaler converts all the values of feature matrix into the same scale because machine learning algorithm otherwise will treat the attributes with larger values as of more priority for the prediction which is not true. Therefore it is very important to convert the feature matrix in the same scale. Here all the values are converted in the scale of -10 to 10.

# In[ ]:



from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
X=sc.fit_transform(X)


# Visualizing the dataset in order to have an insight the which attribute has a linear relationship with video views. 

# In[ ]:



pd.scatter_matrix(dataset,alpha=0.4)


# In[ ]:



from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test=train_test_split(X,y,test_size=0.2)


# Implementing linear regression model on the dataset.

# In[ ]:



from sklearn.linear_model import LinearRegression
lin_reg=LinearRegression()
lin_reg.fit(X_train,y_train)


# In[ ]:



y_predict=lin_reg.predict(X_test)

lin_reg.score(X_train,y_train)
lin_reg.score(X_test,y_predict)
lin_reg.score(X,y)


# In[ ]:


lin_reg.coef_

