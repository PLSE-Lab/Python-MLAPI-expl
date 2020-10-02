#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#importing the basic library

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import pandas_profiling


# In[ ]:


import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# In[ ]:


# reading the CSV file to python

hotel = pd.read_csv("../input/hotel-booking-demand/hotel_bookings.csv")


# In[ ]:


hotel


# In[ ]:





# In[ ]:


# checking the summary statistics of data
hotel.describe()


# In[ ]:


#check the top 5 row of dataset
hotel.head()


# In[ ]:


#checking the bottom 5 row of dataset
hotel.tail()


# In[ ]:


#checking the data information
hotel.info()


# In[ ]:


#calculating the missing value in each column
hotel.isnull().sum()


# In[ ]:


#commpany column in dataset has maximum no of null values
# so we remove the column  
hotel= hotel.drop(['company'],axis=1)


# In[ ]:


#removing all the row having missing value
hotel= hotel.dropna(axis=0)


# In[ ]:


hotel.info()


# In[ ]:


# again checking the missing value 
hotel.isnull().sum()


# In[ ]:


#checking the unique value of hotel
hotel['hotel'].unique()


# In[ ]:


#checking the data type for all feature in dataset  
hotel.info()


# In[ ]:


#converting the required object type feature to categorical
categorical_features = ['hotel','is_canceled','arrival_date_week_number','meal','country','market_segment',
                        'distribution_channel','is_repeated_guest','reserved_room_type','assigned_room_type',
                        'deposit_type','agent','customer_type','reservation_status','arrival_date_month']


# In[ ]:


hotel[categorical_features]=hotel[categorical_features].astype('category')


# In[ ]:


# checking the converted data type
hotel.info()


# In[ ]:



hotel['meal'].unique()


# In[ ]:


# seperating the dataset into features and target variables
y=hotel['is_canceled']


# In[ ]:


y


# In[ ]:


X = hotel.drop(['is_canceled','reservation_status_date'],axis=1)


# In[ ]:


X


# In[ ]:


#converting the categorical data into dummy variable  
X_dum=pd.get_dummies(X,prefix_sep='-',drop_first=True)


# In[ ]:


X_dum


# In[ ]:


#Splitting the data into train and test
from sklearn.model_selection import train_test_split


# In[ ]:


X_train,X_test,y_train,y_test= train_test_split(X_dum,y, test_size=.25,random_state=40)


# In[ ]:


X_train


# In[ ]:


# preparing a logistic regression model
from sklearn.linear_model import LogisticRegression


# In[ ]:


logistic=LogisticRegression()


# In[ ]:


logistic.fit(X_train,y_train)


# In[ ]:


#predicting the test data
y_pred= logistic.predict(X_test)


# In[ ]:


# calculating the  accuracy, precision,recall and f1-score for logistic regression
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report


# In[ ]:


accuracy_score(y_test,y_pred)


# In[ ]:


classification_report(y_test,y_pred)


# In[ ]:


#calculating the ROC and AUC  for the logistics regression
from sklearn.metrics import roc_curve,roc_auc_score


# In[ ]:


roc_curve(y_test,y_pred)


# In[ ]:


roc_auc_score(y_test,y_pred)


# In[ ]:


#now  we will make a model of random forest and  gradient boosting 
from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier


# In[ ]:


rand=RandomForestClassifier(n_jobs=10, random_state=40)


# In[ ]:


gb=GradientBoostingClassifier(random_state=50)


# In[ ]:


rand.fit(X_train,y_train)


# In[ ]:


gb.fit(X_train,y_train)


# In[ ]:


# predicting the test sample for randomforest and gradient boosting 
rand_pred=rand.predict(X_test)


# In[ ]:


gb_pred=gb.predict(X_test)


# In[ ]:


# checking accuracy, precision,recall and f1-score for data
accuracy_score(y_test,rand_pred)


# In[ ]:


accuracy_score(y_test,gb_pred)


# In[ ]:


classification_report(y_test,rand_pred)


# In[ ]:


classification_report(y_test,gb_pred)


# In[ ]:


roc_auc_score(y_test,rand_pred)


# In[ ]:


roc_auc_score(y_test,gb_pred)


# In[ ]:


#creating  confusion matrix for logistic reression,random forest and gradient boosting
from sklearn.metrics import confusion_matrix


# In[ ]:


confusion_matrix(y_test,y_pred)


# In[ ]:


confusion_matrix(y_test,rand_pred)


# In[ ]:


confusion_matrix(y_test,gb_pred)


# from all three algorithm we will found that random forest and gradient boosting gives maximum accuracy

# In[ ]:




