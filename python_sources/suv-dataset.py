#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


# reading dataset
suv_data = pd.read_csv("../input/suv-data/suv_data.csv")


# In[ ]:


sns.countplot('Gender', data=suv_data)


# In[ ]:


#Plotting data about how many Male and Female are able to purchased the SUV
sns.countplot('Purchased', hue='Gender', data=suv_data)


# In[ ]:


# Plotting Age to check how old are the people in dataset
suv_data['Age'].plot.hist()


# In[ ]:


#checking head of the dataset
suv_data.head()


# In[ ]:


#Dropping User ID as we don't need for futher process
suv_data.drop('User ID', inplace=True, axis=1)


# In[ ]:


#checking head of the dataset after dropping of User ID
suv_data.head()


# In[ ]:


# using get_dummies to convert Gender's String value in binary
gender = pd.get_dummies(suv_data['Gender'], drop_first=True)
gender.head()


# In[ ]:


suv_data = pd.concat([suv_data,gender], axis=1)
suv_data.head()


# In[ ]:


#dropping Gender as we already converted it's data
suv_data.drop('Gender',axis=1,inplace=True)


# # Train Data

# In[ ]:


from sklearn.preprocessing import StandardScaler
sc = StandardScaler()


# In[ ]:


from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report


# In[ ]:


X = suv_data.drop('Purchased',axis=1)
y = suv_data['Purchased']


# In[ ]:


# Chossing data for training and Testing #test_size is use to choose for the testing and random_state is used to ensure the same result.
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.25,random_state=0)
X_train = sc.fit_transform(X_train)
X_test = sc.fit_transform(X_test)


# In[ ]:


# Fitting data to the Logistic Regression Model
log_model = LogisticRegression(solver='lbfgs')
log_model.fit(X_train, y_train)
# Predicting data to test the our trained data
predictions = log_model.predict(X_test)


# In[ ]:


#Checking Classification report
print(classification_report(y_test, predictions))


# In[ ]:


#checking confusion matrix
confusion_matrix(y_test, predictions)


# In[ ]:


# Checking prediction accuracy score in percentage
accuracy_score(y_test, predictions)*100

