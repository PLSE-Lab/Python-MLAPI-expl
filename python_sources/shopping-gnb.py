#!/usr/bin/env python
# coding: utf-8

# # Gaussian Naive Bayes classifier 

# In[ ]:


# import all libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


# load dataset
df = pd.read_csv("/kaggle/input/shopping/shopping.csv")


# In[ ]:


# print all data
df


# In[ ]:


# First 5 records
df.head(5)


# In[ ]:


# Last 5 records
df.tail(5)


# In[ ]:


# rows and columns
df.shape


# In[ ]:


#Rows
df.shape[0]


# In[ ]:


#Column
df.shape[1]


# In[ ]:


# Encode string values (Label Encoder)
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
day_encoded = pd.Series(le.fit_transform(df['Day']))
dis_encoded = pd.Series(le.fit_transform(df['Discount']))
del_encoded = pd.Series(le.fit_transform(df['FreeDelivery']))
pur_encoded = pd.Series(le.fit_transform(df['Purchase']))
df = pd.concat([day_encoded, dis_encoded, del_encoded, pur_encoded], axis=1)
df


# In[ ]:


# find X and Y
Y = df[3].values
X = df.drop(3, axis=1).values


# In[ ]:


# Split data into 70:30
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size = 0.3, random_state = 0)


# In[ ]:


#Prepare model
from sklearn.naive_bayes import GaussianNB


# In[ ]:


# Fit the data
model = GaussianNB().fit(x_train,y_train)


# In[ ]:


#predict the result
y_pred = model.predict(x_test)


# In[ ]:


# Classification report
from sklearn.metrics import classification_report
print(classification_report(y_test,y_pred))


# In[ ]:


# Confusion matrix
from sklearn.metrics import confusion_matrix
confusion_matrix(y_test,y_pred)


# In[ ]:


#Accuracy score
from sklearn.metrics import accuracy_score
print(accuracy_score(y_test,y_pred))


# In[ ]:




