#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# In[3]:


df = pd.read_csv("../input/Admission_Predict.csv")


# In[4]:


df.head()


# In[5]:


df.columns


# In[6]:


Features = df[["GRE Score","TOEFL Score","University Rating","SOP","LOR ","CGPA","Research"]]


# In[7]:


Features.head()


# In[8]:


Target = df[["Chance of Admit "]]


# In[9]:


df.isnull().sum()


# In[10]:


df.corr(method ='kendall') 


# # Linear Regression

# <h1>Model by taking GRE Score, TOEFL Score, University Rating, SOP, LOR, CGPA, Research to predict the Chance of Admit

# In[12]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(Features, Target, test_size=0.33, random_state=42)


# In[13]:


from sklearn.linear_model import LinearRegression
reg = LinearRegression().fit(X_train, y_train)
reg.score(X_train, y_train)
print("Training Score ",reg.score(X_train,y_train))


# In[14]:


print("Test Score ",reg.score(X_test,y_test))


# Accurary of the model for Training Data is 0.7978103892165577 <br>
# Accurary of the model for Training Data is 0.8044753910534395

# <h1>Model by taking GRE Score, TOEFL Score,University Rating, SOP and CGPA to predict the Chance of Admit</h1>

# In[15]:


Features_new = df[["GRE Score","TOEFL Score","University Rating","SOP","CGPA"]]
Features_new.head()


# In[16]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(Features_new, Target, test_size=0.33, random_state=42)


# In[17]:


from sklearn.linear_model import LinearRegression
reg = LinearRegression().fit(X_train, y_train)
reg.score(X_train, y_train)
print("Training Score ",reg.score(X_train,y_train))


# In[18]:


print("Test Score ",reg.score(X_test,y_test))


# Accurary of the model for Training Data is 0.7912759984131291 <br>
# Accurary of the model for Training Data is 0.7834612358585443

# In[ ]:




