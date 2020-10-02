#!/usr/bin/env python
# coding: utf-8

# ## Logistic Regression
# * **Regression analysis is a set of statistical processes for estimating the relationships among variables**
# * **In logistic regression, the outcome (dependent variable) is binary**

# In[33]:


#Import the packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix


# In[2]:


#Import the data
data = pd.read_csv("../input/insurance_data.csv")


# ## Exploratory Data Analysis

# In[3]:


#First 5 lines of the data
data.head()


# In[4]:


#Basic statistics of the data
data.describe()


# In[5]:


#Basic info about the data
data.info()


# In[6]:


#Correlation of the fields in the data
data.corr()


# In[7]:


#Plot the relationship between the variables using pairplot
sns.pairplot(data)


# ## Data Pre-processing

# In[8]:


#Separate Feature and Traget matrixs
x = data.iloc[:,:-1].values
y = data.iloc[:,1].values


# In[9]:


#Split the train and test dataset
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.2)


# ## Machine Learning

# In[10]:


#Define the Machine Learning Alorithm
ml = LogisticRegression()


# In[11]:


#Train the Machine Learning Algorithm (Learning)
ml.fit(x_train,y_train)


# In[12]:


#Test the Machine Learning Algorithm (Prediction)
y_pred = ml.predict(x_test)


# ## Comparison of the Prediction Results

# In[32]:


plt.scatter(x_test,y_test,color= 'red', marker='+')
plt.scatter(x_test,y_pred,color='blue', marker='.')
plt.xlabel("Age of person")
plt.ylabel("Bought Insurance 1=Bought 0=Did not Buy")


# In[28]:


ml.score(x_test,y_test)


# Find the results [TP   FP
# 
#                   FN  TN ]

# In[30]:


confusion_matrix(y_test,y_pred)


# In[ ]:





# In[ ]:




