#!/usr/bin/env python
# coding: utf-8

# ## Linear Regression
# * **Regression analysis is a set of statistical processes for estimating the relationships among variables**
# * **In Linear regression, the outcome (dependent variable) is continuous**

# In[ ]:


#Import the packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression


# In[ ]:


#Import the data
data = pd.read_csv("../input/Salary_Data.csv")


# ## Exploratory Data Analysis

# In[ ]:


#First 5 lines of the data
data.head()


# In[ ]:


#Basic statistics of the data
data.describe()


# In[ ]:


#Basic info about the data
data.info()


# In[ ]:


#Correlation of the fields in the data
data.corr()


# In[ ]:


#Plot the relationship between the variables using pairplot
sns.pairplot(data)


# ## Data Pre-processing

# In[ ]:


#Separate Feature and Traget matrixs
x = data.iloc[:,:-1].values
y = data.iloc[:,1].values


# In[ ]:


#Split the train and test dataset
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.2)


# ## Machine Learning

# In[ ]:


#Define the Machine Learning Alorithm
ml = LinearRegression()
#ml = LogisticRegression()


# In[ ]:


#Train the Machine Learning Algorithm (Learning)
ml.fit(x_train,y_train)


# In[ ]:


#Test the Machine Learning Algorithm (Prediction)
y_pred = ml.predict(x_test)


# ## Comparison of the Prediction Results

# In[ ]:


plt.scatter(x_test,y_test,color= 'red')
plt.plot(x_test,y_pred,color='blue')
plt.xlabel("Years of Experience")
plt.ylabel("Salary")


# In[ ]:


ml.score(x_test,y_test)


# In[ ]:





# In[ ]:





# In[ ]:




