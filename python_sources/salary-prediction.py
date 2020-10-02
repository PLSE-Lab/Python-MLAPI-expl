#!/usr/bin/env python
# coding: utf-8

# # Import Libraries

# In[ ]:


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


# # Import Dataset

# In[ ]:


dataset = pd.read_csv('../input/Baltimore_Salary.csv')
dataset.info()


# # Print first 10 Records

# In[ ]:


dataset.head(10)


# # Select rows and columns from Dataset

# In[ ]:


X = dataset.iloc[:,:-1].values
y = dataset.iloc[:,1].values


# # Import SKLearn Model

# In[ ]:


from sklearn.model_selection import train_test_split


# # Train Model and assign test data

# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.25, random_state=0)


# # Fit our model

# In[ ]:


from sklearn.linear_model import LinearRegression
reg = LinearRegression()
reg.fit(X_train, y_train)


# # Predictation as per our model

# In[ ]:


y_pred = reg.predict(X_test)


# # Print Y test data

# In[ ]:


print(y_test)


# # Print Y predicate data

# In[ ]:


print(y_pred)


# # Display in chart using train dataset

# In[ ]:


plt.scatter(X_train,y_train, color = 'red')
plt.plot(X_train, reg.predict(X_train), color = 'blue')
plt.title('Salary vs Experience (Training Set)')
plt.xlabel('Year of Experience')
plt.ylabel('Salary')
plt.show()


# # Display in chart using test dataset

# In[ ]:


plt.scatter(X_test,y_test, color = 'red')
plt.plot(X_train, reg.predict(X_train), color = 'blue')
plt.title('Salary vs Experience (Training Set)')
plt.xlabel('Year of Experience')
plt.ylabel('Salary')
plt.show()

