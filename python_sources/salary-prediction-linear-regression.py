#!/usr/bin/env python
# coding: utf-8

# ## Import Libraries 

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split


# ## Load Data

# In[ ]:


Salary = pd.read_csv("../input/salary/Salary.csv")
print(f"Data Shape: {Salary.shape}")


# ## Selecting Desired Features

# In[ ]:


# Checking dataset columns
print(Salary.columns)

#Selecting Corresponding Features
X = Salary['YearsExperience'].values
y = Salary['Salary'].values

X = X.reshape(-1,1)
y = y.reshape(-1,1)


# ## Divide data into train and test test
# The most common train and test split size is 80% data for train set and 
# 20 % data for test set

# In[ ]:


# X_feature - Years of Experience
# y_output - Salary in $
x_train, x_test, y_train, y_test = train_test_split(X,y,train_size=0.8,test_size=0.2,random_state=100)
print(f"X_train shape {x_train.shape}")
print(f"y_train shape {y_train.shape}")
print(f"X_test shap {x_test.shape}")
print(f"y_test shape {y_test.shape}")


# ## Exploring data by ploting

# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')
plt.scatter(x_train,y_train,color='red')
plt.xlabel('Years of experience')
plt.ylabel('Salary in $')
plt.title('Training data')
plt.show()


# ## Building Linear Regression Model using Sklearn

# In[ ]:


lm = LinearRegression()
lm.fit(x_train,y_train)
y_predict = lm.predict(x_test)
print(f"Train accuracy {round(lm.score(x_train,y_train)*100,2)} %")
print(f"Test accuracy {round(lm.score(x_test,y_test)*100,2)} %")


# ## Plotting the Linear Model

# In[ ]:


plt.scatter(x_train,y_train,color='red')
plt.plot(x_test,y_predict)
plt.xlabel("Years of Experience")
plt.ylabel("Salary in $")
plt.title("Trained model plot")
plt.plot


# ## Predicting with custom data

# In[ ]:


yoe = np.array([15,1.5,7.3,9.65])
yoe = yoe.reshape(-1,1)
yoe_salary = lm.predict(yoe)
for salary in yoe_salary:
    print(f"$ {salary}")

