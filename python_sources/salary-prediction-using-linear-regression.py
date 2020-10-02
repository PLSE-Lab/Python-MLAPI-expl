#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns

#Algorithms
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# In[ ]:


#Getting the Dataset
df=pd.read_csv('../input/Salary_Data.csv')


# **Data Pre Processing**

# In[ ]:


#checking data
df.head()


# In[ ]:


#checking for missing values
df.isnull().sum()                   #No missing value exist


# In[ ]:


df.info()


# In[ ]:


#Summarizing data
df.describe()      


# **Visualizing Data**

# In[ ]:


#Barplot showing gradual inracrease in salary with increase in experience
sns.barplot(x=df['YearsExperience'],y=df['Salary'])


# In[ ]:


#Scatter plot showing Linear relationship between two variables
plt.scatter(x=df['YearsExperience'],y=df['Salary'])


# In[ ]:


#Assigning data
X = df.iloc[:, :-1].values     #input variable
y = df.iloc[:, 1].values      #Target Variable


# **Model**

# In[ ]:


#splitting the data into train and test
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=1/3,random_state=0)


# In[ ]:


#Fitting Linear Regression to training data
regressor=LinearRegression()
regressor.fit(X_train,y_train)


# In[ ]:


y_pred=regressor.predict(X_test)


# **Visualizing Results**

# In[ ]:


# Visualising the Training set results
plt.scatter(X_train, y_train, color = 'red')
plt.plot(X_train, regressor.predict(X_train), color = 'blue')
plt.title('Salary vs Experience (Training set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()


# In[ ]:


# Visualising the Test set results
plt.scatter(X_test, y_test, color = 'red')
plt.plot(X_train, regressor.predict(X_train), color = 'blue')
plt.title('Salary vs Experience (Test set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()


# **Model Evaluation**

# In[ ]:


MAE= metrics.mean_absolute_error(y_test,y_pred) #Mean absolute error 
MSE= metrics.mean_squared_error(y_test,y_pred) #Mean squared error 
RMSE = np.sqrt(MSE) # Root Mean squared error


# In[ ]:


print("Mean Absoulte Eroor",MAE)


# **Interpretation**

# The above model shows that the Salary of an employess depends upon Years of experience. With increase of experience an employee would get more salary. 
