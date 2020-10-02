#!/usr/bin/env python
# coding: utf-8

# ![image.png](attachment:image.png)

# ## Multiple Linear Regression Model

# In statistics, **Multiple linear regression** (MLR) is a multivariate statistical technique for examining the linear correlations between two or more independent variables (IVs) and a single dependent variable (DV).

# **Case study:** To predict the profit of a company based on various spends like R&D, Administration, Marketing, and the State in which the company operates. Build a model to check if there are some linear dependencies between all these variables. Use the Multiple Regression model, plus, handle any missing data as well as a categorical variable (if any).

# ### Importing libraries

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# ### Importing dataset

# In[ ]:


df = pd.read_csv("../input/50_Startups.csv")
X = df.iloc[:,:-1]
y = df.iloc[:,4]


# In[ ]:


df.head()


# ## Data Preparation

# ### Missing value check

# In[ ]:


#check if there is any null values or not using below command. If yes, then use Imputer class to handle missing values.
df.isnull().sum()


# ### Feature Engineering

# In[ ]:


#check for any categorical variable (if any)
#Encoding categorical data in this case (independent variable)
from sklearn.preprocessing import LabelEncoder
labelencoder_X = LabelEncoder()
X.iloc[:,3] = labelencoder_X.fit_transform(X.iloc[:,3])


# ### Encoding character based variables into numeric based

# In[ ]:


#encoding the independent variables
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [3])], remainder='passthrough')
X = ct.fit_transform(X)
X = np.array(X)
X


# In[ ]:


#Alert! Avoid the dummy variable trap by removing any one dummy variable
X = X[:, 1:]


# ### Split dataset into training and testing dataset

# In[ ]:


from sklearn.model_selection import train_test_split  #(for python2)
#from sklearn.model_selection import train_test_split  (for python3)
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=0)


# In[ ]:


#no feature scaling is reuqired as Multiple Linear Refression algorithm take care by itself


# ### Model Fitting

# In[ ]:


#fit data into the model
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train,y_train)


# ### Model Prediction

# In[ ]:


y_pred = regressor.predict(X_test)


# In[ ]:


y_compare = pd.DataFrame(
    {'Original Profit': y_test,
     'Predicted Profit': y_pred,
     'Residual Error' : y_test-y_pred
    }).reset_index()
y_compare


# We can infer from the above results that the model did preity well for the testing dataset with error rate ranging between 250-13500 for 77500-191000 range of values. Good work!

# ### Follow me for more upcoming Machine Learning episode series. Upvote this kernel if you like it! Thank you! <br>
# 
# Previous episode 1 link: https://www.kaggle.com/prtk13061992/ml-series-episode-1-data-preprocessing <br>
# Previous episode 2 link: https://www.kaggle.com/prtk13061992/ml-series-episode-2-linear-regression <br>

# Citation: https://www.udemy.com/, https://en.wikiversity.org/
