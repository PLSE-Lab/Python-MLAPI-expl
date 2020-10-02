#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# # Basic Linear Regression
# 

# First we import the csv. Notice that pandas stores data in DataFrames. This will be relevant later on.

# In[ ]:


file = pd.read_csv('/kaggle/input/weight-height/weight-height.csv')


# Check the file for the relevant columns that you may want to pick

# In[ ]:


file.head()


# Check the pairplot to spot any linear relationships.

# In[ ]:


sns.pairplot(file)


# Importing and applying Scikit Learn's Linear Regression

# In[ ]:


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression


# # Dealing with categoricals

# For our categoricals we would like to convert to an indicator function, this can be done by creating a dummy for any value appearing in a column.
# 
# For gender:
# 1 = Male
# 0 = Female
# 
# gnd will be our n x 1 vector (n being the number of entries/rows), as we will drop the Female column with drop_first=True
# 

# In[ ]:


gnd = pd.get_dummies(file['Gender'],drop_first=True)


# Checking to see for ourselves

# In[ ]:


gnd.head()


# After creating our single column vector, we drop gender because we want to replace it with the Male dummy variable.

# In[ ]:


file.drop('Gender',axis=1,inplace=True)


# We then reassemble it all with the concat function of pandas. We specify axis = 1 to add a column rather than a row (axis=0)
# 

# In[ ]:


file = pd.concat([file,gnd],axis=1)


# In[ ]:


file.shape


# # Choosing predictors and predicted

# Here we choose the independent variables and dependent variables.
# 
# In this case, we extract the Height and Male columns, our predictors , and assign them to X.
# 
# We then extract the desired predicted value, being Weight, and assign it to y.

# In[ ]:


X = file[['Height','Male']]
y = file['Weight']


# # train_test_split

# The purpose of train_test_split is to randomly generate a sample of a certain proportion (*test_size*) for you to first train your model with X_train and y_train, then test it's accuracy with X_test and y_test.

# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=432)


# We first create and then fit the model using the training data.

# # Fitting our Model

# In[ ]:


lmodel = LinearRegression()
lmodel.fit(X_train,y_train)


# We may observe X_test below, where it is obvious to see it has the same columns that we chose as our predictors X.

# In[ ]:


X_test.head()


# We assign our predictions for each sample observations to pred. pred is now a n x 1  single column vector containing the weight predictions.

# # Predicting obeservations and Quantifying accuracy

# In[ ]:


pred = lmodel.predict(X_test)


# Below we use a few statistics to measure our model accuracy. It is important to note that a low value for all of the following is associated with predictions with low error, but it may be hard to define how low is good enough.

# In[ ]:


from sklearn import metrics

print('MAE:', metrics.mean_absolute_error(y_test, pred))
print('MSE:', metrics.mean_squared_error(y_test, pred))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, pred)))


# Below we plot our predictions agains the actual observations in the scatterplot below

# In[ ]:


plt.scatter(X_test['Height'],y_test,color = 'blue',alpha='0.3')
plt.scatter(X_test['Height'],pred,color = 'red',alpha='0.1')


# # Bonus: Predicting a custom observation

# First let's review the format of our file

# In[ ]:


file.head()


# We first create a new DataFrame with the columns of our cleaned file.
# We can extract the columns from this file by writing file.columns

# In[ ]:


df2 = pd.DataFrame(columns=file.columns)    


# To verify, we check the head.
# 
# Indeed, our empty datafram lies below.

# In[ ]:


df2.head()


# Now let me remove the weight column.

# In[ ]:


df2 = df2.drop('Weight',axis=1)


# Assume we want to  predict a female weighing 180lbs measuring 5'11 roughly. We now populate the empty dataframe with its first entry!
# 
# Note that often you will just have a csv of your independent observations without the dependent obeservations included. This bonus is to predict small quantities of hypothetical observations

# In[ ]:


df2 =df2.append({'Height':70.07874,'Male': 0},ignore_index = True)


# Now finally to predict the weight based on the observations we supplied.
# 
# Essentially a linear regression line will be the equivalent of the average BMI + a constant based on gender.

# In[ ]:


print(lmodel.predict(df2)[0])


# It turns out the average BMI of our sample predicts our sample will have a weight of 173.7lbs. 

# ### This was just a little project I did to help those unfamiliar with Linear Regression and giving a step by step description of the thought process you should approach the problem with. While it is very basic it helped me familiarize myself more with Python dataframes and the general procedure of applying Statistical Learning.
