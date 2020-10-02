#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# Importing the dataset

# In[ ]:


dataset = pd.read_csv("/kaggle/input/family-income-and-expenditure/Family Income and Expenditure.csv")


# Viewing the columns using .head() function

# In[ ]:


dataset.head()


# In[ ]:


null_data = dataset[dataset.isnull().any(axis=1)]
print(null_data.shape)


# Our dataset has total 7536 rows which have missing values, so we will have to deal with them respectively. We shall try replacing these missing values by mean and median, and then removing the rows, and try to find the effect on the model. Then we shall draw our conclusions for the dataset. First we shall replace these missing values using mean of the dataset.
# In the first method we shall be replacing the NAN values using mean of the column.

# In[ ]:


dataset = dataset.fillna(dataset.mean())


# We shall be dropping some columns like: Household Head Occupation, Household Head Class of Worker, Type of Roof, Type of Walls, Toilet Facilities, Main Source of Water Supply.
# 

# In[ ]:


dataset = dataset.drop(['Household Head Occupation', 'Household Head Class of Worker', 'Type of Roof', 'Type of Walls', 'Toilet Facilities', 'Main Source of Water Supply'], axis= 1)


# Now that we have filled the NA and missing values, we shall begin with creating dummy varibales for Region, Main Source of Income, Household Head Sex, Household Head Marital Status, Household Head Highest Grade Completed, Household Head Job or Business Indicator, Type of Household, Type of Building/House and Tenure Status. We shall be using pandas.get_dummies() for that.

# In[ ]:


dataset = pd.get_dummies(dataset, columns=['Region','Main Source of Income','Household Head Sex', 'Household Head Marital Status', 'Household Head Highest Grade Completed', 'Household Head Job or Business Indicator', 'Type of Household', 'Type of Building/House', 'Tenure Status'])


# In[ ]:


y = dataset['Total Household Income']
dataset = dataset.drop(['Total Household Income'], axis = 1)
x = dataset


# Creating test and train set, with random_state set at 0.

# In[ ]:


from sklearn.model_selection import train_test_split
xTrain, xTest, yTrain, yTest = train_test_split(x, y, random_state = 0)


# Now we shall be scaling the dataset, and then compare the models accuracy on grounds of scaled and not scaled. The first model we shall be using is Random Forest Regression regression using 10 trees only.

# In[ ]:


from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators = 100, random_state=0)
regressor.fit(xTrain, yTrain)
predictedWithoutScaling = regressor.predict(xTest) 


# Now we shall compare the predicted and true labels using R2Score and Mean Square Error.

# In[ ]:


from sklearn.metrics import r2_score, mean_squared_error
r2score = r2_score(yTest, predictedWithoutScaling)
mse = mean_squared_error(yTest, predictedWithoutScaling)
print('R2 Score using Random Forest without scaling using mean to fill NA values: ',r2score)
print('Mean Squared Error using Random Forest without scaling using mean to fill NA values: ',mse)


# As we can see that the model predicted not so good without Scaling, we shall now scale the data using MinMaxScaler.

# In[ ]:


from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(xTrain)
xTrain = scaler.transform(xTrain)
xTest = scaler.transform(xTest)
regressor.fit(xTrain, yTrain)
predictedWithScaling = regressor.predict(xTest) 
r2score2 = r2_score(yTest, predictedWithScaling)
mse2 = mean_squared_error(yTest, predictedWithScaling)
print('R2 Score using Random Forest without scaling using mean to fill NA values: ',r2score2)
print('Mean Squared Error using Random Forest without scaling using mean to fill NA values: ',mse2)


# We can infer that the Random Forest Regressor perfroms same, with or without scaling, So we shall look into another Regressor. Now we shall use Gradient boosted decision trees

# In[ ]:


from sklearn.ensemble import GradientBoostingRegressor
boostedRegressor = GradientBoostingRegressor( loss ='ls', learning_rate = 0.1, n_estimators= 200)
boostedRegressor.fit(xTrain, yTrain)
boostedPredicted = boostedRegressor.predict(xTest)
r2score3 = r2_score(yTest, boostedPredicted)
mse3 = mean_squared_error(yTest, boostedPredicted)
print('R2 Score using Gradient Boosted Forest without scaling using mean to fill NA values: ',r2score3)
print('Mean Squared Error using Gradient Boosted Forest without scaling using mean to fill NA values: ',mse3)

