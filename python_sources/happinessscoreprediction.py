#!/usr/bin/env python
# coding: utf-8

# **Happiness score prediction**
# 
# This notebook is about exploring and comparing Decision Tree and Random Forest models in a task: predict the happiness score for some countries.
# 
# Firstly, we should explore one of datasets to understand the use of columns and its' values types.

# In[309]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

data = pd.read_csv('../input/2015.csv')
print(data.columns)
data[:10]


# This dataset has no any missing values, so we can select predictors just having some knowledges about key factors. The folowing predictors can do a big influence on the target variable - Happiness Score.

# In[310]:


y = data['Happiness Score'] # target variable
happiness_score_predictors = ['Country','Economy (GDP per Capita)', 'Family', 'Health (Life Expectancy)', 'Freedom', 'Trust (Government Corruption)',
       'Generosity', 'Dystopia Residual']
X = data[happiness_score_predictors] # predictors


# In the following steps we evently split the entire data in two groups - train and validation subsets. A validation subset called that because here we just compare the predictive models.
# 
# The first model is Decision Tree. We need to create an object of function *DecisionTreeRegressor()*, fit it with train data, make prediction with validation data and calculate the difference between expected and actual values.

# In[311]:


from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split

model_decision_tree = DecisionTreeRegressor()

train_X, val_X, train_y, val_y = train_test_split(X, y, random_state = 0)
model_decision_tree.fit(train_X.drop(['Country'], axis=1), train_y)
prediction_tree = model_decision_tree.predict(val_X.drop(['Country'], axis=1))
#val_y - prediction # to see the actual difference between expected and calculated values
error_tree = mean_absolute_error(val_y, prediction_tree)
print(error_tree)


# The second model is Random Forest. We proceed it with the same steps as for previous model.

# In[312]:


from sklearn.ensemble import RandomForestRegressor

model_random_forest = RandomForestRegressor()
model_random_forest.fit(train_X.drop(['Country'], axis=1), train_y)
prediction_forest = model_random_forest.predict(val_X.drop(['Country'], axis=1))
error_forest = mean_absolute_error(val_y, prediction_forest)
print(error_forest)

Now let's visualize this comparison using standard tools of Matplotlib. By the way, 'Country' variable deliberately was left in X subset to collerate Happiness Score with country name.
# In[319]:


import matplotlib.pyplot as plt

dt = data[:40]['Country']
sorted_val_y = val_y.sort_values(ascending=False)

plt.plot(np.sort(prediction_tree), marker='o', label='Decision Tree model')
plt.plot(np.sort(prediction_forest), marker='o', label='Random Forest model')
plt.plot(np.sort(val_y.values), marker='o', label='Actual')
plt.legend()
plt.xticks(range(len(val_y.values)), val_X['Country'], rotation = 60)

fig_size = plt.rcParams["figure.figsize"]
fig_size[0] = 30
fig_size[1] = 5
plt.rcParams["figure.figsize"] = fig_size
plt.show()


# In this kernel there were used two of the most simple predictive models. Even thought this dataset is not a very good choice for predictive analysis, created models can be used with some new subsets of countries data. For example, when GDP or Generosity lavel changes the target value can also changes. Further we will use two other datasets of 2016 and 2017 to create something else interesting. But as for now, we can change the output of the plot by varying the *random_state* parameter of *train_test_split* function.
