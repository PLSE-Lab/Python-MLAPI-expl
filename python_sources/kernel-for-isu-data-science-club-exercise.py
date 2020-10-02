#!/usr/bin/env python
# coding: utf-8

# This notebook has most of the lessons found on KUC lesson <s>Copypasted</s> implemented. Do your best to improve the final result of this notebook. If you have any questions about what this notebook is doing, [please look at the lesson that covers it](http://https://www.kaggle.com/learn/machine-learning) or ask me. Let's start by loading our data:

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.tree import DecisionTreeRegressor

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[ ]:


training_data = pd.read_csv('../input/train.csv')
test_data = pd.read_csv('../input/test.csv')
training_data.head()


# Now that we have our data, we need to get it in a format we can use.
# 
# First, we drop anything that isn't a number, such as text and NaN values.
# 
# Next, we separate the variable we want to predict from the rest of our variables, then drop that column.
# 
# Optionally, we can take a look at our statistics about the data we have left.

# In[ ]:


X = training_data.select_dtypes(exclude='object')
X = X.dropna()

y = X.SalePrice

X = X.drop(labels="SalePrice", axis=1)
X.describe()


# Now, we create our model.

# In[ ]:


ames_model = DecisionTreeRegressor(random_state=1)
ames_model.fit(X.iloc[200:], y.iloc[200:])


# In[ ]:


print("Making predictions for the following 5 houses:")
print(X.iloc[200:205])
print("The predictions are")
print(ames_model.predict(X.iloc[200:205]))
print("Actual values:")
print(y.iloc[200:205])


# Heck, that's some tasty accuracy!
# 
# Now, let's check the total error using Mean Absolute Error

# In[ ]:


from sklearn.metrics import mean_absolute_error #As someone who's most experienced with C and java, this line physically hurts me

predicted_home_prices = ames_model.predict(X.iloc[200:])
mean_absolute_error(y.iloc[200:], predicted_home_prices)


# Oooooooowweeeeeeeee! perfect! except, that's probably not accurate. This model has been overfitted, and won't be accurate for any new observations. let's test this out using the 200 we left out of our first trial:

# In[ ]:


predicted_home_prices = ames_model.predict(X[:200])
mean_absolute_error(y[:200], predicted_home_prices)


# That's closer to what I would expect. while a model can be fitted to be (basially) perfectly accurate on the data you train them on, it's worthless on new data. We can 'avoid' this using the train-test split.

# In[ ]:


from sklearn.model_selection import train_test_split

train_X, val_X, train_y, val_y = train_test_split(X, y, random_state = 0)
ames_model = DecisionTreeRegressor()
ames_model.fit(train_X, train_y)

# get predicted prices on validation data
val_predictions = ames_model.predict(val_X)
print(mean_absolute_error(val_y, val_predictions))


# This may seems just as bad, but now we know it's our model that's inaccurate, not that we've overfitted the model. So now let's make it better! Here's a copypasted way to compare changes to our model.

# In[ ]:


def get_mae(max_leaf_nodes, train_X, val_X, train_y, val_y):
    model = DecisionTreeRegressor(max_leaf_nodes=max_leaf_nodes, random_state=0)
    model.fit(train_X, train_y)
    preds_val = model.predict(val_X)
    mae = mean_absolute_error(val_y, preds_val)
    return(mae)

for max_leaf_nodes in [5, 50, 500, 5000]:
    my_mae = get_mae(max_leaf_nodes, train_X, val_X, train_y, val_y)
    print("Max leaf nodes: %d  \t\t Mean Absolute Error:  %d" %(max_leaf_nodes, my_mae))


# With this, we see that having 50 maximum leaf nodes is the most accurate model. the max leaf nodes is called a 'hyperparameter', it's something that we can change to get a better fitted model. The value that the hyperparameters should be at is different for every dataset.
# 
# We'll come back to hyperparameters later. let's improve our data set now! First let's estimate those NaN values we dropped with a tool called imputation.

# In[ ]:


from sklearn.preprocessing.imputation import Imputer

# make copy to avoid changing original data (when Imputing)
new_data = training_data.copy().select_dtypes(exclude='object')

# make new columns indicating what will be imputed
cols_with_missing = (col for col in new_data.columns 
                                 if new_data[col].isnull().any())
for col in cols_with_missing:
    new_data[col + '_was_missing'] = new_data[col].isnull()

# Imputation
my_imputer = Imputer()
new_data = pd.DataFrame(my_imputer.fit_transform(new_data))

new_y = new_data[40] # The imputer added two columns, I don't know why...
new_data = new_data.drop(labels=40,axis=1)
train_X, val_X, train_y, val_y = train_test_split(X, y, random_state = 0)

for max_leaf_nodes in [5, 50, 500, 5000]:
    my_mae = get_mae(max_leaf_nodes, train_X, val_X, train_y, val_y)
    print("Max leaf nodes: %d  \t\t Mean Absolute Error:  %d" %(max_leaf_nodes, my_mae))


# Hmm, that didn't do much.  Let's try including the text (catagorical) data.

# In[ ]:


y = training_data.SalePrice
X = training_data.drop(labels="SalePrice", axis=1)
one_hot_encoded_training_predictors = pd.get_dummies(X)
new_data = one_hot_encoded_training_predictors.copy()

cols_with_missing = (col for col in new_data.columns 
                                 if new_data[col].isnull().any())
for col in cols_with_missing:
    new_data[col + '_was_missing'] = new_data[col].isnull()
new_data = pd.DataFrame(my_imputer.fit_transform(new_data))

train_X, val_X, train_y, val_y = train_test_split(new_data, y, random_state = 0)
for max_leaf_nodes in [5, 50, 500, 5000]:
    my_mae = get_mae(max_leaf_nodes, train_X, val_X, train_y, val_y)
    print("Max leaf nodes: %d  \t\t Mean Absolute Error:  %d" %(max_leaf_nodes, my_mae))


# Hmm, still not good, let's try a different model.

# In[ ]:


from xgboost import XGBRegressor

my_model = XGBRegressor(n_estimators=100, learning_rate=0.02)
my_model.fit(train_X, train_y, early_stopping_rounds=2, 
             eval_set=[(val_X, val_y)], verbose=False)

predictions = my_model.predict(val_X)

print("Mean Absolute Error : " + str(mean_absolute_error(predictions, val_y)))


# Now there we go! cut the error down to 30,000! n_estimators, learning_rate, and early_stopping_rounds are hyperparameters. play around with these values and see if you can improve the model! I got it down below 17,000, so that's your target!
# 
# If you want to try something more advanced, look through these tutorials :https://www.kaggle.com/c/house-prices-advanced-regression-techniques#tutorials
