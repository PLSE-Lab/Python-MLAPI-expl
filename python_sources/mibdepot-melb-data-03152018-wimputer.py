# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.preprocessing import Imputer

import matplotlib.pyplot as plt 
plt.rc("font", size=14)
import seaborn as sns
sns.set(style="white")
sns.set(style="whitegrid", color_codes=True)

from scipy import stats

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))


melbourne_file_path = '../input/melb_data.csv'
melbourne_data_wNAN = pd.read_csv(melbourne_file_path) 
print(melbourne_data_wNAN.columns)
print(melbourne_data_wNAN.isnull().sum())

# imputer function using sklearn
my_imputer = Imputer()
#data_with_imputed_values = my_imputer.fit_transform(melbourne_data_wNAN)
#print(data_with_imputed_values.isnull().sum())
# end imputer function using sklearn

#melbourne_data = melbourne_data_wNAN.dropna()
melbourne_data = melbourne_data_wNAN
melbourne_data.describe()

# heat map
#sns.heatmap(melbourne_data.corr(), square=True, cmap='RdYlGn')
#plt.show()

y = melbourne_data.Price
#y = my_imputer.fit_transform(y)
y.describe()
melbourne_predictors = ['Rooms', 'Bathroom', 'Landsize', 'BuildingArea', 'YearBuilt', 'Lattitude', 'Longtitude']

X = melbourne_data[melbourne_predictors]
X = my_imputer.fit_transform(X)
#X.describe()

stats.describe(X)
#-------------------------------------------------------------------------------------------------------------------
# DecisionTreeRegressor
#-------------------------------------------------------------------------------------------------------------------
from sklearn.tree import DecisionTreeRegressor

# Define model
melbourne_model = DecisionTreeRegressor()

# Fit model
melbourne_model.fit(X, y)
#-------------------------------------------------------------------------------------------------------------------

# Make predictions
print("Making predictions for the following 5 houses:")
#print(X.head())
print (X[[0, 1, 2, 3, 4, 5], :])
print("The predictions are")
#print(melbourne_model.predict(X.head()))
print(melbourne_model.predict(X[[0, 1, 2, 3, 4, 5], :]))
# Any results you write to the current directory are saved as output.

#Calculate MAE - Mean Absolute Error
from sklearn.metrics import mean_absolute_error

predicted_home_prices = melbourne_model.predict(X)
mean_absolute_error(y, predicted_home_prices)

# Working with Test, Train sample data
from sklearn.model_selection import train_test_split

# split data into training and validation data, for both predictors and target
# The split is based on a random number generator. Supplying a numeric value to
# the random_state argument guarantees we get the same split every time we
# run this script.
train_X, val_X, train_y, val_y = train_test_split(X, y,random_state = 0)
# Define model
melbourne_model = DecisionTreeRegressor()
# Fit model
melbourne_model.fit(train_X, train_y)

# get predicted prices on validation data
val_predictions = melbourne_model.predict(val_X)
print(mean_absolute_error(val_y, val_predictions))

# controlling tree depth
from sklearn.metrics import mean_absolute_error
from sklearn.tree import DecisionTreeRegressor

# Function to return mae if num nodes are passed
def get_mae(max_leaf_nodes, predictors_train, predictors_val, targ_train, targ_val):
    model = DecisionTreeRegressor(max_leaf_nodes=max_leaf_nodes, random_state=0)
    model.fit(predictors_train, targ_train)
    preds_val = model.predict(predictors_val)
    mae = mean_absolute_error(targ_val, preds_val)
    return(mae)
    
# compare MAE with differing values of max_leaf_nodes
for max_leaf_nodes in [5, 50, 500, 5000]:
    my_mae = get_mae(max_leaf_nodes, train_X, val_X, train_y, val_y)
    print("Max leaf nodes: %d  \t\t Mean Absolute Error:  %d" %(max_leaf_nodes, my_mae))
    
# RandomForestRegressor
#-------------------------------------------------------------------------------------------------------------------
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

forest_model = RandomForestRegressor()
forest_model.fit(train_X, train_y)
melb_preds = forest_model.predict(val_X)
print("RandomForestRegressor prediction: " %mean_absolute_error(val_y, melb_preds))

#-------------------------------------------------------------------------------------------------------------------