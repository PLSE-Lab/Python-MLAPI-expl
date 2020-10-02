# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
#imports pandas
import pandas as pd

#sets variable equal to data adn makes it readable
main_file_path = '../input/train.csv' # this is the path to the Iowa data that you will use
data = pd.read_csv(main_file_path)

main = '../input/test.csv' 
tests = pd.read_csv(main)
#Prints the statistical description of the data and the collumn headers of teh data
print(data.describe())
print(data.columns)

#sets variable sale_price equal to the sales price and prints out the first few rows of sales prices
sale_price=data.SalePrice
print(sale_price.head())

#sets coolcol equal to 2 column headers, sets coolcol_data equal to the corresponding data, and prints the description of coolcol_data
coolcol=['LotShape','LotConfig']
coolcol_data=data[coolcol]
print (coolcol_data.describe())

#sets y equal to the prediction target, the sale price, and sets data_predictors equal to various column headers, and sets X equal to the corresponding data
y=sale_price
data_predictors=['LotArea','YearBuilt','1stFlrSF','2ndFlrSF','FullBath','BedroomAbvGr','TotRmsAbvGrd']
X=data[data_predictors]

test_y=tests.SalePrice
test_X=tests[data_predictors]
#imports Decision Tree Regressor and defines cool_model as that, it fits cool model to X and y and print predictions from the model
from sklearn.tree import DecisionTreeRegressor
cool_model = DecisionTreeRegressor()
cool_model.fit(X, y)
print(cool_model.predict(X.head()))

#imports the mean_absolute_error and train_test_split functions and defines the train and test data sets
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
train_X, val_X, train_y, val_y = train_test_split(X, y,random_state = 0)

#defines cool_model as a decision tree regressor and fits the model to the training set
cool_model = DecisionTreeRegressor()
cool_model.fit(train_X, train_y)

#sets val_data_predict equal to the predictions on
val_data_predict=cool_model.predict(val_X)
print(mean_absolute_error(val_y, val_data_predict))

#defines the get_mae function which returns the mean absolute error of the predictions of the data for different numbers of leafs
def get_mae(max_leaf_nodes, predictors_train, predictors_val, targ_train, targ_val):
    model = DecisionTreeRegressor(max_leaf_nodes=max_leaf_nodes, random_state=0)
    model.fit(predictors_train, targ_train)
    preds_val = model.predict(predictors_val)
    mae = mean_absolute_error(targ_val, preds_val)
    return(mae)

#runs a loop for different amounts of leaves and returns the mean absolute error for each
for max_leaf_nodes in [5, 50, 500, 5000]:
    my_mae = get_mae(max_leaf_nodes, train_X, val_X, train_y, val_y)
    print("Max leaf nodes: %d  \t\t Mean Absolute Error:  %d" %(max_leaf_nodes, my_mae))

#imports the random forest regressor and sets forest_model equal to that and fits it to the training set
from sklearn.ensemble import RandomForestRegressor
forest_model = RandomForestRegressor()
forest_model.fit(X, y)

#sets data_predict equal to the prediction on the validation model and prints the mean absolute error of that
data_predict = forest_model.predict(test_X)
print(mean_absolute_error(test_y, data_predict))

my_submission = pd.DataFrame({'Id': tests.Id, 'SalePrice': data_predict})

my_submission.to_csv('submission.csv', index=False)