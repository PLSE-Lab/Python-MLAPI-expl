# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

iowa_file_path = '../input/iowahousing/train.csv'

home_data = pd.read_csv(iowa_file_path)

filtered_home_data = home_data.dropna(axis=1)
filtered_home_data.head()

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error

y = filtered_home_data.SalePrice
features = ['LotArea', 'YearBuilt', '1stFlrSF', '2ndFlrSF', 'FullBath', 'BedroomAbvGr', 'TotRmsAbvGrd']
X = home_data[features]

X_train,x_test,y_train,y_test = train_test_split(X,y,random_state=1)
iowa_model = DecisionTreeRegressor(random_state=1)
iowa_model.fit(X_train,y_train)
score = iowa_model.score(x_test,y_test)
print(score)

vals_predictions = iowa_model.predict(x_test)
val_mae = mean_absolute_error(vals_predictions,y_test)
print("Validation MAE: {:,.0f}".format(val_mae))

def get_mae(max_leaf_nodes,X_train,x_test,y_train,y_test):
    model = DecisionTreeRegressor(max_leaf_nodes=max_leaf_nodes,random_state=0)
    model.fit(X_train,y_train)
    preds_val = model.predict(x_test)
    mae = mean_absolute_error(y_test,preds_val)
    return(mae)

for max_leaf_nodes in [5, 25, 50, 100, 250, 500]:
    my_mae = get_mae(max_leaf_nodes,X_train,x_test,y_train,y_test)
    print("Max leaf nodes:%d \t\t Mean Absolute Error:%d" %(max_leaf_nodes,my_mae))

scores = {leaf_size: get_mae(leaf_size,X_train,x_test,y_train,y_test) for leaf_size in [5, 25, 50, 100, 250, 500]}
best_tree_size = min(scores, key=scores.get)
print(best_tree_size)

final_model = DecisionTreeRegressor(max_leaf_nodes=best_tree_size,random_state=1)
final_model.fit(X,y)