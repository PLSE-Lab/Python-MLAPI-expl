import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor

file = '../input/winequality-red.csv'

wine_data = pd.read_csv(file)

y = wine_data.quality # variable to be predicted
features = ['fixed acidity', 'citric acid', 'chlorides', 'free sulfur dioxide', 'total sulfur dioxide', 'density', 'pH' ,'sulphates', 'alcohol'] # input variables
X = wine_data[features]
train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=1) # split test data

# Decision Tree model

wine_model = DecisionTreeRegressor(random_state=1)
wine_model.fit(train_X, train_y) 

# Initial Decision Tree model without optimal amount of nodes
val_predictions = wine_model.predict(val_X)
val_mae = mean_absolute_error(val_predictions, val_y)
print("Validation MAE when not specifying nodes: {:,.0f}".format(val_mae))

# Finding optimal number of nodes for Decision Tree Model
for node in [5, 10, 25, 50, 75, 100, 250, 500, 1000, 2000]:
    wine_model = DecisionTreeRegressor(max_leaf_nodes=node, random_state=1)
    wine_model.fit(train_X, train_y)
    val_predictions = wine_model.predict(val_X)
    val_mae = mean_absolute_error(val_predictions, val_y)
    print("Validation MAE when specifying {} nodes: {}".format(node, val_mae))

# Random Forest Model  
rf_model = RandomForestRegressor()
rf_model.fit(train_X, train_y)
rf_val_predictions = rf_model.predict(val_X)
rf_val_mae = mean_absolute_error(rf_val_predictions, val_y)
print("Validation MAE when using Random Forest {}".format(rf_val_mae))


    
    