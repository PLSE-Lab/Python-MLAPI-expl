# %% imports

import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split

# %% helpers

def get_mae(max_leaf_nodes, train_X, val_X, train_y, val_y):
    model = DecisionTreeRegressor(max_leaf_nodes=max_leaf_nodes, random_state=0)
    model.fit(train_X, train_y)
    preds_val = model.predict(val_X)
    mae = mean_absolute_error(val_y, preds_val)
    return(mae)

# %% data

melbourne_data = pd.read_csv("../input/melbourne_data.csv")
y = melbourne_data.Price
melbourne_features = ['Rooms', 'Bathroom', 'Landsize', 'Lattitude', 'Longtitude']
x = melbourne_data[melbourne_features]

# %% in sample model

in_sample_melbourne_model = DecisionTreeRegressor(random_state=1)
in_sample_melbourne_model.fit(x,y)

# %% validate in sample model

in_sample_predicted_home_prices = in_sample_melbourne_model.predict(x)
print("in sample validation")
print(mean_absolute_error(y, in_sample_predicted_home_prices))
print("")

# %% split sample model

train_x, val_x, train_y, val_y = train_test_split(x, y, random_state=0)
split_melbourne_model = DecisionTreeRegressor()
split_melbourne_model.fit(train_x, train_y)

split_predicted_home_prices = split_melbourne_model.predict(val_x)
print("split validation")
print(mean_absolute_error(val_y, split_predicted_home_prices))

# %% compare MAE with differing values of max_leaf_nodes

for max_leaf_nodes in [5, 50, 500, 5000]:
    my_mae = get_mae(max_leaf_nodes, train_x, val_x, train_y, val_y)
    print("Max leaf nodes: %d  \t\t Mean Absolute Error:  %d" %(max_leaf_nodes, my_mae))
