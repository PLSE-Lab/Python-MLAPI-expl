#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# In[ ]:



import pandas as pd

# Path of the file to read
fire_file_path = '../input/fires-from-space-australia-and-new-zeland/fire_archive_M6_96619.csv'

# Fill in the line below to read the file into a variable space_data
space_data = pd.read_csv(fire_file_path) 

# Call line below with no argument to check that you've loaded the data correctly
#step_1.check()
space_data.describe()


# In[ ]:


print(space_data)


# In[ ]:


import pandas as pd

# Path of the file to read
nrt_file_path = '../input/fires-from-space-australia-and-new-zeland/fire_nrt_M6_96619.csv'

# Fill in the line below to read the file into a variable fire_data
fire_data = pd.read_csv(nrt_file_path) 

# Call line below with no argument to check that you've loaded the data correctly
#step_1.check()
fire_data.describe()


# In[ ]:


print(fire_data)


# In[ ]:


fire_data.columns


# In[ ]:


y = fire_data.brightness


# In[ ]:


# Create the list of features below
feature_names = ['latitude','longitude','scan','track','bright_t31','frp','confidence']

# Select data corresponding to features in feature_names
X = fire_data[feature_names]


# In[ ]:


# Review data
# print description or statistics from X
#print(_)
X.describe()


# In[ ]:


# print the top few lines
#print(_)
X.head()


# In[ ]:


# from _ import _
#specify the model. 
#For model reproducibility, set a numeric value for random_state when specifying the model
from sklearn.tree import DecisionTreeRegressor
fire_model = DecisionTreeRegressor(random_state=1)

# Fit the model
fire_model.fit(X,y) 


# In[ ]:


print(X.head())


# In[ ]:


predictions = fire_model.predict(X)
print(predictions)


# In[ ]:


print(y)


# In[ ]:


# Import the train_test_split function and uncomment
from sklearn.model_selection import train_test_split


# fill in and uncomment
train_X, val_X, train_y, val_y = train_test_split(X, y, random_state = 1)


# In[ ]:


# You imported DecisionTreeRegressor in your last exercise
# and that code has been copied to the setup code above. So, no need to
# import it again

# Specify the model
fire_model = DecisionTreeRegressor(random_state = 1)

# Fit fire_model with the training data
fire_model.fit(train_X, train_y)


# In[ ]:



# Predict with all validation observations
val_predictions = fire_model.predict(val_X)


# In[ ]:




# print the top few validation predictions
print(y.head())
# print the top few brightness from validation data
print((val_y, val_predictions))


# In[ ]:


from sklearn.metrics import mean_absolute_error
val_mae =mean_absolute_error(val_y, val_predictions)

# uncomment following line to see the validation_mae
print(val_mae)
#val_predictions = fire_model.predict(val_X)


# In[ ]:



def get_mae(max_leaf_nodes, train_X, val_X, train_y, val_y):
    model = DecisionTreeRegressor(max_leaf_nodes=max_leaf_nodes, random_state=0)
    model.fit(train_X, train_y)
    preds_val = model.predict(val_X)
    mae = mean_absolute_error(val_y, preds_val)
    return(mae)


# In[ ]:



candidate_max_leaf_nodes = [5, 25, 50, 100, 250, 500]
# Write loop to find the ideal tree size from candidate_max_leaf_nodes
for max_leaf_nodes in [5, 50, 500, 5000]:
    scores = {leaf_size: get_mae(leaf_size, train_X, val_X, train_y, val_y) for leaf_size in candidate_max_leaf_nodes}
    print("Max leaf nodes: %d  \t\t Mean Absolute Error:  %d" %(max_leaf_nodes, val_mae))

# Store the best value of max_leaf_nodes (it will be either 5, 25, 50, 100, 250 or 500)
best_tree_size = min(scores , key=scores.get)


# In[ ]:


# Fill in argument to make optimal size and uncomment
final_model = DecisionTreeRegressor(max_leaf_nodes=best_tree_size, random_state=1)

# fit the final model and uncomment the next two lines
final_model.fit(X,y, )


# In[ ]:


from sklearn.ensemble import RandomForestRegressor

# Define the model. Set random_state to 1
rf_model =  RandomForestRegressor()




# fit your model
rf_model.fit(train_X, train_y)

# Calculate the mean absolute error of your Random Forest model on the validation data
rf_val_predictions = rf_model.predict(val_X)
rf_val_mae = mean_absolute_error(rf_val_predictions, val_y)

print("Validation MAE for Random Forest Model: {}".format(rf_val_mae))


# In[ ]:


# To improve accuracy, create a new Random Forest model which you will train on all training data
rf_model_on_full_data = RandomForestRegressor()

# fit rf_model_on_full_data on all data from the training data
rf_model_on_full_data.fit(X, y)


# In[ ]:




