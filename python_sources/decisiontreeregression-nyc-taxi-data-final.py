#!/usr/bin/env python
# coding: utf-8

# # DECISION TREE MODEL FOR NYC

# Objective:
# ----------
# Build a model that predicts the total trip duration of taxi trips in New York City with least RMSE using Decision tree Regressor algorithm

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


#my_local_path = "B:/UPX docs/Machine Learning/Project_datasets/Project datasets modified/NYC Taxi Trip/NYC Taxi Trip/"
taxi_data = pd.read_csv('../input/Taxi_new.csv')
taxi_data.head(5)


# In[ ]:



y = taxi_data["trip_duration"].values

columns = ["hour_of_day", "day_of_month", "month_of_date", "day_of_week_num","distance"]
x = taxi_data[list(columns)].values
x


# In[ ]:



from sklearn import tree
my_tree_one = tree.DecisionTreeRegressor(criterion="friedman_mse", max_depth=10, random_state=42)
my_tree_one


# # TRAINING THE MODEL

# In[ ]:


my_tree_one = my_tree_one.fit(x,y)


# In[ ]:


print(my_tree_one.score(x, y))


# In[ ]:


x_pred=my_tree_one.predict(x, check_input=True)
print('This is the length of predicted values of duration:',len(x_pred))
print(x_pred)


# In[ ]:


# Visualize the decision tree graph

with open('tree.dot','w') as dotfile:
    tree.export_graphviz(my_tree_one, out_file=dotfile, feature_names=columns, filled=True)
    dotfile.close()
    
# You may have to install graphviz package using 
# conda install graphviz
# conda install python-graphviz

from graphviz import Source

with open('tree.dot','r') as f:
    text=f.read()
    plot=Source(text)
plot   


# In[ ]:


from sklearn import metrics


# In[ ]:


print ('MAE:', metrics.mean_absolute_error(y, x_pred))
print ('MSE:', metrics.mean_squared_error(y, x_pred))
print ('RMSE:', np.sqrt(metrics.mean_squared_error(y, x_pred)))


# In[ ]:


list(zip(columns,my_tree_one.feature_importances_))


# # Comparing models with train/test split and RMSE  

# In[ ]:


from sklearn.model_selection import train_test_split

# define a function that accepts a list of features and returns testing RMSE
def train_test_rmse(columns):
    X = taxi_data[columns]
    Y = taxi_data.trip_duration
    X_train, X_test, y_train, y_test = train_test_split(X, Y, random_state=123)
    DTreg = tree.DecisionTreeRegressor(criterion="friedman_mse", max_depth=10, random_state=123)
    DTreg.fit(X_train, y_train)
    y_pred = DTreg.predict(X_test)
    return np.sqrt(metrics.mean_squared_error(y_test, y_pred))


# In[ ]:


print (train_test_rmse(['distance', 'hour_of_day', 'day_of_month']))
print (train_test_rmse(['distance', 'hour_of_day','day_of_week_num']))
print (train_test_rmse(['distance', 'hour_of_day']))
print (train_test_rmse(['distance', 'hour_of_day','day_of_week_num', 'day_of_month','month_of_date']))
print (train_test_rmse(['distance']))


# **Findint the best parameters for the Decision Tree using Grid Search**

# In[ ]:


max_depth = [10,15,20] 
criterion = ['mse', 'friedman_mse']


# In[ ]:


from sklearn.model_selection import GridSearchCV
#import GridSearchCV


# In[ ]:


DT_GS = tree.DecisionTreeRegressor()
grid = GridSearchCV(estimator = DT_GS, cv=3, 
                    param_grid = dict(max_depth = max_depth, criterion = criterion))


# In[ ]:


grid.fit(x,y)


# In[ ]:


grid.best_score_


# In[ ]:


# Best parameters for the model

grid.best_params_


# In[ ]:


new_DT_GS = tree.DecisionTreeRegressor(criterion= 'friedman_mse', max_depth= 10, random_state=42)


# In[ ]:


new_DT_GS.fit(x,y)


# In[ ]:


new_DT_GS.score(x,y)


# In[ ]:


grid_modelPred = new_DT_GS.predict(x)


# In[ ]:


from sklearn.metrics import mean_squared_error
from math import sqrt


# In[ ]:


grid_meanSquaredError=mean_squared_error(y, grid_modelPred)
print("MSE:", grid_meanSquaredError)


# In[ ]:


grid_rootMeanSquaredError = sqrt(grid_meanSquaredError)
print("RMSE:", grid_rootMeanSquaredError)


# Conclusion:
# -----------
# We can observe that in our Decision Tree Regressor model there is significant reduction in Root Mean Sqaure error from 369 seconds to 363 seconds after using the best parameters which were obtained by using the Grid search technique for the Trip Duration Predictive model
