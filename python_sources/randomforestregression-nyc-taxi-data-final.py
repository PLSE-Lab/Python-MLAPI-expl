#!/usr/bin/env python
# coding: utf-8

# # RandomForest-NYC Taxi data

# Objective:
# ----------
# Build a model that predicts the total trip duration of taxi trips in New York City with least RMSE using Random Forest Regressor algorithm

# In[36]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[37]:


#my_local_path = "B:/UPX docs/Machine Learning/Project_datasets/Project datasets modified/NYC Taxi Trip/NYC Taxi Trip/"
taxi_data = pd.read_csv('../input/Taxi_new.csv')
taxi_data.head(5)


# In[38]:


y = taxi_data["trip_duration"].values

columns = ["hour_of_day", "day_of_month", "month_of_date", "day_of_week_num","distance"]
x = taxi_data[list(columns)].values
x


# In[39]:


from sklearn.ensemble import RandomForestRegressor

forest = RandomForestRegressor(max_depth = 10, n_estimators = 100, random_state = 1)


# Training Random Forest Model

# In[40]:


# Fitting the model on Train Data

RF = forest.fit(x, y)


# In[41]:


print(RF.score(x, y))


# In[42]:


modelPred = RF.predict(x)


# In[43]:


list(zip(columns,RF.feature_importances_))


# In[44]:


RF.get_params


# In[45]:


from sklearn.metrics import mean_squared_error
from math import sqrt


# In[46]:


meanSquaredError=mean_squared_error(y, modelPred)
print("MSE:", meanSquaredError)


# In[47]:


rootMeanSquaredError = sqrt(meanSquaredError)
print("RMSE:", rootMeanSquaredError)


# #  Randomized Search

# In[48]:


# Different parameters we want to test

# Number of trees in random forest
n_estimators = [int(x) for x in np.linspace(start = 10, stop = 100, num = 10)]
# Number of features to consider at every split
max_features = ['auto', 'sqrt']
# Maximum number of levels in tree
max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
max_depth.append(None)
# Minimum number of samples required to split a node
min_samples_split = [2, 5, 10]
# Minimum number of samples required at each leaf node
min_samples_leaf = [1, 2, 4]
# Method of selecting samples for training each tree
bootstrap = [True, False]


# In[49]:


random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'bootstrap': bootstrap}


# In[50]:


# Importing RandomizedSearchCV

from sklearn.model_selection import RandomizedSearchCV


# In[51]:


RS_RF = RandomForestRegressor()

# Fitting 3 folds for each of 50 candidates, totalling 300 fits
rf_random = RandomizedSearchCV(estimator = RS_RF, param_distributions = random_grid, 
                               n_iter = 50, cv = 3, verbose=2, random_state=42)


# In[ ]:


rf_random.fit(x,y)


# ![image.png](attachment:image.png)![](http://)

# In[ ]:


rf_random.best_params_


# ![image.png](attachment:image.png)

# In[ ]:


best_forest = RandomForestRegressor(max_depth = 90, n_estimators = 90,min_samples_split= 2,min_samples_leaf= 4,
                                    max_features= 'sqrt',bootstrap= True, random_state = 1)


# In[ ]:


best_RF = best_forest.fit(x, y)


# In[ ]:


print(best_RF.score(x, y))


# ![image.png](attachment:image.png)

# In[ ]:


best_modelPred = best_RF.predict(x)


# In[ ]:


best_meanSquaredError=mean_squared_error(y, best_modelPred)
print("MSE:", best_meanSquaredError)


# In[ ]:


best_rootMeanSquaredError = sqrt(best_meanSquaredError)
print("RMSE:", best_rootMeanSquaredError)


# ![image.png](attachment:image.png)

# # Comparing models with train/test split and RMSE  

# In[ ]:


from sklearn import metrics


# In[ ]:


from sklearn.model_selection import train_test_split

# define a function that accepts a list of features and returns testing RMSE
def train_test_rmse(columns):
    X = taxi_data[columns]
    Y = taxi_data.trip_duration
    X_train, X_test, y_train, y_test = train_test_split(X, Y, random_state=123)
    RFreg = RandomForestRegressor(max_depth = 90, n_estimators = 90,min_samples_split= 2,min_samples_leaf= 4,
                                    max_features= 'sqrt',bootstrap= True, random_state = 1)
    RFreg.fit(X_train, y_train)
    y_pred = RFreg.predict(X_test)
    return np.sqrt(metrics.mean_squared_error(y_test, y_pred))


# In[ ]:


print (train_test_rmse(['distance', 'hour_of_day','day_of_week_num', 'day_of_month','month_of_date']))


# ![image.png](attachment:image.png)

# Conclusion:
# -----------
# Random Forest Regressor model has reduced the overall RMSE value 363 seconds to 355 seconds when compared with Decision Tree Regressor model. 
# 
# Further we can say that Randomised search technique which is used for selection of best parameters has helped us pick the best parameters and thus helping us obtain the least RMSE score for the test values of x.
# 
# 

# In[ ]:




