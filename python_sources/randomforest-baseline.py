#!/usr/bin/env python
# coding: utf-8

# # Strategy
# 
# * Set probem as regression (0-1)
# * Predict all format matches in single model
# * Because standing is defined by team (not each person), predict ranking in team unit.
# * Select `RandomforestRegressor` as baseline algorithm
# * Some feature enginerring (add/filter features)
# * While outlier treatment will be needed for higher accuracy, I won't do in baseline model construction.

# # import & load data

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import seaborn as sns
import lightgbm as lgb
from sklearn.metrics import mean_squared_error

get_ipython().run_line_magic('matplotlib', 'inline')
pd.set_option('display.max_columns', None)
pd.options.display.notebook_repr_html = True
pd.options.display.precision = 3

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV


# In[ ]:


train = pd.read_csv("../input/train_V2.csv")
test = pd.read_csv("../input/test_V2.csv")


# # feature extraction
# 
# * remove useless features
# * add some features
# * aggregate in team (min, mean, max)

# In[ ]:


def preprocess_pubg_data(df, label = True, frac = 1):
    """
    Do all preprocess in this function
    
    parameter
    ---
    label: whether df includes label column
    frac: sampling (0-1)
    
    return
    ---
    gid: groupID (correspond to each row of X)
    X: features
    y: target variable (if train = True)
    """
    
    # filter features
    df = df.drop(["Id", "matchId"], axis=1)
    df = df.drop(["numGroups"], axis=1)
    df = df.drop(["vehicleDestroys", "maxPlace", "roadKills", "teamKills", "rankPoints", "killPoints", "winPoints", "matchDuration"], axis=1)
    
    # add features
    df["moveDistance"] = df.loc[:, df.columns.str.endswith("Distance")].apply(np.sum, axis=1)
    
    # aggregate in team
    df_team = df.groupby("groupId").agg(["min", "mean", "max"])
    
    # remove NA
    df_team = df_team.dropna()
    
    # sampling
    df_team = df_team.sample(frac = frac, random_state = 123)
    gid = df_team.index.ravel()
    
    # return
    if label:
        X = df_team.drop("winPlacePerc", axis=1).values
        y = df_team["winPlacePerc"]["max"].values

        return gid, X, y
    
    else:
        X = df_team.values
        
        return gid, X


# In[ ]:


gid_train, X_train, y_train = preprocess_pubg_data(train, frac = 0.01)
gid_test, X_test = preprocess_pubg_data(test, False)


# # Training

# In[ ]:


param = {"n_estimators": [1000], 
         "max_features": [10, 20, 40],
         "max_depth": [5, 10, None]}


# In[ ]:


reg = GridSearchCV(estimator=RandomForestRegressor(),
             param_grid=param, cv=3, n_jobs=1, scoring="r2",
              return_train_score=False)


# In[ ]:


get_ipython().run_cell_magic('time', '', 'cv_result = reg.fit(X_train, y_train)')


# # Prediction

# In[ ]:


def my_predict(model, X_test):
    pred = model.predict(X_test)

    global test
    id_list = test.loc[:,["Id", "groupId"]]
    
    predict = pd.DataFrame({"groupId": gid_test, "winPlacePerc": pred})
    submission = pd.merge(id_list, predict, how="left", on="groupId").drop(["groupId"], axis=1)
    
    return submission


# In[ ]:


submission = my_predict(reg, X_test)
submission.to_csv("submission.csv", index=False)

