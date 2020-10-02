#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
from sklearn.model_selection import train_test_split 
from sklearn.metrics import mean_squared_error

get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


df = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")


# In[ ]:


df.head()
# test.head()


# In[ ]:


df.info()
print("")
test.info()


# In[ ]:


df.isnull().sum()
# test.isnull().sum()


# In[ ]:


df[df.isnull().any(axis=1)]


# In[ ]:


test[test.isnull().any(axis=1)]


# In[ ]:


df = df.fillna(df.mean())
test = test.fillna(test.mean())


# In[ ]:


df.corr()


# In[ ]:


test.head()


# In[ ]:


df["rating"].value_counts()


# In[ ]:


sb.pairplot(df)


# # Actual work

# In[ ]:


def preprocess(df):
    from sklearn import preprocessing
    
    encoded = pd.get_dummies(df.type)
    final = df.iloc[:,np.r_[1:10,11:13]]
    final = pd.concat([final,encoded],axis=1)
                            
    final = final.astype('float64')
                            
#     scaler = preprocessing.MinMaxScaler()
#     final_scaled = scaler.fit_transform(final)
    
#     final_df = pd.DataFrame(final)
#     final_df.columns = final.columns
                        
    return final


# In[ ]:


X = preprocess(df)
Y = pd.DataFrame(df.rating)
Y.head(10)

X_final = preprocess(test)
X_final.head()


# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.25)


# In[ ]:


# Linear Regression (sad model)
from sklearn.linear_model import LinearRegression

lin_reg = LinearRegression()
lin_reg.fit(X_train, y_train)
pred = lin_reg.predict(X_test)
test_set_rmse = (np.sqrt(mean_squared_error(y_test, pred)))
test_set_rmse


# In[ ]:


# Random Forest + Grid Search (NOT WORKING dunno why)

from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestRegressor

def rfr_model(X, y):
    
    # Perform Grid-Search
    gsc = GridSearchCV(
        estimator=RandomForestRegressor(),
        param_grid={
            'max_depth': range(6,12),
            'n_estimators': (2, 5, 10, 25),
        },
        cv=5, 
        scoring='neg_mean_squared_error', 
        verbose=0, 
        n_jobs=-1
    )
    
    grid_result = gsc.fit(X, y)
    best_params = grid_result.best_params_
    
    rfr = RandomForestRegressor(
        max_depth=best_params["max_depth"], 
        n_estimators=best_params["n_estimators"],
        random_state=False, 
        verbose=False
    )
    
    rfr.fit(X,y)
    
    # Perform K-Fold CV
    scores = cross_val_score(rfr, X, y, cv=5, scoring='neg_mean_absolute_error')
    
    predicted = rfr.predict(X_final)

    return predicted

preds = rfr_model(X, Y["rating"].ravel())


# In[ ]:


# Random Forest

from sklearn.ensemble import ExtraTreesRegressor
rf = ExtraTreesRegressor(n_estimators = 60, max_depth=25)
rf.fit(X_train, y_train)
pred = rf.predict(X_test)
pred = pred.round()
test_set_rmse = (np.sqrt(mean_squared_error(y_test, pred)))
test_set_rmse


# In[ ]:


# Random Forest

from sklearn.ensemble import ExtraTreesRegressor
rf = ExtraTreesRegressor(n_estimators = 60, max_depth=25)
rf.fit(X, Y["rating"].ravel())
preds = rf.predict(X_final)
outDF = pd.DataFrame(preds)
outDF.columns = ["rating"]
outDF = outDF.round()
outDF = outDF.astype('int64')

final_output = pd.concat([test["id"], outDF], axis=1)
final_output.to_csv("output.csv", index=False)


# In[ ]:




