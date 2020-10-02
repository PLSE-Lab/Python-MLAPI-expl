#!/usr/bin/env python
# coding: utf-8

# Load Libraries.

# In[ ]:


import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# Load train set and peek data.

# In[ ]:


train_csv = os.path.join("../input", "train.csv")
train_base = pd.read_csv(train_csv)
train_base.head(5)


# split train data and validation data.

# In[ ]:


from sklearn.model_selection import train_test_split
X = train_base.drop('SalePrice', axis=1)
y = train_base['SalePrice'].copy()

X_train, X_validate, y_train, y_validate = train_test_split(X, y, random_state=0)


# list up features.

# In[ ]:


X_train.info()


# explore data

# In[ ]:


X_train['Street'].factorize()


# street and allay features are pave or not.

# In[ ]:


X_train['Alley'].factorize()


# In[ ]:


X_train['PoolQC'].factorize()


# In[ ]:


X_train['GarageCond'].factorize()


# I think that the size of the land and the building will affect the price, so let's first check the correlation.

# In[ ]:


attributes = ['LotFrontage', 'LotArea', 'TotalBsmtSF', 'SalePrice']
pd.plotting.scatter_matrix(train_base[attributes], figsize=(12,8))


# In[ ]:


train_base.plot(kind="scatter", x="SalePrice", y="LotArea", alpha=.1)


# In[ ]:


train_base.plot(kind="scatter", x="SalePrice", y="TotalBsmtSF", alpha=.1)


# In[ ]:


X_train_selected = X_train[["LotArea", "TotalBsmtSF"]]


# I will try to predict the combination of confirmed data in a linear model.

# In[ ]:


from sklearn.linear_model import LinearRegression

lin_reg = LinearRegression()
lin_reg.fit(X_train_selected, y_train)


# In[ ]:


X_train_selected.info()


# In[ ]:


from sklearn.metrics import mean_squared_error
y_pred = lin_reg.predict(X_train_selected)
lin_mse = mean_squared_error(y_train, y_pred)
lin_rmse =np.sqrt(lin_mse)
lin_rmse


# I got better score than not.

# Since most of the features are character string notation, one hot encoding is performed.

# In[ ]:


X_train_dummies = pd.get_dummies(X_train)
X_train_dummies.shape


# In[ ]:


X_train_dummies.head(10)


# In[ ]:


from sklearn.preprocessing import Imputer
imputer = Imputer(strategy="mean")
imputer.fit(X_train_dummies)
X_train_filled = imputer.transform(X_train_dummies)


# In[ ]:


from sklearn.ensemble import RandomForestRegressor

forest_reg = RandomForestRegressor(n_estimators=200, random_state=0)
forest_reg.fit(X_train_filled, y_train)


# In[ ]:


y_pred = forest_reg.predict(X_train_filled)
lin_mse = mean_squared_error(y_train, y_pred)
lin_rmse = np.sqrt(lin_mse)
lin_rmse


# In[ ]:


from sklearn.model_selection import cross_val_score
scores = cross_val_score(forest_reg, X_train_filled, y_train, scoring="neg_mean_squared_error", cv=10)
forest_rmse_scores = np.sqrt(-scores)

def display_scores(scores):
    print("Scores: ", scores)
    print("Mean: ", scores.mean())
    print("Standard devitation:", scores.std())
    
display_scores(forest_rmse_scores)


# In[ ]:


lin_reg2 = LinearRegression()
lin_reg2.fit(X_train_filled, y_train)


# In[ ]:


scores = cross_val_score(lin_reg2, X_train_filled, y_train, scoring="neg_mean_squared_error", cv=10)
lin_rmse_scores = np.sqrt(-scores)
display_scores(lin_rmse_scores)


# RandomForestRegressor is much better than LinearRegression.

# In[ ]:


test_csv = os.path.join("../input", "test.csv")
test_raw = pd.read_csv(test_csv)

test_concat = pd.concat([X_train, test_raw])
test_concat_dummies = pd.get_dummies(test_concat)


# In[ ]:


test_dummies = test_concat_dummies[1095:]


# In[ ]:


test_raw.head(1)


# In[ ]:


test_dummies.head(1)


# In[ ]:


for c1, c2 in zip(X_train_dummies.columns, test_dummies.columns):
    if c1 != c2:
        print(c1, ",", c2)


# In[ ]:


test_dummies = test_dummies.drop(['Condition2_PosA', 'Functional_Sev'], axis=1)
for c1, c2 in zip(X_train_dummies.columns, test_dummies.columns):
    if c1 != c2:
        print(c1, ",", c2)


# In[ ]:


test_filled = imputer.transform(test_dummies)


# In[ ]:


test_pred = forest_reg.predict(test_filled)


# In[ ]:


result_df = pd.DataFrame([test_dummies['Id'], test_pred], ['Id', 'SalePrice']).swapaxes(0,1)


# In[ ]:


result_df.head()


# In[ ]:


result_df.to_csv("house_prediction.csv")


# In[ ]:




