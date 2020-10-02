#!/usr/bin/env python
# coding: utf-8

# **I'm mainly making this kernel to share with you the problems that I see in the data and see if we can find a suitable solution for it.
# If you like this please upvote and in the discussion if you have any proposed solutions that would improve mine please share them.**

# ## Load Libraries

# In[ ]:


import pandas as pd
from sklearn.model_selection import cross_val_score, train_test_split
from xgboost import XGBRegressor
from sklearn.metrics import r2_score
from sklearn.pipeline import Pipeline
import numpy as np


# ## Load Data

# In[ ]:


train_df = pd.read_csv("../input/train.csv")
test_df = pd.read_csv("../input/test.csv")


# ## One hot encode categorical variables

# In[ ]:


data = train_df.append(test_df)
data = pd.get_dummies(data)
train, test = data[0:len(train_df)], data[len(train_df):]


# ## Organize data for training

# In[ ]:


X = train.drop(["y", "ID"], axis=1)
Y = train["y"]
X_Test = test.drop(["y", "ID"], axis=1)


# ## Handle duplicate values

# In[ ]:


def average_dupes(x):
	Y.loc[list(x.index)] = Y.loc[list(x.index)].mean()
    
dupes = X[X.duplicated()]
dupes.groupby(dupes.columns.tolist()).apply(average_dupes)


# ## Split into Train, Validation

# In[ ]:


X, XVal, Y, YVal = train_test_split(X, Y)


# ## Remove outliers
# ### This is optional, there are many ways to handle outliers

# In[ ]:


out = Y[Y > 125].index.values  # Approximately 0.02% of the data
X.drop(out, axis=0, inplace=True)
Y.drop(out, axis=0, inplace=True)


# ## Initialize XGBoost

# In[ ]:


xgb = XGBRegressor(n_jobs=-1, max_depth=2, colsample_bytree=0.7, min_child_weight=5, gamma=0.2, n_estimators=200, learning_rate=0.05, subsample=0.95)


# ## Score using Cross validation

# In[ ]:


print(np.mean(cross_val_score(xgb, X, Y, scoring="r2", n_jobs=-1, verbose=2, cv=3)))


# ## Train XGBoost

# In[ ]:


eval_set = [(XVal.as_matrix(), YVal.as_matrix())]
xgb.fit(X.as_matrix(), Y.as_matrix(), eval_set=eval_set, early_stopping_rounds=5)
print(r2_score(YVal, xgb.predict(XVal.as_matrix())))


# ## Save output for submission

# In[ ]:


Y_Test = xgb.predict(X_Test.as_matrix())

results_df = pd.DataFrame(data={'y':Y_Test}) 
ids = test_df["ID"]
joined = pd.DataFrame(ids).join(results_df)
joined.to_csv("mercedes.csv", index=False)

