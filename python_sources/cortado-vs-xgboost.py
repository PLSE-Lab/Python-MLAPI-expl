#!/usr/bin/env python
# coding: utf-8

# **cortado** contains a high performance implementation of extreme gradient boosted tree. Let's try it!
# 
# First, we install cortado and xgboost:

# In[ ]:


get_ipython().system('pip install cortado')
get_ipython().system('pip install xgboost')


# We will be using a dataset with 1M observations:

# In[ ]:


import pandas as pd

csvpath = "../input/airlinetrain1m/airlinetrain1m.csv"
df_pd = pd.read_csv(csvpath)
df_pd.head()


# In[ ]:


df_pd.info()


# We have 2 numeric columns and 7 categorical ones. We would like to use *dep_delayed_15min* as our label and all other columns as features.
# 
# Let's start with XGBoost. It expects all features to be numeric so we will one hot encode categorical features with pandas *get_dummies* and convert all data into sparse format:

# In[ ]:


from scipy.sparse import coo_matrix
import numpy as np

covariates_xg = ["DepTime", "Distance"]
factors_xg = ["Month", "DayofMonth", "DayOfWeek", "UniqueCarrier", "Origin", "Dest"]

sparse_covariates = list(map(lambda col: df_pd[col].astype(pd.SparseDtype("float32", 0.0)), covariates_xg))
sparse_factors = list(map(lambda col: pd.get_dummies(df_pd[col], prefix=col, sparse=True, dtype=np.float32), factors_xg))

data = pd.concat(sparse_factors + sparse_covariates, axis=1)
sparse_data = coo_matrix(data.sparse.to_coo()).tocsr()


# We also need to make the label a numeric feature:

# In[ ]:


label_xg = df_pd["dep_delayed_15min"].map({"N": 0, "Y": 1})


# We will use these model parameters:

# In[ ]:


eta = 0.1
nrounds = 100
max_depth = 6


# **cortado** extreme boosted tree is single threaded at the moment and effectively uses *tree_method = 'exact'* so we will use the same options for easy comparison:

# In[ ]:


import xgboost as xgb
from datetime import datetime

start = datetime.now()
model = xgb.XGBClassifier(max_depth=max_depth, nthread=1, learning_rate=eta, tree_method="exact", n_estimators=nrounds)
model.fit(sparse_data, label_xg)
pred_xg = model.predict_proba(sparse_data)
end = datetime.now()
print("xgboost elapsed: {e}".format(e=(end - start)))


# We can now try to run extreme boosted tree model in **cortado**.
# 
# First we import the data directly from pandas dataframe:

# In[ ]:


import cortado as cr

df_cr = cr.DataFrame.from_pandas(df_pd)


# During the import all numeric features will be converted into cortado *covariates* and non numeric into *factors*:

# In[ ]:


df_cr.covariates


# In[ ]:


df_cr.factors


# **cortado** expects all features in a boosted tree model to be categorical factors. We have 2 numeric covariates which we can easily convert to factors:

# In[ ]:


deptime = cr.Factor.from_covariate(df_cr["DepTime"])
distance = cr.Factor.from_covariate(df_cr["Distance"])


# *Factor.from_covariate()* will bucketize the covariate using all of its unique values. Internally factors keep a list of unique levels and a *uint8* or *uint16* array of level indices:

# In[ ]:


deptime.levels[:5]


# *Factor.from_covariate* is a lazy operation: it does not compute the actual factor data (level indices). This will save memory but if you want speed then you can cache any factor or covariate in memory:**

# In[ ]:


deptime = deptime.cached()
distance = distance.cached()


# *cortado* logistic xgboost implementation expects the label to be numeric (covariate). We can easily convert from a factor into a covariate:

# In[ ]:


dep_delayed_15min = df_cr["dep_delayed_15min"]
label = cr.Covariate.from_factor(dep_delayed_15min, lambda level: level == "Y")
print(label)


# We can now create a list of factors which will be used in the model as features:

# In[ ]:


factors = df_cr.factors + [deptime, distance]
factors.remove(dep_delayed_15min)


# *cortado* supports categorical data out of the box so we do not have to create dummy vars, hot encode etc. It knows a difference between ordinal and not ordinal factors. For ordinal factors the tree boosting algorithm will only consider range splits: < x and >= x. For non ordinal factors the possible splits are only "level x " vs "not level x". Each factor has a property *isordinal*: 
# 

# In[ ]:


deptime.isordinal


# In[ ]:


df_cr["Month"].isordinal


# *deptime* is ordinal because it is a result of bucketization so its levels are naturally ordered.

# We can now run extreme boosted tree with *xgblogit*:

# In[ ]:


start = datetime.now()
trees, pred_cr = cr.xgblogit(label, factors,  eta = eta, lambda_ = 1.0, gamma = 0.0, minh = 1.0, nrounds = nrounds, maxdepth = max_depth, slicelen=1000000)
end = datetime.now()
print("cortado elapsed: {e}".format(e=(end - start)))


# The result is a list of *trees* and predicted probabilities:

# In[ ]:


from sklearn.metrics import roc_auc_score
y = label.to_array() # convert to numpy array
auc_cr = roc_auc_score(y, pred_cr) # cortado auc
auc_xg = roc_auc_score(y, pred_xg[:, 1]) # xgboost auc
print("cortado auc: {auc_cr}".format(auc_cr=auc_cr))
print("xgboost auc: {auc_xg}".format(auc_xg=auc_xg))
diff = np.max(np.abs(pred_xg[:, 1] - pred_cr))
print("max pred diff: {diff}".format(diff=diff))


# Cortado and XGBoost give the same result but cortado is 3x faster!
