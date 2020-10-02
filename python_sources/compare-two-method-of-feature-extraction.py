#!/usr/bin/env python
# coding: utf-8

# ## Compare two different methods of Feature Extraction: Tsvd VS RF
# 
# This post is aiming to compare two different categories of feature selection technique.
# 
# The first one is pretty traditional, just the normal dimensional reduction technique. The typical one of those examples are PCA/LDA/T-svd.
# 
# Second example is tree-based regressor(Random forest as example). Those kind of regressors are normally considered better for the fact that they will consider correlations between different feature and is capable of forward/backward feature elimination.
# 
# However, it is hard to say which one is better in this competition without further investgation, For something special showed in this competiition:
# 
# 1. There are many zeros exist in both trainset and testset. In this case, sparse matrix shows and tsvd is considered better to recongnize the eigenvector and eigenvalue.
# 
# 2. If you conpute correlations between feature, normally they ranges from 0.03 to 0.1, which suggests a weak corrrelation between each other.(np.corr can achieve this).
# 
# 3. It is possible to overfit dataset with random forest feature selection.

# **For this notebook, I want show my special thanks to @AmarjeetKumarRandom for his work in preprocessing and lightGBM, give me a good lesson on how it works!**
# 

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
from sklearn.preprocessing import MaxAbsScaler
from sklearn.ensemble import RandomForestRegressor
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns


# In[ ]:


train=pd.read_csv('../input/train.csv')
test=pd.read_csv('../input/test.csv')


# If you go check , you will find some zero constant columns.  What we do here is just delete them.
# 
# In the meanwhile, "ID" column cannot be used for regression so we delete it. I checked it locally and there is no duplicate values so we just use index to override it.
# 
# You can consider this as a label encoder transformation.

# In[ ]:


constant = train.nunique().reset_index()
constant.columns = ["col", "count"]
constant = constant.loc[constant["count"]==1]
train = train.drop(columns=constant.col,axis = 1)
test = test.drop(columns=constant.col,axis = 1)


# In[ ]:


y = train["target"]
train = train.drop(["ID","target"],axis=1)
test = test.drop("ID",axis=1)
train["ID"] = train.index
test["ID"] = test.index


# Here we use maxabs scaler to scale the data to help optimization converge in a faster speed.
# 
# Minmax is also a preferable choice. Anyone will work here, given that they can preserve the sparse of the matrix.

# In[ ]:


def maxabs(train,test):
    scaler = MaxAbsScaler()
    scaler.fit(train)
    train = scaler.transform(train)
    test = scaler.transform(test)
    return train,test


# ## Test plan
# Here we will copy our datasets and use correspond datasets to check the performance. 
# 
# You can obtain result via Lightgbm training process.

# ## Test on tsvd algorithm

# In[ ]:


y_train=np.log1p(y)
##start to test RF and tsvd below


# In[ ]:


trainSVD = train.copy()
testSVD = test.copy()


# We will add some additional columns in the feature selection process.
# 
# Those are what being mentioned in the discussion section  and someone names it as "Row aggregation". 
# 
# In case for missing/NAN shows after modification, I will place some check at the end of the function.

# In[ ]:


def rowagg(train,test):
    ##
    train["sum"] = train.sum(axis=1)
    test["sum"] = test.sum(axis=1)
    train["var"] = train.var(axis=1)
    test["var"] = test.var(axis=1)
    train["median"] = train.median(axis=1)
    test["median"] = test.median(axis=1)
    train["mean"] = train.mean(axis=1)
    test["mean"] = test.mean(axis=1)
    train["std"] = train.std(axis=1)
    test["std"] = test.std(axis=1)
    train["max"] = train.max(axis=1)
    test["max"] = test.max(axis=1)
    train["min"] =train.min(axis=1)
    test["min"] = test.min(axis=1)
    train["skew"] = train.skew(axis=1)
    test["skew"] = test.skew(axis=1)
    print ("Null values in train: "+ str(np.sum(np.sum(pd.isnull(train)))))
    print ("NAN values in train: "+ str(np.sum(np.isnan(train.values))))
    print ("Null values in test: "+ str(np.sum(np.sum(pd.isnull(test)))))
    print ("NAN values in test: "+ str(np.sum(np.isnan(test.values))))
    return train,test


# In[ ]:


from sklearn.decomposition import TruncatedSVD
trainSVD,testSVD = rowagg(trainSVD,testSVD)
svd = TruncatedSVD(n_components=2000)
res = svd.fit(trainSVD)
print (np.sum(res.explained_variance_ratio_))


# In[ ]:


trainSVD = res.transform(trainSVD)
testSVD = res.transform(testSVD)


# **The following block of code are from @AmarjeetKumarRandom 's work, his lgb notebook is concise on how lgb runs. Really a good job.**
# 

# In[ ]:


import lightgbm as lgb
def run_lgb(X_train, Y_train, X_valid, Y_valid, test):
    seed = 42
    params = {
        "objective" : "regression",
        "metric" : "rmse",
        "task": "train",
        "boosting type":'dart',
        "num_leaves" :500,
        "learning_rate" : 0.005,
        "bagging_fraction" : 0.8,
        "feature_fraction" : 0.8,
        "bagging_frequency" : 5,
        "bagging_seed" : seed,
        "verbosity" : -1,
        "seed": seed
    }
    lgtrain = lgb.Dataset(X_train,label= Y_train)
    lgval = lgb.Dataset(X_valid,label =Y_valid)
    evals_result = {}
    model = lgb.train(params, lgtrain, 5000, 
                  valid_sets=[lgtrain, lgval], 
                  early_stopping_rounds=300, 
                  verbose_eval=100, 
                  evals_result=evals_result)
    lgb_prediction = np.expm1(model.predict(test, num_iteration=model.best_iteration))
    return lgb_prediction, model, evals_result


# In[ ]:


from sklearn.model_selection import train_test_split
trainSVD,testSVD = maxabs(trainSVD,testSVD)
X_train, X_test, Y_train, Y_test = train_test_split(trainSVD, y_train, test_size=0.1, random_state=0)
lgb_predSVD, model, evals_resultRF = run_lgb(X_train, Y_train, X_test, Y_test, testSVD)


# ## Test on Random forest feature selection

# In[ ]:


##stage2 :test on Random Forest
trainRF = train.copy()
testRF = test.copy()
trainRF,testRF = rowagg(trainRF,testRF)


# In[ ]:


rf_clf=RandomForestRegressor(random_state=42,n_jobs=-1)
rf_clf.fit(trainRF,y_train)
rank = pd.DataFrame()
rank["importance"] = np.array(rf_clf.feature_importances_)
rank["feature"] = np.array(trainRF.columns).T
rank = rank.sort_values(by=['importance'], ascending=False)
col = rank[:2000]


# In[ ]:


trainRF=trainRF[col.feature]
testRF=testRF[col.feature]
trainRF,testRF = maxabs(trainRF,testRF)
X_train, X_test, Y_train, Y_test = train_test_split(trainRF, y_train, test_size=0.1, random_state=0)
lgb_predRF, model, evals_resultRF = run_lgb(X_train, Y_train, X_test, Y_test, testRF)


# In[ ]:


sub=pd.read_csv('../input/sample_submission.csv')
sub["target"] = lgb_predRF
sub.to_csv('sub.csv', index=False)


# In[ ]:


sub.head()


# ## Summary
# It is surprising to see RF is still working better than tsvd base on cv score showed during lgb train process. 
# 
# You may wonder how it works in k-fold cv. I tested it locally(k=10) and as it turns out, on average, the RF's score is still better than tsvd's. Due to the kernel time limit, I cannot run k-fold test on this kernel because it takes longer than time allowed.
# 
# I guess you may find something here are questionable. If yes, please comment this notebook so I can see and update it.
