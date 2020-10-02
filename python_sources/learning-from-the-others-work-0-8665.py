#!/usr/bin/env python
# coding: utf-8

# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from xgboost.sklearn import XGBClassifier


# In[ ]:



np.random.seed(0)

#Loading data
df_train = pd.read_csv('../input/train_users_2.csv')
df_test = pd.read_csv('../input/test_users.csv')
labels = df_train['country_destination'].values
df_train = df_train.drop(['country_destination'], axis=1)
id_test = df_test['id']
piv_train = df_train.shape[0]


# In[ ]:


#Creating a DataFrame with train+test data
df_all = pd.concat((df_train, df_test), axis=0, ignore_index=True)
#Removing id and date_first_booking
df_all = df_all.drop(['id', 'date_first_booking'], axis=1)
#Filling nan
df_all = df_all.fillna(-1)


# In[ ]:


#####Feature engineering#######
#date_account_created
dac = np.vstack(df_all.date_account_created.astype(str).apply(lambda x: list(map(int, x.split('-')))).values)
df_all['dac_year'] = dac[:,0]
df_all['dac_month'] = dac[:,1]
df_all['dac_day'] = dac[:,2]
df_all = df_all.drop(['date_account_created'], axis=1)


# In[ ]:


#timestamp_first_active
tfa = np.vstack(df_all.timestamp_first_active.astype(str).apply(lambda x: list(map(int, [x[:4],x[4:6],x[6:8],x[8:10],x[10:12],x[12:14]]))).values)
df_all['tfa_year'] = tfa[:,0]
df_all['tfa_month'] = tfa[:,1]
df_all['tfa_day'] = tfa[:,2]
df_all = df_all.drop(['timestamp_first_active'], axis=1)


# In[ ]:


#Age
av = df_all.age.values
df_all['age'] = np.where(np.logical_or(av<14, av>100), -1, av)


# In[ ]:


#One-hot-encoding features
ohe_feats = ['gender', 'signup_method', 'signup_flow', 'language', 'affiliate_channel', 'affiliate_provider', 'first_affiliate_tracked', 'signup_app', 'first_device_type', 'first_browser']
for f in ohe_feats:
    df_all_dummy = pd.get_dummies(df_all[f], prefix=f)
    df_all = df_all.drop([f], axis=1)
    df_all = pd.concat((df_all, df_all_dummy), axis=1)


# In[ ]:


#Splitting train and test
vals = df_all.values
X = vals[:piv_train]
le = LabelEncoder()
y = le.fit_transform(labels)   
X_test = vals[piv_train:]


# In[ ]:


#Classifier
xgb = XGBClassifier(max_depth=6, learning_rate=0.3, n_estimators=25,
                    objective='multi:softprob', subsample=0.5, colsample_bytree=0.5, seed=0)                  
xgb.fit(X, y)
y_pred = xgb.predict_proba(X_test)  


# In[ ]:


import matplotlib.pyplot as plt
import seaborn as sns

# Draw inline
get_ipython().run_line_magic('matplotlib', 'inline')
# Set figure aesthetics
sns.set_style("white", {'ytick.major.size': 10.0})
sns.set_context("poster", font_scale=1.1)


# In[ ]:


#Taking the 5 classes with highest probabilities
ids = []  #list of ids
cts = []  #list of countries
for i in range(len(id_test)):
    idx = id_test[i]
    ids += [idx] * 5
    cts += le.inverse_transform(np.argsort(y_pred[i])[::-1])[:5].tolist()

#Generate submission
sub = pd.DataFrame(np.column_stack((ids, cts)), columns=['id', 'country'])
#sub.to_csv('../input/sub1.csv',index=False)


# >This prediction giving the score 0.86659 but why? Can I see, what are the features that are most important for selecting the destination countries?

# In[ ]:


import xgboost as xgb


# In[ ]:


# loading the input data into dmatrix
dtrain = xgb.DMatrix(X, label=y)


# In[ ]:


#number of class labels
number_of_classes = len(le.classes_)


# In[ ]:


# Parameter of gradient boosted tree
params = {"objective": "multi:softprob",
          "eval_metric" : "mlogloss",
          "num_class" : number_of_classes}


# In[ ]:


# Training xgboost on training data with 50 boosters
nround = 50
bst = xgb.train(params=params, dtrain = dtrain, num_boost_round=nround, verbose_eval=1)


# In[ ]:


# Visualizing the train xgboost
featuresNames = df_all.columns


# In[ ]:


dictScore = bst.get_fscore()


# In[ ]:


feature_number_to_name = dict(zip(sorted(dictScore.keys(), key = lambda x: int(x[1:])), featuresNames))


# In[ ]:


n = 20
topN = sorted(((feature_number_to_name[num], val) for num, val in 
                              dictScore.items()), key=lambda x:x[1], reverse=True)[:n]
    
dictScore_feature_name = dict(topN)


# In[ ]:


# Plotting the feature importance but do not know what is doing
xgb.plot_importance(dictScore_feature_name)
print()


# From the above graph, we can see what are the most important features but I don't know why?

# In[ ]:




