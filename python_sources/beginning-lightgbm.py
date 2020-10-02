#!/usr/bin/env python
# coding: utf-8

# Hello there,  this is my first kernel ever in Kaggle. So I will be happy to get any suggestions from you.
# 
# I have used LightGBM Classifier for the classification. This is all what I have done untill now:
# * Replaced NaN values with 0
# * Dropped object type columns for time being
# * Used Class weights to cater the class imbalance problem
# * Leaderboard score is **0.415**

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[ ]:


train_df = pd.read_csv("../input/train.csv")
test_df = pd.read_csv("../input/test.csv")
test_Id = test_df['Id']


# In[ ]:


train_df.info()


# In[ ]:


test_df.info()


# In[ ]:


print (train_df.shape, test_df.shape)
train_df.head()


# In[ ]:


# lets bar plot to see the frequencies of targets
train_df['Target'].value_counts().plot.bar()


# In[ ]:


def preprocess(train_df, test_df, dropna = False, drop_obtype = True, replace_nan_with = 0, return_with_top100 = False, lgbfactor = 0):

    train_df.drop(columns = ['Id'],inplace = True, errors="ignore")
    test_df.drop(columns = ['Id'], inplace = True, errors = "ignore")
    nanvalues = train_df.isnull().sum(axis = 0).values
    nullvalues = pd.DataFrame({"Column":train_df.columns, "Count":nanvalues})
    nullvalues.sort_values(by=["Count"], ascending = False, inplace = True)
    #nullvalues.head()
    nancols = nullvalues[nullvalues['Count']>0]["Column"].values.tolist()
    
    # columns with object type. We are dropping them for time being.
    objcols = train_df[train_df.columns[(train_df.dtypes == 'object').values]].columns.values.tolist()
    
    if (drop_obtype == True):
        train_df.drop(columns = objcols,inplace = True)
        test_df.drop(columns = objcols,inplace = True)
    if (dropna == True):
        train_df.drop(columns = nancols, inplace = True)
        test_df.drop(columns = nancols, inplace = True)
    if (dropna == False):
        train_df.replace(np.nan, replace_nan_with, inplace = True)
        test_df.replace(np.nan, replace_nan_with, inplace = True)
    
    
    labels = train_df['Target']
    train_df.drop(columns = ['Target'],inplace = True, errors="ignore")

    
    # Below we use Random Forest to select top 100 columns. Actually in LightGBM classifier I pass all features.
    from sklearn.ensemble import RandomForestClassifier
    rfc = RandomForestClassifier()
    rfc.fit(train_df, labels)
    
    scores = rfc.feature_importances_.tolist()
    report = pd.DataFrame({"feature":train_df.columns, "score":scores})
    report.sort_values(by = ["score"],ascending = False, inplace = True)
    top100cols = report.head(100)['feature'].values.tolist()
    
    
    if (return_with_top100 == True):
        train_df = train_df[top100cols]
        test_df = test_df[top100cols]
    
    # Since the classes are imbalance. We need to give weights to each class and pass this parameter in LightGBM classifier.
    from sklearn.utils import class_weight
    class_weights = class_weight.compute_class_weight('balanced',
                                                     np.unique(labels),
                                                     labels)
    # You might be wondering what is lgbfactor here. Actually LightGBM takes labels starting from 0 i.e 0,1,2,3 instead of 1,2,3,4
    # so we have to subtract 1 from labels while we use LightGBM.
    # Note that while submitting the predictions, we have to add 1 again to the predictions so that our labels are from 1,2,3,4
    class_weights = dict(zip(np.unique(labels.values-lgbfactor), class_weights))
    
    
    return train_df, test_df, labels, nancols, objcols, report, top100cols, class_weights
    


# In[ ]:


train, test, labels, nancols,objcols, report, top100cols, class_weightss = preprocess(train_df.copy(), test_df.copy(),dropna = False, lgbfactor = 1)


# In[ ]:


print (train.shape, test.shape)


# In[ ]:





# In[ ]:


import lightgbm as lgb
import sklearn.model_selection as model_selection
from sklearn.metrics import f1_score, make_scorer


# In[ ]:





# In[ ]:


lgmodel = lgb.LGBMClassifier(class_weight=class_weightss, metric = "multi_logloss",num_class = 4)
#rf_classif = RandomForestClassifier()


# In[ ]:





# In[ ]:


def get_score(model, train, label, fold):
    score = model_selection.cross_val_score(model, train , label, cv = fold, scoring = make_scorer(f1_score, average = "macro"))
    return score.mean()


# In[ ]:


kf = model_selection.KFold(n_splits=5, shuffle=True,random_state=2017)
templabel = labels.loc[train['parentesco1']==1]
temptrain = train[train['parentesco1']==1]
heads_scores = get_score(lgmodel, temptrain, templabel-1, kf) # subtract 1 because of LightGBM, remember the expalantion above 0,1,2,3
overall_scores = get_score(lgmodel, train, labels-1, kf) # subtract 1 because of LightGBM, remember the expalantion above 0,1,2,3
print ("Heads mean score:",heads_scores.mean())
print ("Overall mean score:",overall_scores.mean())


# In[ ]:


lgmodel.fit(train, labels-1)
lgpreds = lgmodel.predict(test)


# In[ ]:


# Making a submission file #
sub_df = pd.DataFrame({"Id":test_Id.values})
sub_df["Target"] = lgpreds + 1 # ----------> note here we add 1 to the predictions making the predictions start from 1
sub_df.to_csv("LightGBM.csv", index=False)


# **I know there is still much work remaining like Preprocessing, feature selection, feature engineering, one hot encoding the object type columns.**

# In[ ]:




