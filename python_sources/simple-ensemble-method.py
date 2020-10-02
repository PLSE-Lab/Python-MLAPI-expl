#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn import cross_validation
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.feature_selection import SelectFromModel
import seaborn as sns
import xgboost as xgb
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


train_data = pd.read_csv('../input/train.csv')
test_data = pd.read_csv('../input/test.csv')


# In[ ]:


id_test = test_data['ID']
target = train_data['TARGET'].values
X_train = train_data.drop(['ID','TARGET'], axis=1)
X_test = test_data.drop(['ID'], axis=1).values
print ("The number of features before the domentionality reduction approach : ",X_train.shape[1])


# ## Dimentionality Reduction using ExtraTreesClassifier 
# We have 369 features in train data, the goal of this step is to reduce the number of features and the dimesion of data, a lot of features are duplicated or they have zero variances. Some features are higly correlated. The dimentionality redcution imporve the quality of model because it reduce noises in data and accelerates the execution time of machine learning models. I decided to use and `Extra Trees Classifier` to do the dimentionality reduction. We can also use `Principal Components Analysis (PCA)` or `Linear Discriminant Analysis (LDA)` in order to reduce the dimension of data.

# In[ ]:


clf = ExtraTreesClassifier()
clf = clf.fit(X_train,target)
clf.feature_importances_
model = SelectFromModel(clf,prefit=True)
Xr_Train = model.transform(X_train)
Xr_Test = model.transform(X_test)
print ("The number of features after the domentionality reduction approach : ",Xr_Test.shape[1])


# ## Random Forest Classifier

# In[ ]:


clf = RandomForestClassifier(n_estimators=120, max_depth=17, random_state=1)
clf.fit(Xr_Train, target)
y_pred = clf.predict_proba(Xr_Test)
scores = cross_validation.cross_val_score(clf, Xr_Train, target, scoring='roc_auc', cv=5) 
print(scores.mean())
submission = pd.DataFrame({"ID":id_test, "TARGET":y_pred[:,1]})
submission.to_csv("submission_rfc.csv", index=False)


# ## XGB Classifier

# In[ ]:


xgbClassifier = xgb.XGBClassifier(n_estimators=580, max_depth=5, seed=1234, missing=np.nan, learning_rate=0.02, subsample=0.7, colsample_bytree=0.7, objective='binary:logistic') 
xgbClassifier.fit(Xr_Train,target)
y_xgb_pred = xgbClassifier.predict_proba(Xr_Test)
scores = cross_validation.cross_val_score(xgbClassifier, Xr_Train, target, scoring='roc_auc', cv=5) 
print(scores.mean())

