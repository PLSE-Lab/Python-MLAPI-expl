#!/usr/bin/env python
# coding: utf-8

# # Use naive Bayes results as feature of LightGBM
# 
# - naive Bayes alone is a very bad Classifier for the given dataset (ROC score = .62)
# - ROC too big to ignore
# - add as feature because performance of LGBM is much better

# Varaibles from previous steps
# ------------------------------
# 
# - features: numpy array containing features of train data
# - labels: labels of train data
# - test_feat: numpy array containing features of train data
# - lgbm_params: parameter for LGBM classifier

# In[ ]:


# code snippet
"""
from sklearn.naive_bayes import GaussianNB
import lightgbm as lgbm

# Perform naive Bayes classification
nb = GaussianNB()
nb.fit(features,labels)
nb_res  = gau.predict_proba(features)[:,1]

# add results of Bayes classification to feature
features  = np.c_[features,nb_res]
test_feat = np.c_[test_feat,nb.predict_proba(test_feat)]

# Perform LGBM classification
clf = lgbm.LGBMClassifier(**lgbm_params)
clf.fit(features,labels)
results = clf.preict_proba(test_feat)
"""


# Results
# -----------------------------
# 
# For my LGBM setting:
# - without Bayes : LB=.779
# - with Bayes: LB=.782

# In[ ]:




