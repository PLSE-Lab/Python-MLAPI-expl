#!/usr/bin/env python
# coding: utf-8

# ### Importing libraries

# In[62]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pylab as plt
plt.rcParams["figure.figsize"] = [8,5]
import seaborn as sns
sns.set()
import re
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, train_test_split, StratifiedKFold
from sklearn.metrics import (roc_auc_score, confusion_matrix)
from sklearn.cluster import KMeans
import xgboost as xgb


import os
print(os.listdir("../input"))


# ### Importing dataset + light preprocessing

# In[28]:


infile = "../input/train_sample.csv"
df = pd.read_csv(infile)
df["attributed_time"] = df["attributed_time"].fillna("0")
df["click_time"] = pd.to_datetime( df["click_time"])
df["ct_month"] = df["click_time"].map( lambda x: x.month)
df["ct_year"] = df["click_time"].map( lambda x: x.year)
df["ct_day"] = df["click_time"].map( lambda x: x.day)
df["ct_timeofday"] = df["click_time"].map( lambda x: x.hour*60 + x.minute)

df_click_span = df.groupby("ip").agg({"click_time": lambda x : (x.max() - x.min()).seconds }).reset_index(level=0).rename(columns={"click_time":"click_time_span"})
df = df.merge( df_click_span, how="left", on="ip")
df_click_count = df.groupby("ip").count()[["channel"]].reset_index(level=0).rename(columns={"channel":"click_count"})
df = df.merge( df_click_count, how="inner", on="ip")

del df["click_time"]
del df["attributed_time"]


# We simply added two features: one counting the number of clicks per ip address, one counting the time span between the first and the last one.

# In[71]:


df.head(10)


# ### Class balance of the data

# In[63]:


df["is_attributed"].plot(kind="hist",normed=True,bins=2)
ax = plt.gca()
ax.set_yticks(())
ax.set_xticks( (.25,.75))
ax.set_xticklabels( ["attributed", "not attributed"])
df["is_attributed"].value_counts()


# The training set is very imbalanced with ~0.25% of the data labelised "not attributed".

# ### Constructing features and label vectors

# In[68]:


features = ["app", "device", "os", "channel", "ct_year", "ct_month", "ct_day", "ct_timeofday","click_time_span","click_count"]
X = df[features].values
y = df["is_attributed"].values


# ## Predictive models

# This first trial will use different classifiers (logistic regression, random forests and boosted trees to be be compared to a baseline given by the dummy classifier).  The train/test split is done with no regard to the class balance between the two, and the classifiers are not passed any arguments to penalize misclassification of the minority class.

# In[69]:


print( "-"*80)
print( "-"*80)
print( "\t\tNO REBALANCE")
print( "-"*80)
print( "-"*80)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)

dc = DummyClassifier(strategy="most_frequent")
lr = LogisticRegression(C=0.035)
rf = RandomForestClassifier( n_estimators=11, max_depth=9)
bt = xgb.XGBClassifier(max_depth=10, n_estimators=11, eta=0.6)

for classifier in (dc,lr,rf,bt):
    classifier.fit(X_train,y_train)
    y_pred_proba = classifier.predict_proba( X_test)[:,1]
    y_pred = y_pred_proba > 0.5
    print( "-"*80)
    print( "%s" % type(classifier).__name__)
    print( "\tconfusion matrix: ", confusion_matrix(y_test, y_pred).tolist())
    print( "\troc_auc score: %.3f" % roc_auc_score(y_test, y_pred_proba))


# The tree classifier dominate over the logistic regression which is very normal since we have not done any preprocessing work for scaling or one-hot encoding the features. It is almost surprising that the logistic regression is closer to the trees than to the baseline.
# 
# Most classification mistakes are false negatives, which is a normal consequence of the extreme minority of positive examples.
# 
# Note that if you run this block several times, the number of the non-dummy classifiers will fluctuate. This is because the number of positives in test/train will change significantly. This is because of using pure randomness in splitting train/test.

# In[70]:


print( "-"*80)
print( "-"*80)
print( "\t\tWITH REBALANCE")
print( "-"*80)
print( "-"*80)
s = StratifiedKFold(n_splits=10,shuffle=True)
indices_train, indices_test = list(s.split(X,y))[0]
X_train, X_test, y_train, y_test = X[indices_train], X[indices_test], y[indices_train], y[indices_test]
c_weights = { val:1./np.mean(y==val) for val in np.unique(y)}
scale_pos = np.sum(y==0)/np.sum(y==1)

dc = DummyClassifier(strategy="most_frequent")
lr = LogisticRegression(C=0.035, class_weight=c_weights)
rf = RandomForestClassifier( n_estimators=11, max_depth=9, class_weight=c_weights)
bt = xgb.XGBClassifier(max_depth=10, n_estimators=11, eta=0.6, scale_pos_weight=scale_pos)

for classifier in (dc,lr,rf,bt):
    classifier.fit(X_train,y_train)
    y_pred_proba = classifier.predict_proba( X_test)[:,1]
    y_pred = y_pred_proba > 0.5
    print( "-"*80)
    print( "%s" % type(classifier).__name__)
    print( "\tconfusion matrix: ", confusion_matrix(y_test, y_pred).tolist())
    print( "\troc_auc score: %.3f" % roc_auc_score(y_test, y_pred_proba))


# Here the splitting is done with stratification, implying that the ratio of positives in test and train are very similar. Second, the classifiers were passed an argument to be aware of the class imbalance (`c_weights` or `scale_pos`). These increases the false negative penality over the false positive one. The consequence is that the rate of false negative has diminished significantly and the roc_auc score also took a jump in the right direction.

# TO DO:
#     - more preprocessing
#     - look into feature engineering
#     - hyperparameter selection
#     - possibly stack multiple models

# In[ ]:




