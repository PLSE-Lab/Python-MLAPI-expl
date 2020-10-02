#!/usr/bin/env python
# coding: utf-8

# As you may well know in most Kaggle competitions the winners usually resort to stacking or meta-ensembling, which is a technique involving the combination of several 1st level predictive models to generate a 2nd level model which tends to outperform all of them.  
# 
# This usually happens because the 2nd level model is somewhat able to exploit the strengths of each 1st level model where they perform best, while smoothing the impact of their weaknesses in other parts of the dataset.
# 
# There are different methods and "schools of thought" on how stacking can be performed. If you are interested in this topic, then I suggest you to have a look at [this][1] and [this][2] to start. 
# 
# Here I will show a simple technique that is known as "Soft Voting" and can be implemented with Sklearn VotingClassifier.
# 
# 
#   [1]: http://blog.kaggle.com/2016/12/27/a-kagglers-guide-to-model-stacking-in-practice/
#   [2]: http://mlwave.com/kaggle-ensembling-guide/

# In[ ]:


# import what we need

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier 
from xgboost import XGBClassifier


# First, let's read the data and perform the feature engineering for numerical features of [Li Li's notebook][1].
# 
# 
#   [1]: https://www.kaggle.com/aikinogard/two-sigma-connect-rental-listing-inquiries/random-forest-starter-with-numerical-features

# In[ ]:


df = pd.read_json(open("../input/train.json", "r"))


# In[ ]:


df["num_photos"] = df["photos"].apply(len)
df["num_features"] = df["features"].apply(len)
df["num_description_words"] = df["description"].apply(lambda x: len(x.split(" ")))
df["created"] = pd.to_datetime(df["created"])
df["created_year"] = df["created"].dt.year
df["created_month"] = df["created"].dt.month
df["created_day"] = df["created"].dt.day


# In[ ]:


num_feats = ["bathrooms", "bedrooms", "latitude", "longitude", "price",
             "num_photos", "num_features", "num_description_words",
             "created_year", "created_month", "created_day"]
X = df[num_feats]
y = df["interest_level"]
X.head()


# Let's split the training data in two for validation

# In[ ]:


# random state for reproducing same results
random_state = 54321

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.33, random_state = 54321)


# For this little experiment we will combine the predictions of 4 different classifiers, trained with basic parametrization. Tuning the parameters is beyond the scope of this notebook.
# 
# The classifiers are:
# 1) RandomForestClassifier with "entropy" criterion
# 2) RandomForestClassifier with "gini" criterion
# 3) Sklearn GradientBoostingClassifier
# 4) XGBoost classifier
# 
# 

# In[ ]:


rf1 = RandomForestClassifier(n_estimators=250, criterion='entropy',  n_jobs = -1,  random_state=random_state)
rf1.fit(X_train, y_train)
y_val_pred = rf1.predict_proba(X_val)
log_loss(y_val, y_val_pred)


# In[ ]:


rf2= RandomForestClassifier(n_estimators=250, criterion='gini',  n_jobs = -1, random_state=random_state)
rf2.fit(X_train, y_train)
y_val_pred = rf2.predict_proba(X_val)
log_loss(y_val, y_val_pred)


# In[ ]:


gbc = GradientBoostingClassifier(random_state=random_state)
gbc.fit(X_train, y_train)
y_val_pred = gbc.predict_proba(X_val)
log_loss(y_val, y_val_pred)


# In[ ]:


xgb = XGBClassifier(seed=random_state)
xgb.fit(X_train, y_train)
y_val_pred = xgb.predict_proba(X_val)
log_loss(y_val, y_val_pred)


# It looks like rf1 tops all the other classifiers, with XGB coming last (as I said before, we are not concerned with tuning the parameters). 

# Let's now see whether we can improve performances by combining model predictions through "Soft Voting". **Soft voting entails computing a weighed sum of the predicted probabilities of all models for each class.** If the weights are equal for all classifiers, then the probabilities are simply the averages for each class. 
# 
# The predicted label is the argmax of the summed prediction probabilities.

# Soft voting can be easily performed using a VotingClassifier, which will "retrain" the model prototypes we specified before. Please note that one could avoid retraining the models (in this case) and manually perform the soft voting using the output of each classifier's "predic_proba" method.

# In[ ]:


eclf = VotingClassifier(estimators=[
    ('rf1', rf1), ('rf2', rf2), ('gbc', gbc), ('xgb',xgb)], voting='soft')
eclf.fit(X_train, y_train)
y_val_pred = eclf.predict_proba(X_val)
log_loss(y_val, y_val_pred)


# We can already see a decent improvement with respect to the best performing model. We could improve further by using a different coefficients for each classifier, based on their performances.
# 
# For instance, it is common to assign greater weight to the best performing one.

# In[ ]:


eclf = VotingClassifier(estimators=[
    ('rf1', rf1), ('rf2', rf2), ('gbc', gbc), ('xgb',xgb)], voting='soft', weights = [3,1,1,1])
eclf.fit(X_train, y_train)
y_val_pred = eclf.predict_proba(X_val)
log_loss(y_val, y_val_pred)


# That seemed to work (a bit), but of course if we assign too much weight to the best classifier then we lose all the benefits of stacking.

# One could do without the extra modeling and computational efforts of stacking in most situations. However, it makes a difference in Kaggle contests. Stacking is particularly fruitful when combining 1st level models which are different in nature and complementary.
# 
# Cheers.
