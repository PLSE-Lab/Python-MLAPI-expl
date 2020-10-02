#!/usr/bin/env python
# coding: utf-8

# (This post has been edited after Philippe Lonjoux's insightful comment. The amendments I made are highlighted in my answer to his comment.)
# 
# 
# Hello there!
# 
# This notebook exploits the analysis I did [here][1] to boost the performance of a classifier.
# 
# In particular, I will show how additional features computed from 'manager_id' can substantially improve performances by building upon the neat ["Random Forest Starter"][2] by Li Li
# 
#   [1]: https://www.kaggle.com/den3b81/two-sigma-connect-rental-listing-inquiries/do-managers-matter-some-insights-on-manager-id
#   [2]: https://www.kaggle.com/aikinogard/two-sigma-connect-rental-listing-inquiries/random-forest-starter-with-numerical-features

# In[ ]:


# ...let's import the modules
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_validation import train_test_split
from sklearn.metrics import log_loss

import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


# ... and load the training data
df = pd.read_json(open("../input/train.json", "r"))


# ### Naive feature engineering (from Li Li's Random Forest Starter)

# In[ ]:


df["num_photos"] = df["photos"].apply(len)
df["num_features"] = df["features"].apply(len)
df["num_description_words"] = df["description"].apply(lambda x: len(x.split(" ")))
df["created"] = pd.to_datetime(df["created"])
df["created_year"] = df["created"].dt.year
df["created_month"] = df["created"].dt.month
df["created_day"] = df["created"].dt.day

features_to_use = ["bathrooms", "bedrooms", "latitude", "longitude", "price",
                   "num_photos", "num_features", "num_description_words",
                   "created_year", "created_month", "created_day"]


# ### Basic encoding of 'manager_id', from SRK ["XGBoost starter"][1] 
# 
# 
#   [1]: https://www.kaggle.com/sudalairajkumar/two-sigma-connect-rental-listing-inquiries/xgb-starter-in-python

# In[ ]:


from sklearn import preprocessing

lbl = preprocessing.LabelEncoder()
lbl.fit(list(df['manager_id'].values))
df['manager_id'] = lbl.transform(list(df['manager_id'].values))

# let's add this feature
features_to_use.append('manager_id')


# ### Let's now add 3 columns contaning the fractions of 'low','medium' and 'high' interest level obtained by each manager in the training dataset. 
# 
# ### We also add the simple 'manager_skill' feature I introduced in my [previous notebook][1].
# 
# ### We use mean values for those managers who don't have enough entries for being ranked (minimum = 20 here).
# 
# 
# ----------
# 
# 
# ### As pointed out by Philippe Lonjoux's, since these new features involve the target variable, we split the dataset first so to avoid "cheating" during the validation phase. The features are computed for each manager on the training part and the values obtained are then copied for the instances in the validation dataset. 
# 
# 
#   [1]: https://www.kaggle.com/den3b81/two-sigma-connect-rental-listing-inquiries/do-managers-matter-some-insights-on-manager-id.

# In[ ]:


# Let's split the data
X = df[features_to_use]
y = df["interest_level"]
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.33)


# In[ ]:


# compute fractions and count for each manager
temp = pd.concat([X_train.manager_id,pd.get_dummies(y_train)], axis = 1).groupby('manager_id').mean()
temp.columns = ['high_frac','low_frac', 'medium_frac']
temp['count'] = X_train.groupby('manager_id').count().iloc[:,1]

# remember the manager_ids look different because we encoded them in the previous step 
print(temp.tail(10))


# In[ ]:


# compute skill
temp['manager_skill'] = temp['high_frac']*2 + temp['medium_frac']

# get ixes for unranked managers...
unranked_managers_ixes = temp['count']<20
# ... and ranked ones
ranked_managers_ixes = ~unranked_managers_ixes

# compute mean values from ranked managers and assign them to unranked ones
mean_values = temp.loc[ranked_managers_ixes, ['high_frac','low_frac', 'medium_frac','manager_skill']].mean()
print(mean_values)
temp.loc[unranked_managers_ixes,['high_frac','low_frac', 'medium_frac','manager_skill']] = mean_values.values
print(temp.tail(10))


# In[ ]:


# inner join to assign manager features to the managers in the training dataframe
X_train = X_train.merge(temp.reset_index(),how='left', left_on='manager_id', right_on='manager_id')
X_train.head()


# In[ ]:


# add the features computed on the training dataset to the validation dataset
X_val = X_val.merge(temp.reset_index(),how='left', left_on='manager_id', right_on='manager_id')
new_manager_ixes = X_val['high_frac'].isnull()
X_val.loc[new_manager_ixes,['high_frac','low_frac', 'medium_frac','manager_skill']] = mean_values.values
X_val.head()


# In[ ]:


# add manager fractions and skills to the features to use
features_to_use.extend(['high_frac','low_frac', 'medium_frac','manager_skill'])


# ### Let's train and validate a few random forest classifiers to see whether we can improve performances with thee additional features

# ### Basic model with no manager-related features, this is the model in Li Li's Random Forest Starter

# In[ ]:


# features to use for this classifier == only basic numerical
these_features = [f for f in features_to_use if f not in ['manager_id','high_frac','low_frac', 'medium_frac','manager_skill']]

clf = RandomForestClassifier(n_estimators=1000)
clf.fit(X_train[these_features], y_train)
y_val_pred = clf.predict_proba(X_val[these_features])
log_loss(y_val, y_val_pred)


# In[ ]:


# Let's visualize features importance, 
# price is the most important feature, followed by number of descriptive words, latitude and longitude
pd.Series(index = these_features, data = clf.feature_importances_).sort_values().plot(kind = 'bar')


# ### Let's add manager_id and see if we can get some improvement already

# In[ ]:


# add manager_id
these_features = [f for f in features_to_use if f not in ['high_frac','low_frac', 'medium_frac','manager_skill']]

clf = RandomForestClassifier(n_estimators=1000)
clf.fit(X_train[these_features], y_train)
y_val_pred = clf.predict_proba(X_val[these_features])
log_loss(y_val, y_val_pred)


# In[ ]:


# Let's visualize features importance
pd.Series(index = these_features, data = clf.feature_importances_).sort_values().plot(kind = 'bar')


# ### We get a small improvement, but we can do better. Let's remove 'manager_id' and use manager interest fractions and skill instead

# In[ ]:


# no manager_id, use fractions and skill instad
these_features = [f for f in features_to_use if f not in ['manager_id']]

clf = RandomForestClassifier(n_estimators=1000)
clf.fit(X_train[these_features], y_train)
y_val_pred = clf.predict_proba(X_val[these_features])
log_loss(y_val, y_val_pred)


# In[ ]:


# Let's visualize features importance
pd.Series(index = these_features, data = clf.feature_importances_).sort_values().plot(kind = 'bar')


# ### That's an improvement, but maybe we can do better if we just use the fractions.

# In[ ]:


# no manager_id, no skill, use fractions
these_features = [f for f in features_to_use if f not in ['manager_id','manager_skill']]

clf = RandomForestClassifier(n_estimators=1000)
clf.fit(X_train[these_features], y_train)
y_val_pred = clf.predict_proba(X_val[these_features])
log_loss(y_val, y_val_pred)


# In[ ]:


# Let's visualize features importance
pd.Series(index = these_features, data = clf.feature_importances_).sort_values().plot(kind = 'bar')


# ### What if we use the manager skill and not the fractions?!

# In[ ]:


# no manager_id, no fraction, use skill instead
these_features = [f for f in features_to_use if f not in ['manager_id','high_frac','low_frac', 'medium_frac']]

clf = RandomForestClassifier(n_estimators=1000)
clf.fit(X_train[these_features], y_train)
y_val_pred = clf.predict_proba(X_val[these_features])
log_loss(y_val, y_val_pred)


# In[ ]:


# Let's visualize features importance
pd.Series(index = these_features, data = clf.feature_importances_).sort_values().plot(kind = 'bar')


# ### Oh cool, manager_skill still does boost our performances further and it is almost as important as the price feature.
# 
# ### Therefore, I suggest you to use this feature in your classifier, no matter its nature. I am confident you gonna get a nice boost if you are not including similar features already in your model.
# 
# 
# ### Cheers
