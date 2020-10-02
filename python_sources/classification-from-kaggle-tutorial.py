#!/usr/bin/env python
# coding: utf-8

# ### Exercise from the Kaggle Course
# 
# * https://www.kaggle.com/learn/feature-engineering

# In[ ]:


import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# In[ ]:


import numpy as np
import pandas as pd 
import seaborn as sns


# * Set the type of 'deadline', 'launched' as datetime type in pandas

# In[ ]:


df=pd.read_csv("/kaggle/input/kickstarter-projects/ks-projects-201801.csv", parse_dates=['deadline', 'launched'])


# In[ ]:


df.head()


# ### Target
# * drop live
# * set successful as 1
# * set the rest as 0

# In[ ]:


df.groupby('state')['ID'].count()


# In[ ]:


df = df.drop(df[df.state == "live"].index)
df.groupby('state')['ID'].count()


# In[ ]:


mapping = {"canceled": 0 , "failed" : 0, "suspended" : 0, "undefined": 0, "successful": 1}
df["outcome"] = df["state"].map(mapping)


# In[ ]:


(df.groupby('outcome')['ID'].count())


# In[ ]:


df["outcome"].hist()


# ### Categorical data
# * https://stackoverflow.com/questions/24458645/label-encoding-across-multiple-columns-in-scikit-learn[](http://)

# In[ ]:


df.head()


# In[ ]:


from sklearn.preprocessing import LabelEncoder
cat_features = ['category', 'currency', 'country']
encoder = LabelEncoder()
encoded = df[cat_features].apply(encoder.fit_transform)
encoded.head()


# In[ ]:


df = df.drop(df[cat_features], axis = 1)
df.head()


# In[ ]:


df = pd.concat([df, encoded], axis = 1)


# In[ ]:



df = df.assign(hour=df.launched.dt.hour,
               day=df.launched.dt.day,
               month=df.launched.dt.month,
               year=df.launched.dt.year)
df.head()


# In[ ]:


df = df[['goal', 'hour', 'day', 'month', 'year', 'category', 'currency', 'country', 'outcome']]
df.head()


# # Training test split

# In[ ]:


X = df[['goal', 'hour', 'day', 'month', 'year', 'category', 'currency', 'country']]
y = df["outcome"]

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    stratify=y, 
                                                    test_size=0.25)


# ## Training
# 

# In[ ]:


import lightgbm as lgb

dtrain = lgb.Dataset(X_train, label=y_train)
dvalid = lgb.Dataset(X_test, label=y_test)

param = {'num_leaves': 64, 'objective': 'binary'}
param['metric'] = 'auc'
num_round = 100
bst = lgb.train(param, dtrain, num_round, valid_sets=[dvalid], early_stopping_rounds=10, verbose_eval=False)


# ### Evaluation
# 

# In[ ]:


from sklearn import metrics
ypred = bst.predict(X_test)
score = metrics.roc_auc_score(y_test, ypred)

print(f"Test AUC score: {score}")


# In[ ]:


ypred.shape


# In[ ]:


from sklearn.metrics import roc_curve, auc , roc_auc_score
from matplotlib import pyplot

lr_fpr, lr_tpr, _ = roc_curve(y_test, ypred)


# In[ ]:


ns_probs = [0 for _ in range(len(y_test))]
ns_auc = roc_auc_score(y_test, ns_probs)
ns_fpr, ns_tpr, _ = roc_curve(y_test, ns_probs)

pyplot.plot(ns_fpr, ns_tpr, marker='.', label='None')
pyplot.plot(lr_fpr, lr_tpr, marker='.', label='model')
# axis labels
pyplot.xlabel('False Positive Rate')
pyplot.ylabel('True Positive Rate')
# show the legend
pyplot.legend()
# show the plot
pyplot.show()


# ### 2) Count encodings
# 
# Here, encode the categorical features `['category', 'currency', 'country']` using the count of each value in the data set. Using `CountEncoder` from the `category_encoders` library, fit the encoding using the categorical feature columns defined in `cat_features`. Then apply the encodings to the train and validation sets, adding them as new columns with names suffixed `"_count"`.

# In[ ]:


import category_encoders as ce


# In[ ]:


cat_features = ['category', 'currency', 'country']
count_enc = ce.CountEncoder(cols=cat_features)
count_enc.fit(X_train[cat_features])


# In[ ]:


train_encoded = X_train.join(count_enc.transform(X_train[cat_features]).add_suffix('_count'))
valid_encoded = X_test.join(count_enc.transform(X_test[cat_features]).add_suffix('_count'))


# In[ ]:


valid_encoded.head()


# # Train the model on the encoded dataset

# In[ ]:


import lightgbm as lgb

dtrain = lgb.Dataset(train_encoded, label=y_train)
dvalid = lgb.Dataset(valid_encoded, label=y_test)

param = {'num_leaves': 64, 'objective': 'binary'}
param['metric'] = 'auc'
num_round = 100
bst = lgb.train(param, dtrain, num_round, valid_sets=[dvalid], early_stopping_rounds=10, verbose_eval=False)


# In[ ]:


ypred = bst.predict(valid_encoded)
score = metrics.roc_auc_score(y_test, ypred)
print (score)


# * Count encoding is a good idea, how does it improve the model score?
# 
# * Rare values tend to have similar counts (with values like 1 or 2), so you can classify rare values together at prediction time. Common values with large counts are unlikely to have the same exact count as other values. So, the common/important values get their own grouping.

# ### 4) Target encoding
# 
# * supervised encodings that use the labels (the targets) to transform categorical features. 
# * Target encoding replaces a categorical value with the average value of the target for that value of the feature. 

# In[ ]:


import category_encoders as ce
cat_features = ['category', 'currency', 'country']
# Create the encoder itself
target_enc = ce.TargetEncoder(cols=cat_features)
# Fit the encoder using the categorical features and target
target_enc.fit(X_train[cat_features], y_train)

train_encoded = X_train.join(target_enc.transform(X_train[cat_features]).add_suffix('_target'))
valid_encoded = X_test.join(target_enc.transform(X_test[cat_features]).add_suffix('_target'))

train_encoded.head()


# In[ ]:


import lightgbm as lgb

dtrain = lgb.Dataset(train_encoded, label=y_train)
dvalid = lgb.Dataset(valid_encoded, label=y_test)

param = {'num_leaves': 64, 'objective': 'binary'}
param['metric'] = 'auc'
num_round = 100
bst = lgb.train(param, dtrain, num_round, valid_sets=[dvalid], early_stopping_rounds=10, verbose_eval=False)
ypred = bst.predict(valid_encoded)
score = metrics.roc_auc_score(y_test, ypred)
print (score)


# ### CatBoost Encoding
# * This is similar to target encoding in that it's based on the target probablity for a given value. However with CatBoost, for each row, the target probability is calculated only from the rows before it.

# In[ ]:


import category_encoders as ce
cat_features = ['category', 'currency', 'country']
# Create the encoder itself
target_enc = ce.CatBoostEncoder(cols=cat_features)
# Fit the encoder using the categorical features and target
target_enc.fit(X_train[cat_features], y_train)

train_encoded = X_train.join(target_enc.transform(X_train[cat_features]).add_suffix('_cb'))
valid_encoded = X_test.join(target_enc.transform(X_test[cat_features]).add_suffix('_cb'))

train_encoded.head()


# In[ ]:


import lightgbm as lgb

dtrain = lgb.Dataset(train_encoded, label=y_train)
dvalid = lgb.Dataset(valid_encoded, label=y_test)

param = {'num_leaves': 64, 'objective': 'binary'}
param['metric'] = 'auc'
num_round = 100
bst = lgb.train(param, dtrain, num_round, valid_sets=[dvalid], early_stopping_rounds=10, verbose_eval=False)
ypred = bst.predict(valid_encoded)
score = metrics.roc_auc_score(y_test, ypred)
print (score)


# # Creating new features from the raw data 

# * start over..

# In[ ]:


df=pd.read_csv("/kaggle/input/kickstarter-projects/ks-projects-201801.csv", parse_dates=['deadline', 'launched'])


# In[ ]:


interactions = df['category'] + "_" + df['country']
print(interactions.head(10))


# In[ ]:


from sklearn.preprocessing import LabelEncoder
label_enc = LabelEncoder()
df = df.assign(category_country=label_enc.fit_transform(interactions))
df.head()


# In[ ]:


launched = pd.Series(df.index, index=df.launched, name="count_7_days").sort_index()
launched.head(20)


# In[ ]:


import matplotlib.pyplot as plt
count_7_days = launched.rolling('7d').count() - 1
print(count_7_days.head(20))

# Ignore records with broken launch dates the first 7 numbers..
plt.plot(count_7_days[7:]);
plt.title("Competitions in the last 7 days");


# In[ ]:


count_7_days.index = launched.values  # launched.values  are the index
count_7_days = count_7_days.reindex(df.index)
count_7_days.head(10)


# In[ ]:


df = df.join(count_7_days)
df.head()


# Do projects in the same category compete for donors? If you're trying to fund a video game and another game project was just launched, you might not get as much money. We can capture this by calculating the time since the last launch project in the same category.
# 
# * .groupby then .transform. The .transform method takes a function then passes a series or dataframe to that function for each group. This returns a dataframe with the same indices as the original dataframe. In our case, we'll perform a groupby on "category" and use transform to calculate the time differences for each category.

# In[ ]:


def time_since_last_project(series):
    # Return the time in hours
    return series.diff().dt.total_seconds() / 3600.

df_temp = df[['category', 'launched']].sort_values('launched')
df_temp.head()


# In[ ]:



df_temp.groupby('category').count().head(10)


# In[ ]:


timedeltas = df_temp.groupby('category').transform(time_since_last_project)
timedeltas.head(20)


# We get NaNs here for projects that are the first in their category. We'll need to fill those in with something like the mean or median. We'll also need to reset the index so we can join it with the other data

# In[ ]:


timedeltas = timedeltas.fillna(timedeltas.median()).reindex(df.index)
timedeltas.head(20)


# In[ ]:


df["competitionValue"] = timedeltas


# In[ ]:


df.head()


# # train test split again

# In[ ]:


df = df.drop(df[df.state == "live"].index)
mapping = {"canceled": 0 , "failed" : 0, "suspended" : 0, "undefined": 0, "successful": 1}
df["outcome"] = df["state"].map(mapping)
from sklearn.preprocessing import LabelEncoder
cat_features = ['category', 'currency', 'country']
encoder = LabelEncoder()
encoded = df[cat_features].apply(encoder.fit_transform)
df = df.drop(df[cat_features], axis = 1)
df = pd.concat([df, encoded], axis = 1)
df = df.assign(hour=df.launched.dt.hour,
               day=df.launched.dt.day,
               month=df.launched.dt.month,
               year=df.launched.dt.year)
df.head()


# In[ ]:


X = df[['goal', 'hour', 'day', 'month', 'year', 'category', 'currency', 'country', 'category_country',
       'count_7_days', 'competitionValue']]
y = df["outcome"]

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    stratify=y, 
                                                    test_size=0.25)


# In[ ]:


import category_encoders as ce
cat_features = ['category', 'currency', 'country']
# Create the encoder itself
target_enc = ce.CatBoostEncoder(cols=cat_features)
# Fit the encoder using the categorical features and target
target_enc.fit(X_train[cat_features], y_train)

train_encoded = X_train.join(target_enc.transform(X_train[cat_features]).add_suffix('_cb'))
valid_encoded = X_test.join(target_enc.transform(X_test[cat_features]).add_suffix('_cb'))

train_encoded.head()


# # Train

# In[ ]:


import lightgbm as lgb

dtrain = lgb.Dataset(train_encoded, label=y_train)
dvalid = lgb.Dataset(valid_encoded, label=y_test)

param = {'num_leaves': 64, 'objective': 'binary'}
param['metric'] = 'auc'
num_round = 100
bst = lgb.train(param, dtrain, num_round, valid_sets=[dvalid], early_stopping_rounds=10, verbose_eval=False)
ypred = bst.predict(valid_encoded)
score = metrics.roc_auc_score(y_test, ypred)
print (score)


# # feature selections

# In[ ]:


from sklearn.feature_selection import SelectKBest, f_classif
# Keep 5 features
selector = SelectKBest(f_classif, k=5)
X_new = selector.fit_transform(train_encoded, y_train)
X_new


# In[ ]:


# Get back the features we've kept, zero out all other features
selected_features = pd.DataFrame(selector.inverse_transform(X_new), 
                                 index=train_encoded.index, 
                                 columns=train_encoded.columns)
selected_features.head()


# In[ ]:


selected_columns = selected_features.columns[selected_features.var() != 0]
train_encoded[selected_columns].head()


# In[ ]:


import lightgbm as lgb

dtrain = lgb.Dataset(train_encoded[selected_columns], label=y_train)
dvalid = lgb.Dataset(valid_encoded[selected_columns], label=y_test)

param = {'num_leaves': 64, 'objective': 'binary'}
param['metric'] = 'auc'
num_round = 100
bst = lgb.train(param, dtrain, num_round, valid_sets=[dvalid], early_stopping_rounds=10, verbose_eval=False)
ypred = bst.predict(valid_encoded[selected_columns])
score = metrics.roc_auc_score(y_test, ypred)
print (score)


# # L1 regularization
# Univariate methods consider only one feature at a time when making a selection decision. Instead, we can make our selection using all of the features by including them in a linear model with L1 regularization. This type of regularization (sometimes called Lasso) penalizes the absolute magnitude of the coefficients, as compared to L2 (Ridge) regression which penalizes the square of the coefficients.

# In[ ]:


from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SelectFromModel


# In[ ]:


# Set the regularization parameter C=1
logistic = LogisticRegression(C=1, penalty="l1", random_state=7).fit(train_encoded, y_train)
model = SelectFromModel(logistic, prefit=True)

X_new = model.transform(train_encoded)


# In[ ]:


# Get back the features we've kept, zero out all other features
selected_features_2 = pd.DataFrame(model.inverse_transform(X_new), 
                                 index=train_encoded.index, 
                                 columns=train_encoded.columns)
selected_features_2.head()


# In[ ]:


selected_columns = selected_features_2.columns[selected_features_2.var() != 0] # var() --> variance of columns
train_encoded[selected_columns].head()


# In[ ]:


import lightgbm as lgb

dtrain = lgb.Dataset(train_encoded[selected_columns], label=y_train)
dvalid = lgb.Dataset(valid_encoded[selected_columns], label=y_test)

param = {'num_leaves': 64, 'objective': 'binary'}
param['metric'] = 'auc'
num_round = 100
bst = lgb.train(param, dtrain, num_round, valid_sets=[dvalid], early_stopping_rounds=10, verbose_eval=False)
ypred = bst.predict(valid_encoded[selected_columns])
score = metrics.roc_auc_score(y_test, ypred)
print (score)


# # extra 
# 
# * np.log transform the goal column

# In[ ]:




