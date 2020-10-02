#!/usr/bin/env python
# coding: utf-8

# What's the impact of the categorical features encoding on the model performance ?
# 
#  * let's start with the basic LabelEncoder
#    
#  * then compare with a custom encoding, based on the impact of each category on the loss

# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')

import itertools
import numpy as np
import pandas as pd
import seaborn as sns
import xgboost as xgb

from sklearn.ensemble import GradientBoostingRegressor

from sklearn.linear_model import SGDRegressor

from sklearn.metrics import mean_absolute_error

from sklearn.model_selection import KFold

from sklearn.preprocessing import FunctionTransformer
from sklearn.preprocessing import Imputer
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler


# In[ ]:


train = pd.read_csv('../input/train.csv')


# # 1- Extract the categories from `train` data

# In[ ]:


catFeatureslist = []
for colName,x in train.iloc[1,:].iteritems():
    if(str(x).isalpha()):
        catFeatureslist.append(colName)


# # 2- Transform categories using LabelEncoder

# In[ ]:


for cf in catFeatureslist:
    le = LabelEncoder()
    le.fit(train[cf].unique())
    train[cf] = le.transform(train[cf])


# Let's plot the relation between `cat100` and the `loss`

# In[ ]:


ax = sns.violinplot(train.cat100, train.loss)
ax.axis([-1,15,0,6000])


# As we can see, there is no trivial relation between the `cat100` values and the `loss`.
# We can have the intuition that it will be harder for a linear regression model to find a curve matching that plot.
# 
# But what about the other models, giving better performance on the Allstate Claims Severity challenge, like the ensemble boosting models ?

# # 3- Train and measure the MAE with LabelEncoder

# For the comparison, I'm using 2 models, whose hyper parameters have already been optimized:
# 
# * SGDRegressor
# * GradientBoostingRegressor
# 
# They are not the best models for this challenge, but the model fitting operation is fast.

# In[ ]:


clf_gbr = GradientBoostingRegressor(
    loss='ls',
    learning_rate=0.1,
    n_estimators=50,
    max_depth=5,
    max_features=0.12,
    random_state=69,
    subsample=0.5,
    verbose=0)


# In[ ]:


clf_sgdr = SGDRegressor(
    fit_intercept=False,
    loss='squared_loss',
    penalty='elasticnet',
    alpha=0.03,
    l1_ratio=0.7,
    learning_rate='invscaling',
    random_state=42,
    shuffle=True)


# In[ ]:


def evaluateModelPerf(clf, train, Y, Y_scaler=None):
    clf.fit(train, Y)
    print("Coefficient of determination on training set:", clf.score(train, Y))
    
    print("Score and Mean Absolute Error on the cross validation sets:")
    cv = KFold(n_splits=5, shuffle=True, random_state=33)
    maes = []
    scores = []
    for _, test_index in cv.split(train):
        Y_predict = clf.predict(train.iloc[test_index])
        if Y_scaler is not None:
            Y_scaled = Y[test_index] * Y_scaler.scale_ + Y_scaler.mean_
            Y_predict_scaled = Y_predict * Y_scaler.scale_ + Y_scaler.mean_
            mae = mean_absolute_error(Y_scaled, Y_predict_scaled)
        else:
            mae = mean_absolute_error(Y[test_index], Y_predict)
        score = clf.score(train.iloc[test_index], Y[test_index])
        print("score: {}, MAE: {}".format(score, mae))
        maes.append(mae)
        scores.append(score)

    print("Average score: {}".format(np.average(scores)))
    print("Average MAE: {}".format(np.average(maes)))


# In[ ]:


Y = train.loss
train = train.drop(["id", "loss"], axis=1)


# In[ ]:


evaluateModelPerf(clf_gbr, train, Y)


# SGDRegressor is sensitive to scaling and normalization:

# In[ ]:


scaler = StandardScaler()
train_scaled = scaler.fit_transform(train)
train_scaled = pd.DataFrame(train_scaled)
train_scaled.columns = train.columns
train = train_scaled

Y = scaler.fit_transform(Y[:, None])[:, 0]


# In[ ]:


evaluateModelPerf(clf_sgdr, train, Y, Y_scaler=scaler)


# # 4- Tranform categories based on their ordered impact on the loss

# Let's restart the process with a custom label encoding

# In[ ]:


train = pd.read_csv('../input/train.csv')


# In[ ]:


categoriesOrder = {}
for cat in catFeatureslist:
    medians = train.groupby([cat])['loss'].median()
    ordered_medians = sorted(medians.keys(), key=lambda x: medians[x])
    categoriesOrder[cat] = ordered_medians


# In[ ]:


def tranformCategories(X):
    for cat, order in categoriesOrder.items():
        class_mapping = {v: order.index(v) for v in order}
        X[cat] = X[cat].map(class_mapping)
    return X

ft = FunctionTransformer(tranformCategories, validate=False)
train = ft.fit_transform(train)


# In[ ]:


ax = sns.violinplot(train.cat100, train.loss)
ax.axis([-1,15,0,6000])


# # 5- Train and measure the MAE with the custom encoder

# In[ ]:


Y = train.loss
train = train.drop(["id", "loss"], axis=1)


# In[ ]:


evaluateModelPerf(clf_gbr, train, Y)


# In[ ]:


scaler = StandardScaler()
train_scaled = scaler.fit_transform(train)
train_scaled = pd.DataFrame(train_scaled)
train_scaled.columns = train.columns
train = train_scaled

Y = scaler.fit_transform(Y[:, None])[:, 0]


# In[ ]:


evaluateModelPerf(clf_sgdr, train, Y, Y_scaler=scaler)


# # 6- Conclusion

# We can notice that the categorized feature encoding method can have an impact on the model performance.
# 
# * The impact will be higher on linear regression model than on ensemble boosting model.
# 
# * The benefit will decrease as we increase the tree depth or the number of iterations.
# 
# But for fast analysis, in the early stage of the feature engineering process, it could be an interesting practice to consider.

# # 7- Notes

# When using, such encoding method, there might be some category values in the `test` data, that are missing from the `train` data.
# 
# Following method aims at identifying and replacing them with the median value for each category.

# In[ ]:


train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')


# In[ ]:


print("Categories found in test data, but not present in train data:")
for cat in catFeatureslist:
    testCat = set(test[cat].unique())
    trainCat = set(train[cat].unique())
    missing = testCat - trainCat
    if missing:
        nb_samples = 0
        for m in missing:
            nb_samples += test[test[cat] == m].shape[0]
        print("Feature: {}. Missing categories: {}. Number of samples: {}".format(cat, list(missing), nb_samples))


# Example: the 3 `E` and `G` values for the `cat92` will be replaced by `NaN` during encoding:

# In[ ]:


train = train.drop(["id", "loss"], axis=1)
test = test.drop(["id"], axis=1)
train = ft.fit_transform(train)
test = ft.fit_transform(test)


# In[ ]:


test[np.isnan(test.cat92)]


# In[ ]:


imp = Imputer(missing_values='NaN', strategy='median', axis=0)
imp.fit(train)
test = imp.transform(test)
test = pd.DataFrame(test)
test.columns = train.columns


# The `NaN` values have now been replaced by the median value for each category:

# In[ ]:


test[np.isnan(test.cat92)]


# In[ ]:


all_labels = set()
for cat in catFeatureslist:
    all_labels.update(train[cat].unique())
le = LabelEncoder()
le.fit(list(all_labels))
for cat in catFeatureslist:
    train[cat] = le.transform(train[cat])


# Alternative to the median value: use the correlation between categories

# In[ ]:


corr = train[catFeatureslist].corr()


# In[ ]:


sns.heatmap(corr)


# In[ ]:


for cat in catFeatureslist:
    corr_order = corr[cat].order()
    print("{} gets maximum correlation factor with {} (corr={:.2})".format(
        cat,
        corr_order.index[-2],
        corr_order[-2]
    ))

