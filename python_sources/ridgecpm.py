#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectPercentile, f_regression
from sklearn.linear_model import RidgeCV, Ridge
from sklearn.model_selection import cross_val_predict, cross_val_score, GridSearchCV, KFold


# In[ ]:


def metric(y_true, y_pred):
    return np.mean(np.sum(np.abs(y_true - y_pred), axis=0)/np.sum(y_true, axis=0))


# # read data

# In[ ]:


bp = '/kaggle/input/trends-assessment-prediction'

loading_data = pd.read_csv(bp+'/loading.csv')
fnc_data = pd.read_csv(bp+'/fnc.csv')

train_scores = pd.read_csv(bp+'/train_scores.csv')
train_scores["is_train"] = True

df = fnc_data.merge(loading_data, on='Id')
df = df.merge(train_scores, on="Id", how="left")

test_df = df[df["is_train"] != True].copy()
df = df[df["is_train"] == True].copy()

fnc_features, loading_features = list(fnc_data.columns[1:]), list(
    loading_data.columns[1:])

features = loading_features + fnc_features

print('Size of train data', df.shape)
print('Size of test data', test_df.shape)


# # Select model

# In[ ]:


# alphas = 10**np.linspace(100,-4,20)*0.5 # specify alphas to search
# ridgeCV = RidgeCV(cv=5, alphas=alphas, scoring='neg_mean_absolute_error')
# score = []
# for p in [10, 30, 50, 80]:
#     feature_selection = SelectPercentile(f_regression, percentile=p)
#     score_temp = []
#     for target, c, w in [("age", 100, 0.3), ("domain1_var1", 10, 0.175), ("domain1_var2", 10, 0.175), 
#                          ("domain2_var1", 10, 0.175), ("domain2_var2", 10, 0.175)]:    
#         model = Pipeline([('univariate feature selection', feature_selection), ('ridge', ridgeCV)])

#         train_df = df[df[target].notnull()]

#         score_temp.append(np.mean(cross_val_score(model, train_df[features], train_df[target], scoring='neg_mean_absolute_error')))
#     score.append(score_temp)


# In[ ]:


# alphas = 10**np.linspace(100,-4,20)*0.5 # specify alphas to search
# ridge = RidgeCV(cv=5, alphas=alphas, scoring='neg_mean_absolute_error')
# feature_selection = SelectPercentile(f_regression, percentile=10)
# estimator = Pipeline([('univariate feature selection', feature_selection), 
#                                ('ridge', ridge)])

# for target, c, w in [("age", 100, 0.3), ("domain1_var1", 10, 0.175), ("domain1_var2", 10, 0.175), 
#                      ("domain2_var1", 10, 0.175), ("domain2_var2", 10, 0.175)]:
# #     y_test = np.zeros((test_df.shape[0], 1))

#     train_df = df[df[target].notnull()]
#     model.fit(train_df[features], train_df[target])
#     test_df[target] = model.predict(test_df[features])


# In[ ]:


# alphas = 10**np.linspace(100,-5,80)*0.5 # specify alphas to search
# ps = [10, 30, 50, 70, 90, 100]

# ridge = Ridge(normalize=True)
# feature_selection = SelectPercentile(f_regression)
# param_grid = {'fs__percentile': ps, 'ridge__alpha': alphas}
# estimator = Pipeline([('fs', feature_selection), ('ridge', ridge)])

# model = GridSearchCV(estimator, param_grid, n_jobs=-1, scoring='neg_mean_absolute_error')

# for target, c, w in [("age", 100, 0.3), ("domain1_var1", 10, 0.175), ("domain1_var2", 10, 0.175), 
#                      ("domain2_var1", 10, 0.175), ("domain2_var2", 10, 0.175)]:
# #     y_test = np.zeros((test_df.shape[0], 1))

#     train_df = df[df[target].notnull()]
#     model.fit(train_df[features], train_df[target])
#     print(model.cv_results_)
#     test_df[target] = model.predict(test_df[features])


# In[ ]:


# %%time

NUM_FOLDS = 8
kf = KFold(n_splits=NUM_FOLDS, shuffle=True, random_state=0)

overal_score = 0

# model
alphas = 10**np.linspace(100,-5,80)*0.5 # specify alphas to search
ps = [50, 80, 100]
ridge = Ridge(normalize=True)
feature_selection = SelectPercentile(f_regression)
param_grid = {'fs__percentile': ps, 'ridge__alpha': alphas}
estimator = Pipeline([('fs', feature_selection), ('ridge', ridge)])
model = GridSearchCV(estimator, param_grid, n_jobs=-1, scoring='neg_mean_absolute_error')

for target, c, w in [("age", 100, 0.3), ("domain1_var1", 10, 0.175), ("domain1_var2", 10, 0.175), ("domain2_var1", 10, 0.175), ("domain2_var2", 10, 0.175)]:    
    y_oof = np.zeros(df.shape[0])
    y_test = np.zeros((test_df.shape[0], NUM_FOLDS))
    
    for f, (train_ind, val_ind) in enumerate(kf.split(df, df)):
        train_df, val_df = df.iloc[train_ind], df.iloc[val_ind]
        train_df = train_df[train_df[target].notnull()]

        model.fit(train_df[features], train_df[target])

        y_oof[val_ind] = model.predict(val_df[features])
        y_test[:, f] = model.predict(test_df[features])
        
    df["pred_{}".format(target)] = y_oof
    test_df[target] = y_test.mean(axis=1)
    
    score = metric(df[df[target].notnull()][target].values, df[df[target].notnull()]["pred_{}".format(target)].values)
    overal_score += w*score
    print(target, np.round(score, 4))
    print()
    
print("Overal score:", np.round(overal_score, 4))


# In[ ]:


sub_df = pd.melt(test_df[["Id", "age", "domain1_var1", "domain1_var2", "domain2_var1", "domain2_var2"]], 
                 id_vars=["Id"], value_name="Predicted")
sub_df["Id"] = sub_df["Id"].astype("str") + "_" +  sub_df["variable"].astype("str")

sub_df = sub_df.drop("variable", axis=1).sort_values("Id")
assert sub_df.shape[0] == test_df.shape[0]*5
sub_df.head(10)


# In[ ]:


sub_df.to_csv("submission_mean_8fold.csv", index=False)

