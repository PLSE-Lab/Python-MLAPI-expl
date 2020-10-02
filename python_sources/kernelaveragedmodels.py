#!/usr/bin/env python
# coding: utf-8

# # Averaged Models Regression for Finish Times
# This Kernel follows a stacked regression approach in determining the finish times of the data.
# 
# **The following kernel is very helpful in starting machine learning with stacked regressions:**
# 
# https://www.kaggle.com/serigne/stacked-regressions-top-4-on-leaderboard
# 
# First we import the necessary modules for this script.

# In[ ]:


"""
Name: Thijme Langelaar
Collaborators: 
    https://www.kaggle.com/serigne/stacked-regressions-top-4-on-leaderboard
"""
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import ElasticNet, Lasso, Ridge
from sklearn.ensemble import RandomForestRegressor,  GradientBoostingRegressor, AdaBoostRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import mean_squared_error


# # Data Import
# The data is loaded into `df` and `df_test` with custom column names. 

# In[ ]:


column_names_df = ['Place', 'ID', 'Category', 'Finish Time', 'STBtransfer',
                   'B5KM', 'B37.5KM', 'BTRtransfer', 'R3.5KM', 'R4.5KM', 'R8KM']
column_names_df_test = ['Place', 'ID', 'Category', 'STBtransfer',
                        'B5KM', 'B37.5KM', 'BTRtransfer', 'R3.5KM', 'R4.5KM', 'R8KM']

df = pd.read_csv('../input/challenge_train.csv',
                 skiprows=1, names=column_names_df)
df_test = pd.read_csv('../input/challenge_predict_set.csv',
                      skiprows=1, names=column_names_df_test)


# Next up, I make a seperate One-hot column named 'PRO' which tells us whether or not a player is a pro.
# The 'MPRO' value for the Category column is replaced by `20`. Category can now be treated as an 'Age' variable and used as a feature.

# In[ ]:


df = df.set_index('ID')
df_test = df_test.set_index('ID')
df['PRO'] = 0
df_test['PRO'] = 0

column_names_df = ['Place', 'Category', 'Finish Time', 'STBtransfer', 'B5KM',
                'B37.5KM', 'BTRtransfer', 'R3.5KM', 'R4.5KM', 'R8KM', 'PRO']
column_names_df_test = ['Place', 'Category', 'STBtransfer', 'B5KM',
                        'B37.5KM', 'BTRtransfer', 'R3.5KM', 'R4.5KM', 'R8KM', 'PRO']

datasets = [df, df_test]
temp = 0
for dataset in datasets:
    temp += 1
    if temp == 1:
        # PRO or not?
        dataset['PRO'] = dataset[dataset['Category'] == 'MPRO'].Category
        dataset.PRO = dataset.PRO.fillna(0)
        dataset.PRO = dataset.PRO.replace('MPRO', 1)
    
        # Filling the missing ages with 5 years below the lowest age group
        dataset.Category = dataset.Category.replace('MPRO', 20)
        dataset.Category = dataset.Category.astype('int64')
        dataset.columns = column_names_df
    else:
        # PRO or not?
        dataset['PRO'] = dataset[dataset['Category'] == 'MPRO'].Category
        dataset.PRO = dataset.PRO.fillna(0)
        dataset.PRO = dataset.PRO.replace('MPRO', 1)
    
        # Filling the missing ages with 5 years below the lowest age group
        dataset.Category = dataset.Category.replace('MPRO', 20)
        dataset.Category = dataset.Category.astype('int64')
        dataset.columns = column_names_df_test


# # Exploratory Data Analysis
# A quick EDA is provided with a pairplot.

# In[ ]:


sns.pairplot(df, hue='PRO')
plt.show()


# As we had expected, split times have a (visually) strong correlation with `finish times`. This relation is stronger for the running split times then for the bike split times.
# 
# The one orange dot is a pro player that had a bad day. He skews the results on every feature. This becomes clear from the histograms; the second orange 'hump' is our outlier.
# 
# For now, we will leave him in the race.
# 
# 

# # Feature Engineering
# I'm interested if the difference in subsequent split times are potential features.
# 
# The difference in the splittimes `Bike 5KM` and `Bike 37.5KM` is called: `deltaBike`
# 
# The difference between the first two running times is called: `deltaRun1`
# 
# The difference between the last two running times is called: `deltaRun2`

# In[ ]:


for dataset in datasets:
    features = ['B5KM', 'B37.5KM', 'R3.5KM', 'R4.5KM', 'R8KM']
    target_features = ['deltaBike', 'deltaRun1', 'deltaRun2']
    target = 'Finish Time'
    dataset[target_features[0]] = dataset[features[1]]-dataset[features[0]]
    dataset[target_features[1]] = dataset[features[3]]-dataset[features[2]]
    dataset[target_features[2]] = dataset[features[4]]-dataset[features[3]]


# In[ ]:


for feature in target_features:
    sns.scatterplot(x=target, y=feature, hue='PRO', data=df)
    plt.show()


# Again we see some strong correlations going on.

# # Machine Learning
# First we choose our features and target variable and just drop the `NaN` for now.

# In[ ]:


features = ['Category', 'B5KM', 'B37.5KM',
            'BTRtransfer', 'R3.5KM', 'R4.5KM', 'R8KM',
            'deltaBike', 'deltaRun1', 'deltaRun2']
target = 'Finish Time'
df = df.dropna()
X = df[features]
X = StandardScaler().fit_transform(X)
y = df[target]


# Next up, we initiate the models we are going to use. For now, we use some standard values. Hyperparametrization could occur later on if we are not satisfied yet.

# In[ ]:


lasso = Lasso(alpha=0.05, max_iter=100000, random_state=1)
ENet = ElasticNet(alpha=0.05, l1_ratio=0.8,
                  random_state=3)
ridge = Ridge(alpha=0.0005)
GBR = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1,
                                max_depth=2, random_state=0, loss='ls')
RFG = RandomForestRegressor(max_depth=9, random_state=5, n_estimators=100)


# We'll make some functions to help us with all the models.

# In[ ]:


def model_pipeline(model, X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3,
                                                        random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return y_test, y_pred

def rmse_n(y_test, y_pred):
    return np.sqrt(mean_squared_error(y_test, y_pred))


# ## Let the testing begin!

# In[ ]:


y_test, y_lasso_pred = model_pipeline(lasso, X, y)
y_test, y_ENet_pred = model_pipeline(ENet, X, y)
y_test, y_Ridge_pred = model_pipeline(ridge, X, y)
y_test, y_GBR_pred = model_pipeline(GBR, X, y)
y_test, y_RFG_pred = model_pipeline(RFG, X, y)

RMSE_lasso = rmse_n(y_test, y_lasso_pred)
RMSE_enet = rmse_n(y_test, y_ENet_pred)
RMSE_ridge = rmse_n(y_test, y_Ridge_pred)
RMSE_GBR = rmse_n(y_test, y_GBR_pred)
RMSE_RFG = rmse_n(y_test, y_RFG_pred)

print('RMSE Lasso: {:.2f}'.format(RMSE_lasso))
print('RMSE ENet: {:.2f}'.format(RMSE_enet))
print('RMSE Ridge: {:.2f}'.format(RMSE_ridge))
print('RMSE GBR: {:.2f}'.format(RMSE_GBR))
print('RMSE RFG: {:.2f}'.format(RMSE_RFG))


# Not too bad! Lets see what averaging them will do.

# In[ ]:


y_stack = np.column_stack([y_lasso_pred, y_ENet_pred, y_Ridge_pred, y_GBR_pred,
                           y_RFG_pred])
y_predictions = np.mean(y_stack, axis=1)
RMSE_average = np.sqrt(mean_squared_error(y_test,y_predictions))

print('RMSE Average: {:.2f}'.format(RMSE_average))


# A pretty good increase in our RMSE. More importantly, it's nice that we have introduced some Democracy in our digital environment.
# 
# Now it would be nice to know what features are important. I wont be too exhaustive here, and only look at the lasso model.

# In[ ]:


lasso_coef = lasso.coef_

plt.plot(range(len(df[features].columns)), lasso_coef)
plt.xticks(range(len(df[features].columns)), df[features].columns.values, rotation=60)
plt.margins(0.02)
plt.plot(range(len(df[features].columns)),
         [0 for x in range(len(df[features].columns))],
         linestyle='--')
plt.show()

regeq = 'y = '
maxcoef = 0
for i in range(len(lasso_coef)):
    if lasso_coef[i] > maxcoef:
        maxcoef = lasso_coef[i]
        maxvar = features[i]
    else:
        maxcoef = maxcoef
    coefs_temp = str(round(lasso_coef[i],3)) + '*'
    if i == len(lasso_coef)-1:
        var_temp = features[i]
    else:
        var_temp = features[i] + ' + '
    regeq = regeq + coefs_temp + var_temp

print(regeq)
print('The most important feature is '+maxvar+' with a weight of {}'.format(round(maxcoef,3)))


# `R 3.5KM` is apparently super important with a weight that is twice as big as the second most important feature `deltaRun2`
# 
# More importantly however, is the huge contribution of `deltaRun2`! Apparently, if you can keep up your pace in the early stages of the marathon, you have a good shot at acquiring a high end position.
# 
# Another observation we can make is that people that have a good split time on `Bike 5KM` probably started of too fast.
# 
# The last two running times add nothing to our prediction!
# We can safely remove those features next time around and reduce our processing time.
