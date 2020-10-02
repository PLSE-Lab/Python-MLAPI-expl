#!/usr/bin/env python
# coding: utf-8

# ## Import libraries

# In[ ]:


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from matplotlib import pyplot

get_ipython().run_line_magic('matplotlib', 'inline')

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score

from xgboost import XGBRegressor


# ## Import data

# In[ ]:


train = pd.read_csv("../input/train.csv")
test = pd.read_csv("../input/test.csv")


# In[ ]:


train.shape


# In[ ]:


train.describe()


# In[ ]:


train.drop(['Id'], axis=1, inplace=True)


# ## Missing data

# In[ ]:


def show_missing_data(df):
  df_nan = df.isnull().sum()[df.isnull().sum()>0].sort_values(ascending = False)
  df_nan_per = df_nan / df.shape[0] * 100

  print(pd.concat([df_nan, df_nan_per], 
                  axis=1, 
                  keys=['nan Amount', 'Percentage']))


# In[ ]:


train_tmp = train.drop(['SalePrice'], axis=1)
test_ids = test.Id
test = test.drop(['Id'], axis=1)

total = pd.concat([train_tmp, test]).reset_index(drop=True)

print(show_missing_data(total))


# In[ ]:


fig, axs = plt.subplots(1, 2, figsize=(15,5))

sns.countplot(total[pd.notna(total.PoolQC)].PoolQC, ax=axs[0])
sns.countplot(total.PoolArea, ax=axs[1])

plt.suptitle("Pool Quality vs Pool's Area")
axs[0].set_xlabel("Quality")
axs[1].set_xlabel("Area")


# By these graphs we can conclude that ''NA' values are not set. All the null values mean that basically the house has no pool. I'll proceed by filling all these values with 'NA' as the original description described.

# In[ ]:


total.PoolQC = total.PoolQC.fillna('NA')
train.PoolQC = train.PoolQC.fillna('NA')
test.PoolQC = test.PoolQC.fillna('NA')


# In[ ]:


sns.countplot(total.PoolQC)


# This seems to be a constant, so I'll do the same with the rest of the columns that allow 'NA' values

# In[ ]:


def fillNAValues(na_list):
  for elem in na_list:
    total[elem] = total[elem].fillna('NA')
    train[elem] = train[elem].fillna('NA')
    test[elem] = test[elem].fillna('NA')


# In[ ]:


na_list = ['MiscFeature', 'Alley', 'Fence', 'GarageFinish', 'GarageQual', 
           'GarageCond', 'GarageType', 'BsmtQual', 'BsmtCond', 
           'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2', 'FireplaceQu']


# In[ ]:


fillNAValues(na_list)


# For the LotFrontage I'm going to analyze the LotArea

# In[ ]:


total_lot = (total[(pd.notna(total.LotFrontage)) 
            & (total.LotFrontage < 200) 
            & (total.LotArea < 100000)]
          [['LotFrontage','LotArea']])

total_lot.plot.scatter(x='LotFrontage', y='LotArea')


# So it follows a linear correlation. I'll execute a linear regression to replace the null values

# In[ ]:


regressor = LinearRegression()
regressor.fit(total_lot.LotArea.to_frame(), total_lot.LotFrontage)

lot_nan_total = total[pd.isnull(total.LotFrontage)].LotArea
lot_nan_train = train[pd.isnull(train.LotFrontage)].LotArea
lot_nan_test = test[pd.isnull(test.LotFrontage)].LotArea

lot_pred_total = regressor.predict(lot_nan_total.to_frame())
lot_pred_train = regressor.predict(lot_nan_train.to_frame())
lot_pred_test = regressor.predict(lot_nan_test.to_frame())

total.loc[total.LotFrontage.isnull(), 'LotFrontage'] = lot_pred_total
train.loc[train.LotFrontage.isnull(), 'LotFrontage'] = lot_pred_train
test.loc[test.LotFrontage.isnull(), 'LotFrontage'] = lot_pred_test


# In[ ]:


total_lot = (total[(pd.notna(total.LotFrontage)) 
            & (total.LotFrontage < 200) 
            & (total.LotArea < 100000)]
          [['LotFrontage','LotArea']])

total_lot.plot.scatter(x='LotFrontage', y='LotArea')


# With the GarageYrBlt, my first assumption is that it must be strongly correlated with the YearBuilt. So I'll use a scatterplot to check this out.

# In[ ]:


total.plot.scatter(x='YearBuilt', y='GarageYrBlt')


# In[ ]:


total.YearBuilt.corr(total.GarageYrBlt)


# It seems reasonable that in almost every sample the garage is either built the same year as the house or some years after. But just to dig a little bit further, I'll check the percentage of times that they are built on the same year. If the pecentage is high enough, I'll proceed by replacing the null values for the YearBuilt values.

# In[ ]:


total[total.YearBuilt==total.GarageYrBlt].count().YearBuilt.astype('float')/total.shape[0]


# In[ ]:


total.GarageYrBlt = total.GarageYrBlt.fillna(total.YearBuilt)
train.GarageYrBlt = train.GarageYrBlt.fillna(train.YearBuilt)
test.GarageYrBlt = test.GarageYrBlt.fillna(test.YearBuilt)


# And that GarageYrBlt is clearly an outlyer. So let's apply the same logic to it.

# In[ ]:


total.loc[total.GarageYrBlt>2100, 'GarageYrBlt'] = total.YearBuilt
train.loc[train.GarageYrBlt>2100, 'GarageYrBlt'] = train.YearBuilt
test.loc[test.GarageYrBlt>2100, 'GarageYrBlt'] = test.YearBuilt


# In[ ]:


def replace_with_mode(dfs, cols):
  for df in dfs:
    for col in cols:
      df[col] = df[col].fillna(df[col].mode()[0])


# In[ ]:


dfs = [total, train, test]
na_values = ['Electrical', 'Functional', 'Utilities', 
                'Exterior2nd', 'Exterior1st', 'KitchenQual',
                'SaleType', 'MSZoning', 'MasVnrType', 'BsmtHalfBath',
                'BsmtFullBath', 'TotalBsmtSF', 'BsmtUnfSF', 
                'BsmtFinSF2', 'BsmtFinSF1']

replace_with_mode(dfs, na_values)


# Now it is time to deal with MasVnrArea. This value is the masonry veneer.

# In[ ]:


total.loc[total.MasVnrArea.isnull(),['MasVnrArea','MasVnrType']]


# This means that null values were the once I've just replaced for the mode (None). So I'll turn them into 0

# In[ ]:


total.MasVnrArea = total.MasVnrArea.fillna(0)
train.MasVnrArea = train.MasVnrArea.fillna(0)
test.MasVnrArea = test.MasVnrArea.fillna(0)


# In[ ]:


print(show_missing_data(total))


# Let's see what's the issue with the garage null values

# In[ ]:


total.loc[total.GarageArea.isnull(),['GarageFinish', 'GarageCars', 'GarageArea']]


# This means there's no garage. So the rest of the values should be 0

# In[ ]:


total.GarageCars = total.GarageCars.fillna(0)
total.GarageArea = total.GarageArea.fillna(0)

train.GarageCars = train.GarageCars.fillna(0)
train.GarageArea = train.GarageArea.fillna(0)

test.GarageCars = test.GarageCars.fillna(0)
test.GarageArea = test.GarageArea.fillna(0)


# Finally, we take care of BsmtHalfBath and BsmtFullBath 

# In[ ]:


total.groupby('BsmtHalfBath').BsmtHalfBath.count()


# In[ ]:


total.BsmtHalfBath = total.BsmtHalfBath.fillna(0)
total.BsmtFullBath = total.BsmtFullBath.fillna(0)

train.BsmtHalfBath = train.BsmtHalfBath.fillna(0)
train.BsmtFullBath = train.BsmtFullBath.fillna(0)

test.BsmtHalfBath = test.BsmtHalfBath.fillna(0)
test.BsmtFullBath = test.BsmtFullBath.fillna(0)


# In[ ]:


show_missing_data(total)
show_missing_data(train)
show_missing_data(test)


# We are done with missing values!

# ## Data conversion

# In[ ]:


train.SalePrice = np.log(train.SalePrice)


# In[ ]:


final_total = pd.get_dummies(total).reset_index(drop=True)

final_total.shape

y = train.SalePrice
X = final_total.iloc[:len(y),:]

test = final_total.iloc[len(y):,:]

print(X.shape)
print(test.shape)


# In[ ]:


fig , ax = plt.subplots(figsize = (10, 5))

sns.boxplot(X.OverallQual, y)


# ## Scikit-learn Linear Regression

# In[ ]:


model_reg = LinearRegression()
model_reg.fit(X,y)

accuracies = cross_val_score(estimator=model_reg, X=X, y=y, cv=10)
print(accuracies.mean())


# ## XGBoost Regression

# In[ ]:


def print_cv_params(selecter_param, selecter_param_str, parameters):
  
  grid_search = GridSearchCV(estimator = model_xgb,
                            param_grid = parameters,
                            scoring = 'neg_mean_squared_error',
                            cv = 10,
                            n_jobs = -1)

  grid_result = grid_search.fit(X, y)

  print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
  means = grid_result.cv_results_['mean_test_score']
  stds = grid_result.cv_results_['std_test_score']
  params = grid_result.cv_results_['params']
  for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))

  pyplot.errorbar(selecter_param, means, yerr=stds)
  pyplot.title("XGBoost "+ selecter_param_str + " vs Mean Squared Error")
  pyplot.xlabel(selecter_param_str)
  pyplot.ylabel('Mean Squared Error')


# In[ ]:


model_xgb = XGBRegressor()


# In[ ]:


n_estimators = range(50, 800, 150)
parameters = dict(n_estimators=n_estimators)

print_cv_params(n_estimators, 'n_estimators', parameters)


# In[ ]:


learning_rate = np.arange(0.0, 0.2, 0.03)
parameters = dict(learning_rate=learning_rate)

print_cv_params(learning_rate, 'learning_rate', parameters)


# In[ ]:


max_depth = range(0, 7)
parameters = dict(max_depth=max_depth)

print_cv_params(max_depth, 'max_depth', parameters)


# In[ ]:


min_child_weight = np.arange(0.5, 2., 0.3)
parameters = dict(min_child_weight=min_child_weight)

print_cv_params(min_child_weight, 'min_child_weight', parameters)


# In[ ]:


gamma = np.arange(.001, .01, .003)
parameters = dict(gamma=gamma)

print_cv_params(gamma, 'gamma', parameters)


# In[ ]:


subsample = np.arange(0.3, 1., 0.2)
parameters = dict(subsample=subsample)

print_cv_params(subsample, 'subsample', parameters)


# In[ ]:


colsample_bytree = np.arange(.6, 1, .1)
parameters = dict(colsample_bytree=colsample_bytree)

print_cv_params(colsample_bytree, 'colsample_bytree', parameters)


# In[ ]:


parameters = {  
                'colsample_bytree':[.6],
                'subsample':[.9,1],
                'gamma':[.004],
                'min_child_weight':[1.1,1.3],
                'max_depth':[3,6],
                'learning_rate':[.15,.2],
                'n_estimators':[1000],                                                                    
                'reg_alpha':[0.75],
                'reg_lambda':[0.45],
                'seed':[42]
}

grid_search = GridSearchCV(estimator = model_xgb,
                        param_grid = parameters,
                        scoring = 'neg_mean_squared_error',
                        cv = 5,
                        n_jobs = -1)

model_xgb = grid_search.fit(X, y)
best_score = grid_search.best_score_
best_parameters = grid_search.best_params_


# In[ ]:


best_score


# In[ ]:


best_parameters


# In[ ]:


accuracies = cross_val_score(estimator=model_xgb, X=X, y=y, cv=10)
accuracies.mean()


# In[ ]:


y_pred = model_xgb.predict(test)
y_pred = np.floor(np.expm1(y_pred))


# In[ ]:


submission = pd.concat([test_ids, pd.Series(y_pred)], 
                        axis=1,
                        keys=['Id','SalePrice'])


# In[ ]:


submission.to_csv('sample_submission.csv', index = False)


# In[ ]:


submission

