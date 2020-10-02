#!/usr/bin/env python
# coding: utf-8

# # Comparing Feature Importance, Coefficient of Determination, and Linear Model Weights
# 
# This kernel compares a number of different methods to identfy the most important features of the [house prices dataset](https://www.kaggle.com/c/house-prices-advanced-regression-techniques). The accuracy of different regression models is compared for models trained on all available features and a model trained on only the $n$ most important features.
# 
# I created this kernel out of curiosity while working on the [house prices dataset](https://www.kaggle.com/c/house-prices-advanced-regression-techniques). My main motivation was to figure out (1) how well rankings of feature importances based on different methods correlate with each other and (2) to quantify how leaving out less important features affects the model accuracy.
# 
# This kernel is based on the data preparations from [House Prices with simple Ridge Regression](https://www.kaggle.com/mommermi/house-prices-with-simple-ridge-regression).

# ## Preparation of Data

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

data = pd.read_csv('../input/train.csv')

# replace prices with logarithmic prices
data['logSalePrice'] = np.log10(data.SalePrice.values)
data.drop('SalePrice', axis=1, inplace=True)


# ### Feature Types
# 
# Identify the different types of features in the data set.

# In[ ]:


# automatically identify continuous features
continuous_features = [col for col in data.columns if data[col].dtype != 'O']

continuous_features.remove('Id')
continuous_features.remove('logSalePrice') # remove the target feature

# manually select ordinal features that can be ranked in some way
ordinal_features = ['Street', 'Alley', 'LotShape', 'Utilities', 'LandSlope', 'ExterQual', 
                    'LandContour', 'ExterCond', 'BsmtQual', 'BsmtCond', 'BsmtExposure', 
                    'BsmtFinType1', 'BsmtFinType2', 'HeatingQC', 'CentralAir', 
                    'Electrical', 'KitchenQual', 'Functional', 'FireplaceQu', 'GarageType',
                    'GarageFinish', 'GarageQual', 'GarageCond', 'PavedDrive', 'PoolQC', 
                    'Fence'] 

# manually select categorical features that will be ranked based on median `logSalePrice`
categorical_features = ['LotConfig', 'Neighborhood', 'Condition1', 'Condition2', 
                        'HouseStyle', 'RoofStyle', 'RoofMatl', 'Exterior1st', 
                        'Exterior2nd', 'MasVnrType', 'Foundation', 'Heating', 'MiscFeature',
                        'SaleType', 'SaleCondition', 'MSZoning', 'BldgType']


# ### Feature Completeness

# Fill those ordinal/categorical features with `NA` for which this value is defined.

# In[ ]:


for col in ['Alley', 'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2', 
            'FireplaceQu', 'GarageType', 'GarageFinish', 'GarageQual', 'GarageCond', 
            'PoolQC', 'Fence', 'MiscFeature']:
    data.loc[:, col].fillna('NA', inplace=True) 


# For continuous features, fill in the median across the entire feature; for other ordinal/categorical features, fill in the most common value across each feature.

# In[ ]:


data.fillna({col: data.loc[:, col].median() for col in continuous_features}, inplace=True)
data.fillna({col: data.loc[:, col].value_counts().index[0] for 
             col in categorical_features + ordinal_features}, inplace=True)


# ### Transforming Features
# 
# The following dictionary describes the ranking of those features that were previously classified as ordinal features from lowest rank (worst case, starting at `1` so that missing data can be ranked with zero where applicable) to highest rank (best case): 

# In[ ]:


ordinal_transform = {'Street':  {'Grvl': 1, 'Pave': 2}, 
                     'Alley': {'NA': 0, 'Grvl': 1, 'Pave': 2}, 
                     'LotShape': {'IR3': 1, 'IR2': 2, 'IR1': 3, 'Reg': 4}, 
                     'Utilities': {'ELO': 1, 'NoSeWa': 2, 'NoSewr': 3, 'AllPub': 4}, 
                     'LandSlope': {'Sev': 1, 'Mod': 2, 'Gtl': 3}, 
                     'LandContour': {'Low': 1, 'HLS': 1, 'Bnk': 2, 'Lvl': 3}, 
                     'ExterQual': {'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5},
                     'ExterCond': {'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5}, 
                     'BsmtQual': {'NA': 0, 'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5}, 
                     'BsmtCond': {'NA': 0, 'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5}, 
                     'BsmtExposure': {'NA': 0, 'No': 1, 'Mn': 2, 'Av': 3, 'Gd': 4}, 
                     'BsmtFinType1': {'NA': 0, 'Unf': 1, 'LwQ': 2, 'Rec': 3, 'BLQ': 4, 'ALQ': 5, 
                                      'GLQ': 6}, 
                     'BsmtFinType2': {'NA': 0, 'Unf': 1, 'LwQ': 2, 'Rec': 3, 'BLQ': 4, 'ALQ': 5, 
                                      'GLQ': 6}, 
                     'HeatingQC': {'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5},
                     'CentralAir': {'N': 1, 'Y': 2}, 
                     'Electrical': {'Mix': 1, 'FuseP': 2, 'FuseF': 3, 'FuseA': 4, 'SBrkr': 5}, 
                     'KitchenQual': {'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5}, 
                     'Functional': {'Sal': 1, 'Sev': 2, 'Maj2': 3, 'Maj1': 4, 'Mod': 5, 'Min2': 6, 
                                    'Min1': 7, 'Typ': 8}, 
                     'FireplaceQu': {'NA': 0, 'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5}, 
                     'GarageType': {'NA': 0, 'Detchd': 1, 'CarPort': 2, 'BuiltIn': 3, 
                                    'Basment': 4, 'Attchd': 5, '2Types': 6},
                     'GarageFinish': {'NA': 0, 'Unf': 1, 'RFn': 2, 'Fin': 3}, 
                     'GarageQual': {'NA': 0, 'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5}, 
                     'GarageCond': {'NA': 0, 'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5}, 
                     'PavedDrive': {'N': 1, 'P': 2, 'Y': 3}, 
                     'PoolQC': {'NA': 0, 'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5}, 
                     'Fence': {'NA': 0, 'MnWw': 1, 'GdWo': 2, 'MnPrv': 3, 'GdPrv': 4}, 
                    }

# apply transformations
for col in ordinal_features:
    data.loc[:, col] = data.loc[:, col].map(ordinal_transform[col], na_action='ignore')
    
# move some features from continuous to ordinal feature list
for col in ['BsmtFullBath', 'BsmtHalfBath', 'FullBath', 'HalfBath', 'BedroomAbvGr', 'KitchenAbvGr', 'TotRmsAbvGrd',
            'Fireplaces', 'GarageCars', 'MoSold', 'YrSold']:
    continuous_features.remove(col)
    ordinal_features.append(col)
    
# move one feature from continuous to categorial feature list
continuous_features.remove('MSSubClass')
categorical_features.append('MSSubClass')


# In the case of categorical features, we rank each feature based on its median `logSalePrice`.

# In[ ]:


ordinalized = []
for col in categorical_features:
    sorted_labels = data[[col, 'logSalePrice']].groupby(col).logSalePrice.median().sort_values()
    data.loc[:, col] = data.loc[:, col].map({sorted_labels.index.values[i]:i for i in range(len(sorted_labels))})
    ordinalized.append(col)

for col in ordinalized:
    categorical_features.remove(col)
    ordinal_features.append(col)


# Now we have a complete data set consisting of purely numerical data.

# ## Feature Importance Ranking
# 
# In the following, we apply different methods for estimating feature importance. 

# ### Coefficient of Determination
# 
# We derive the [coefficient of determination](https://en.wikipedia.org/wiki/Coefficient_of_determination), $r^2$, which is based on [Pearson's correlation coefficient](https://en.wikipedia.org/wiki/Pearson_correlation_coefficient) $r$, for each feature with respect to `logSalePrice`. The higher $r^2$, the higher the correlation between the two features. 
# 
# The coefficient of determination is typically considered a measure for the variance in a relation between two features that is [explained by the model](https://en.wikipedia.org/wiki/Coefficient_of_determination#Relation_to_unexplained_variance) and and does not rely on any machine learning model.

# In[ ]:


f, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 8), sharey=True)
plt.subplots_adjust(hspace=0.5)

# continuous features
data[continuous_features].corrwith(data.logSalePrice).agg('square').plot.bar(ax=ax1, alpha=0.5)
ax1.set_title('Coefficient of Determination: Continuous Features')
ax1.grid()

# ordinal features
data[ordinal_features].corrwith(data.logSalePrice).agg('square').plot.bar(ax=ax2, alpha=0.5)
ax2.set_title('Coefficient of Determination: Ordinal Features')
ax2.grid()


# Few features stick out with high coefficients of determination.
# 
# We list those ten features with the highest coefficients:

# In[ ]:


results = pd.DataFrame(data.drop('logSalePrice', axis=1).corrwith(data.logSalePrice).agg('square'), 
                       columns=['det_weight'])

ranks = np.zeros(len(results), dtype=np.int)
for i, j in enumerate(np.argsort(results.det_weight)[::-1]):
    ranks[j] = i
results['det_rank'] = ranks

results.sort_values('det_rank').loc[:, ['det_rank', 'det_weight']].iloc[0:10]


# ### Random Forest Feature Importance
# 
# [Decision Tree](https://en.wikipedia.org/wiki/Decision_tree_learning)-based models provide information on the relative importance of each feature in the training data set. This *feature importance* is a diagnostic tool to identify features that carry crucial information.
# 
# Feature importance is only available for Decision Tree-based model. In this case, we use a [Random Forest](https://en.wikipedia.org/wiki/Random_forest) model in combination with a grid search and cross validation in order to obtain a meaningful fit to the data and hence meaningful feature importances.

# In[ ]:


from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV

pipe = Pipeline([('model', RandomForestRegressor(n_jobs=-1, random_state=42))])

param_space = {'model__max_depth': [10, 15, 20],
               'model__max_features': [10, 15, 20],
               'model__n_estimators': [250, 300, 350]}

grid = GridSearchCV(pipe, param_grid=param_space, cv=10, scoring='neg_mean_squared_error')

grid.fit(data.drop('logSalePrice', axis=1), data.logSalePrice)


# In[ ]:


print('best-fit parameters:', grid.best_params_)
print('prediction rms residuals:', 10**(np.sqrt(-grid.best_score_)))


# We achieve a decent model accuracy. What are the most important features?

# In[ ]:


weights = grid.best_estimator_.steps[0][1].feature_importances_

results['rf_weight'] = weights

ranks = np.zeros(len(results), dtype=np.int)
for i, j in enumerate(np.argsort(weights)[::-1]):
    ranks[j] = i
results['rf_rank'] = ranks

results.sort_values('rf_rank').loc[:, ['rf_rank', 'rf_weight']].iloc[0:10]


# ### Linear Model Coefficient Weights
# 
# In the following, we will fit two linear regression models to the entire training set and extract the weights of the individual features from these models a measure for feature importance.

# #### Ridge Regression
# 
# [Ridge Regression](https://en.wikipedia.org/wiki/Tikhonov_regularization) is a linear regression model implementation using L2 regularization. Like in any other linear model, the weights derived for the different features during training are indicative of their relative importance.
# 
# Using a gridsearch and cross-validation, we find the best-fit parameter $\alpha$ for the L2 regularization:

# In[ ]:


from sklearn.preprocessing import RobustScaler
from sklearn.linear_model import Ridge

pipe = Pipeline([('scaler', RobustScaler()), 
                 ('model', Ridge(random_state=42))])

param_space = {'model__alpha': [0.01, 0.1, 1, 10, 100, 1000]}

grid = GridSearchCV(pipe, param_grid=param_space, cv=10, scoring='neg_mean_squared_error')

grid.fit(data.drop('logSalePrice', axis=1), data.logSalePrice)


# In[ ]:


print('best-fit parameters:', grid.best_params_)
print('prediction rms residuals:', 10**(np.sqrt(-grid.best_score_)))


# The best-fit model accuracy is very similar to that obtained by the Random Forest model above. The ten most important features are:

# In[ ]:


weights = grid.best_estimator_.steps[1][1].coef_

results['ridge_weight'] = weights

ranks = np.zeros(len(results), dtype=np.int)
for i, j in enumerate(np.argsort(weights)[::-1]):
    ranks[j] = i
results['ridge_rank'] = ranks

results.sort_values('ridge_rank').loc[:, ['ridge_rank', 'ridge_weight']].iloc[0:10]


# #### Lasso
# 
# The [Lasso](https://en.wikipedia.org/wiki/Lasso_(statistics&#41;) is a linear model utilizing L1 regularization. 
# 
# We take an approach similar to the Ridge Regression model above.

# In[ ]:


from sklearn.linear_model import Lasso

pipe = Pipeline([('scaler', RobustScaler()), 
                 ('model', Lasso(random_state=42))])

param_space = {'model__alpha': [0.001, 0.01, 0.1, 1, 10, 100, 1000]}

grid = GridSearchCV(pipe, param_grid=param_space, cv=10, scoring='neg_mean_squared_error')

grid.fit(data.drop('logSalePrice', axis=1), data.logSalePrice)


# In[ ]:


print('best-fit parameters:', grid.best_params_)
print('prediction rms residuals:', 10**(np.sqrt(-grid.best_score_)))


# Given the value of the regularization parameter $\alpha$, the model is likely to over-fit the training data. Nevertheless, the accuracy of the model is very similar to the models used above. What are the ten most important features?

# In[ ]:


weights = grid.best_estimator_.steps[1][1].coef_

results['lasso_weight'] = weights

ranks = np.zeros(len(results), dtype=np.int)
for i, j in enumerate(np.argsort(weights)[::-1]):
    ranks[j] = i
results['lasso_rank'] = ranks

results.sort_values('lasso_rank').loc[:, ['lasso_rank', 'lasso_weight']].iloc[0:10]


# ## Comparison of Results
# 
# Each of the methods used above allow us to rank the features based on their weights or importances (we will refer to them simply as *weights* in the remainder of this kernel).
# 
# The following plot shows the *mean rank* of each feature, which is defined as the average rank based across the four methods used above, sorted from the most important feature (highest rank) to the least important feature (lowest rank).

# In[ ]:


f, ax = plt.subplots(figsize=(17, 5))

results['mean_rank'] = results.loc[:, ['det_rank', 'rf_rank', 'ridge_rank', 'lasso_rank']].agg('abs').mean(axis=1)
results['mean_rank_std'] = results.loc[:, ['det_rank', 'rf_rank', 'ridge_rank', 'lasso_rank']].std(axis=1)

results.mean_rank.sort_values().plot.bar(width=0.5, color='orange', alpha=0.7, ax=ax)
results.sort_values(by='mean_rank').mean_rank_std.plot.bar(width=0.5, color='black', alpha=0.3, ax=ax)

ax.set_xlim([0,79.5])
ax.plot(ax.get_xlim(), [0, 80], color='red')
ax.set_ylabel('Average Rank')


# Orange bars indicate the mean rank - grey bars indicate the standard deviation of the mean rank across the four methods. The red line symbolizes ideal behavior and unity line: the rank increases by one for each additional feature.
# 
# The mean rank distribution follows the ideal behavior only for the three highest ranked features (`OverallQual`, `Neighborhood`, and `GrLivArea`). For lower ranked features, the mean rank can be above or below the red line, symbolizing variation in the ranking. This behavior is supported by the standard deviations plotted here, and the top feature lists shown above for the individual methods.
# 
# The following plot visualizes the same data in a different way:

# In[ ]:


from matplotlib import ticker

f, ax = plt.subplots(figsize=(5, 17))

for idx, feature in enumerate(results.sort_values(by='mean_rank').index):
    ax.plot(range(4), results.loc[feature, ['det_rank', 'rf_rank', 'ridge_rank', 'lasso_rank']]+0.5, marker='o',
           alpha=np.clip(1/(results.loc[feature, 'mean_rank_std']+0.001), 0.01, 1),
           linewidth=np.clip(1/(results.loc[feature, 'mean_rank_std']+0.1), 0.5, 1)*3)
ax.set_ylim([80, 0])
ax.yaxis.set_major_locator(ticker.LinearLocator(numticks=80))
ax.set_yticklabels(results.sort_values(by='mean_rank').index)
ax.xaxis.set_major_locator(ticker.LinearLocator(numticks=4))
ax.set_xticklabels(['det_rank', 'rf_rank', 'ridge_rank', 'lasso_rank'])


# All features are listed based on their mean ranks from top (highest ranking) to bottom (lowest ranking). The abscissa shows the ranks based on the different methods. Each feature creates a line in this plot following the ranks through the different methods. 
# 
# In cases in which the ranking across the methods is very similar, the plotted line should be more or less horizontal. If the ranking is subject to significant variance, the line has a slope or follows a zig-zag course. This information is coded into the lines: consistently ranked features have thick and solid lines, whereas inconsistently ranked features have thin and transparent lines.
# 
# Similar to the previous plot, three features stand out as important: `OverallQual`, `Neighborhood`, and `GrLivArea`. These feature have consistently high ranks across all four methods.

# Finally, we consider the derived weights instead of the ranks. In order to enable a comparison between the individual weights, we normalize the sum of all weights derived from each method to unity:

# In[ ]:


for col in ['det_weight', 'rf_weight', 'ridge_weight', 'lasso_weight']:
    weightsum = results.loc[:, col].abs().sum(axis=0)
    results.loc[:, col] = results.loc[:, col].apply(lambda x: np.abs(x)/weightsum)


# Now we can plot a cumulative distribution of the weights derived from the different methods:

# In[ ]:


f, ax = plt.subplots(figsize=(17, 5))
ax.xaxis.set_major_locator(ticker.LinearLocator(numticks=80))
results.sort_values('mean_rank').loc[:, ['det_weight', 'rf_weight', 'ridge_weight', 'lasso_weight']].cumsum().plot.line(
    rot='vertical', ax=ax)
ax.grid()


# Interestingly, all methods that are based on linear models (including the coefficient of determination, which is measuring the linear correlation between two features) follow a similar course, whereas the Random Forest cumulative weights follow a significantly steeper curve. 
# 
# The three highest ranking features based on all four methods (`OverallQual`, `Neighborhood`, and `GrLivArea`) make up ~45% of the Random Forest cumulative weights, but only ~10% of the Ridge Regression cumulative weights.
# 
# Let's see how these numers impact the modeling accuracy.

# ## Impact of Feature Selection on Modeling Accuracy

# We train the Ridge Regression model only on the three higest ranking features (`OverallQual`, `Neighborhood`, and `GrLivArea`) and compare the result to the model trained on all available features:

# In[ ]:


pipe = Pipeline([('scaler', RobustScaler()), 
                 ('model', Ridge(random_state=42))])

param_space = {'model__alpha': [1, 5, 10, 50, 100]}

grid = GridSearchCV(pipe, param_grid=param_space, cv=10, scoring='neg_mean_squared_error')

grid.fit(data.drop('logSalePrice', axis=1).loc[:, ['OverallQual', 'Neighborhood', 'GrLivArea']], data.logSalePrice)


# In[ ]:


print('best-fit parameters:', grid.best_params_)
print('prediction rms residuals:', 10**(np.sqrt(-grid.best_score_)))


# 19% RMS loss is still pretty good given the fact that the entire feature palette results in 15% RMS loss.
# 
# Let's redo the Random Forest with only three features:

# In[ ]:


pipe = Pipeline([('model', RandomForestRegressor(n_jobs=-1, random_state=42))])

param_space = {'model__max_depth': [5, 7, 10],
               'model__max_features': [1, 2],
               'model__n_estimators': [70, 100, 120]}

grid = GridSearchCV(pipe, param_grid=param_space, cv=5, scoring='neg_mean_squared_error')

grid.fit(data.drop('logSalePrice', axis=1).loc[:, ['OverallQual', 'Neighborhood', 'GrLivArea']], data.logSalePrice)


# In[ ]:


print('best-fit parameters:', grid.best_params_)
print('prediction rms residuals:', 10**(np.sqrt(-grid.best_score_)))


# As expected from them previous plot, the Random Forest fares slightly better, as the three features combine more total weight in them. 
# 
# What if we include more features? The 15 highest-ranking features combine a cumulative weight of ~80% in them. Will that significantly improve the model accuracy?

# In[ ]:


pipe = Pipeline([('model', RandomForestRegressor(n_jobs=-1, random_state=42))])

param_space = {'model__max_depth': [10, 15, 20, 25],
               'model__max_features': [2, 4, 6],
               'model__n_estimators': [100, 150, 200, 250, 300]}

grid = GridSearchCV(pipe, param_grid=param_space, cv=5, scoring='neg_mean_squared_error')

grid.fit(data.drop('logSalePrice', axis=1).loc[:, results.sort_values('mean_rank').index[:15]], data.logSalePrice)


# In[ ]:


print('best-fit parameters:', grid.best_params_)
print('prediction rms residuals:', 10**(np.sqrt(-grid.best_score_)))


# Yes, the model fares significantly better: the resulting RMS loss is only 1.2% higher than that of the model using the entire feature set. This means in turn that 65 out of 80 features barely carry any useful information! 

# ## Conclusions
# 
# We can conclude the following from this analysis:
# * The coefficient of determination, the Random Forest feature importance, and the weights of linear models provide very consistent results (if the models parameters are picked properly, leading to good generalization) in identifying the most important features; the ranking of less important features, however, can be highly variable.
# * Training the models only on the highest-ranking features greatly improves performance (runtime) but also reduces the accuracy of the models. Feature selection based on feature importance can lead to model accuracies comparable to accuracies achieved with the full feature set, but with a significantly smaller data set.
# * This type of comparison is useful for the interpretation of data sets and model outcomes. 

# In[ ]:




