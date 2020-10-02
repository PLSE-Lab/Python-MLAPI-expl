#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
import warnings
warnings.filterwarnings('ignore')


# In[ ]:


train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')


# In[ ]:


train.head()


# # Visualization 

# ## Normal distribution

# Consider the independent variable SalePrice. The graph below shows that the distribution of data differs from the normal distribution.

# In[ ]:


plt.subplots(figsize=(8, 6))
sns.distplot(train['SalePrice'], kde = False, fit = stats.norm)


# This confirms the probability plot, which shows that the SalePrice has a'peakedness', a positive skewness and does not follow the diagonal line.

# In[ ]:


prob = stats.probplot(train['SalePrice'], plot=plt)


# Numerical measurement of data deviation from the normal distribution. With normal data distribution Skewness is 0, Kurtosis is = 3.

# In[ ]:


print('Skewness: %f' % train['SalePrice'].skew())
print('Kurtosis: %f' % train['SalePrice'].kurt())


# Normal data distribution is important because several statistics tests rely on this (e.g. t-statistics) and it helps to improve model accuracy. Therefore, you should convert the data. This can be done using the logarithmic function.

# In[ ]:


train['SalePrice'] = np.log1p(train['SalePrice'])


# As can be seen in the following graphs, after the conversion, the data distribution is close to normal.

# In[ ]:


plt.subplots(figsize=(8, 4))
sns.distplot(train['SalePrice'], kde = False, fit = stats.norm)
plt.figure()
prob = stats.probplot(train['SalePrice'], plot=plt)


# In[ ]:


print('Skewness: %f' % train['SalePrice'].skew())
print('Kurtosis: %f' % train['SalePrice'].kurt())


# ## Correlation matrix

# Next, I construct a correlation matrix. The spearman method was used because spearman's coefficient looks at the relative order of values for each variable. This makes it appropriate to use with both continuous and discrete data.

# In[ ]:


corrmat = train.corr(method='spearman')
plt.subplots(figsize=(12, 9))

k = 15 #number of variables for heatmap
cols = corrmat.nlargest(k, 'SalePrice')['SalePrice'].index
cm = np.corrcoef(train[cols].values.T)
sns.set(font_scale=1.25)
hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 8}, yticklabels=cols.values, xticklabels=cols.values)
plt.show()


# With the help of the matrix you can see the most important features.For SalePrice this is 'OverallQual', 'GrLivArea', 'GarageCars', 'YearBuilt', 'GarageArea', 'FullBath', 'TotalBsmtSF'. Consider them in more detail.

# In[ ]:


sns.set()
cols = ['SalePrice', 'OverallQual', 'GrLivArea', 'GarageCars', 'YearBuilt', 'GarageArea', 'FullBath', 'TotalBsmtSF']
sns.pairplot(train[cols], size = 2.5)
plt.show();


# Here you can see that there seems to be a connection between the data. Since as the feature increases, SalePrice also increases, this is especially noticeable by OverallQual and GrLivArea. You may also notice observations that may be outliers on the graphs GrLivArea and GarageArea. Further attention should be paid to this.

# I also build a boxplot for some feature.

# In[ ]:


data = pd.concat([train['SalePrice'], train['OverallQual']], axis=1)
f, ax = plt.subplots(figsize=(8, 6))
fig = sns.boxplot(x='OverallQual', y="SalePrice", data=data)


# In[ ]:


data = pd.concat([train['SalePrice'], train['GarageCars']], axis=1)
f, ax = plt.subplots(figsize=(8, 6))
fig = sns.boxplot(x='GarageCars', y="SalePrice", data=data)


# In[ ]:


data = pd.concat([train['SalePrice'], train['FullBath']], axis=1)
f, ax = plt.subplots(figsize=(8, 6))
fig = sns.boxplot(x='FullBath', y="SalePrice", data=data)


# Boxplot also indicates the relationship of independent variables with target variable.

# # Preprocessing

# ## Outliers

# Although it seemed that obvious outliers were identified, in practice their exclusion leads to an increase in the rmse model. Therefore, it was decided not to exclude them.

# ## Missing data

# Compound train and test set to find missing data with the exception of the Id and SalePrice fields

# In[ ]:


all_data = pd.concat((train.iloc[:, 1:-1], test.iloc[:, 1:]))


# Counting Total Missing Values

# In[ ]:


missing = all_data.isnull().sum()
missing = missing[missing > 0]
missing.sort_values(inplace=True)
missing.plot.bar()
missing_data = pd.DataFrame({'Total': missing})
missing_data.sort_values(by='Total',ascending=False)


# In[ ]:


all_data.shape


# Since variables containing missing data more than 100 do not have a strong effect on the target variable or have highly correlated variables without missing data, I consider it possible to delete these variables with missing values. In other cases, apply averaging.

# In[ ]:


all_data = all_data.drop((missing_data[missing_data['Total'] > 100]).index,1)


# Coding quality variables

# In[ ]:


from sklearn.preprocessing import OneHotEncoder
all_data = pd.get_dummies(all_data)


# Apply averaging

# In[ ]:


all_data = all_data.fillna(all_data.mean())


# In[ ]:


all_data.shape


# Check for missing values

# In[ ]:


all_data.isnull().sum().max()


# # Normalization

# compute logarithm of quantitative independent variables in which Skewness is greater than 0.5. In order to achieve a normal distribution of data, as in the case of SalePrice

# In[ ]:


quan_feats = all_data.dtypes[all_data.dtypes != "object"].index

skewed_feats = all_data[quan_feats].apply(lambda x: stats.skew(x))
skewed_feats = skewed_feats[skewed_feats > 0.5].index

all_data[skewed_feats] = np.log1p(all_data[skewed_feats])


# # Modeling

# Perform selection on training and test set

# In[ ]:


X_train = all_data[:train.shape[0]]
X_test = all_data[train.shape[0]:]
y = train['SalePrice']


# In[ ]:


from sklearn.linear_model import Ridge, Lasso
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import RobustScaler

# additionally I produce robust scaling to increase model accuracy
scaler = RobustScaler().fit_transform(X_train) 

#creating the cross validation function for ridge and lasso
def rmse_cv(model):
    rmse= np.sqrt(-cross_val_score(model, scaler, y, scoring="neg_mean_squared_error", cv = 5))
    return(rmse)


# ## Ridge Regularization

# implement the ridge with different lambda and find rmse with cross validation

# In[ ]:


alphas_r = [0.05, 0.1, 0.3, 1, 3, 5, 10, 15, 30, 50, 75]

val_errors_r = []
for alpha in alphas_r:
    ridge = Ridge(alpha = alpha)
    errors_r = rmse_cv(ridge).mean()
    val_errors_r.append(errors_r)


# The graph shows that rmse decreases first, but then with increasing lambda the error increases as well. The larger the lambda, the less model prones to overfit, but this reduces the model's ability to generalize the data

# In[ ]:


plt.plot(alphas_r, val_errors_r)
plt.title('Ridge')
plt.xlabel('lambda')
plt.ylabel('rmse')


# In[ ]:


print('best alpha: {}'.format(alphas_r[np.argmin(val_errors_r)]))
print('Min RMSE: {}'.format(min(val_errors_r)))


# ## Lasso Regularization

# I will do a slightly different approach here and use the built in Lasso to figure out the best alpha.

# In[ ]:


alphas_l = [0.00005, 0.0001, 0.0002, 0.0003, 0.0004, 0.0005,
           0.0006, 0.0007, 0.0008, 1e-3, 5e-3]

val_errors_l = []
for alpha in alphas_l:
    lasso = Lasso(alpha = alpha)
    errors_l = rmse_cv(lasso).mean()
    val_errors_l.append(errors_l)


# In[ ]:


plt.plot(alphas_l, val_errors_l)
plt.title('Lasso')
plt.xlabel('alpha')
plt.ylabel('rmse')


# In[ ]:


print('best alpha: {}'.format(alphas_l[np.argmin(val_errors_l)]))
print('Min RMSE: {}'.format(min(val_errors_l)))


#  The lasso performs even better.

# # XGboost

# In[ ]:


import xgboost as xgb


# In[ ]:


#Create a train and test matrix for xgb
dtrain = xgb.DMatrix(data = X_train, label = y)
dtest = xgb.DMatrix(X_test)


# # Untuned Model

# First we use xgb without tuning parameters and calculate the rmse using cross validation

# In[ ]:


untuned_params = {'objective':'reg:linear'}
untuned_cv = xgb.cv(dtrain = dtrain, params = untuned_params, nfold = 4, metrics='rmse', as_pandas=True, seed = 5)


# In[ ]:


print('Untuned rmse: %f' % (untuned_cv["test-rmse-mean"].tail(1).values[0]))


# # Grid Search

# Use Grid Search to find the optimal parameters. Since Grid Search requires more computer resources, the selection of parameters was performed by gbm_param_grid was implemented several times with fewer parameters.

# In[ ]:


gbm_param_grid = {
    'colsample_bytree': [0.3],
#    'subsample': [0.3,0.5, 0.7, 1],
    'n_estimators': [400, 450, 500],
    'max_depth': [3],
    'learning_rate' : [0.1]
}


# In[ ]:


gbm = xgb.XGBRegressor()


# In[ ]:


from sklearn.model_selection import GridSearchCV
grid_mse = GridSearchCV(param_grid = gbm_param_grid, estimator = gbm, scoring="neg_mean_squared_error", cv = 4)


# In[ ]:


grid_mse.fit(X_train,y)


# In[ ]:


print("Best parameters found: ", grid_mse.best_params_)
print("Lowest RMSE found: ", np.sqrt(np.abs(grid_mse.best_score_)))


# At the end, the following parameters were obtained 'n_estimators': 450, 'learning_rate': 0.1, 'max_depth': 3, 'colsample_bytree': 0.3, 'subsample': 1

# # Tuned Model

# Building xgb with tuning parameters

# In[ ]:


tuned_params = {'objective':'reg:linear', 'n_estimators': 450, 'learning_rate': 0.1, 'max_depth': 3, 'colsample_bytree': 0.3, 'subsample': 1}
tuned_cv = xgb.cv(dtrain = dtrain, params = tuned_params, nfold = 4, num_boost_round = 500, metrics='rmse', as_pandas=True, seed = 5)


# In[ ]:


print('Tuned rmse: %f' % (tuned_cv["test-rmse-mean"].tail(1).values[0]))


# rmse is better than the evaluation of the untuned model

# # XGboost tuned with L2

# further regularization L2 and L1 were made on tuned model

# In[ ]:


l2_params = [0.05, 0.1, 0.3, 1, 3, 5, 10, 15, 30, 50, 75]
tuned_params = {'objective':'reg:linear', 'n_estimators': 450, 'learning_rate': 0.1, 'max_depth': 3, 'colsample_bytree': 0.3, 'subsample': 1}


# In[ ]:


rmses_l2 = []


# In[ ]:


for reg in l2_params:
    tuned_params['lambda'] = reg
    cv_results_rmse = xgb.cv(tuned_params,dtrain, num_boost_round=500, early_stopping_rounds=100, nfold=4, metrics ='rmse', as_pandas=True, seed =123)
    rmses_l2.append(cv_results_rmse['test-rmse-mean'].tail(1).values[0])


# In[ ]:


print("Best rmse as a function of l2:")
print(pd.DataFrame(list(zip(l2_params, rmses_l2)), columns=["l2", "rmse"]), '\n')
print('Min L2 Tuned rmse: %f' % (min(rmses_l2)))
print('Min lambda: %f' % (min(l2_params)))


# # XGboost tuned with L1

# In[ ]:


l1_params = [0.00005, 0.0001, 0.0002, 0.0003, 0.0004, 0.0005, 0.0006, 0.0007, 0.0008, 1e-3, 5e-3]
tuned_params = {'objective':'reg:linear', 'n_estimators': 450, 'learning_rate': 0.1, 'max_depth': 3, 'colsample_bytree': 0.3, 'subsample': 1}


# In[ ]:


rmses_l1 = []


# In[ ]:


for reg in l1_params:
    tuned_params['alpha'] = reg
    cv_results_rmse = xgb.cv(tuned_params,dtrain, num_boost_round=500, early_stopping_rounds=100, nfold=4, metrics ='rmse', as_pandas=True, seed =123)
    rmses_l1.append(cv_results_rmse['test-rmse-mean'].tail(1).values[0])


# In[ ]:


print("Best rmse as a function of l2:")
print(pd.DataFrame(list(zip(l1_params, rmses_l1)), columns=["l1", "rmse"]), '\n')
print('Min L1 Tuned rmse: %f' % (min(rmses_l1)))
print('Min alpha: %f' % (min(l1_params)))


# # Prediction

# For the prediction, we will use the XGboost model with L1 regularization, because this variant has the lowest estimate rmse

# In[ ]:


model_xgb = xgb.XGBRegressor(objective= 'reg:linear',reg_alpha = 0.00005, n_estimators=500, learning_rate=0.1, max_depth=3, colsample_bytree=0.3, subsample=1) 
model_xgb.fit(X_train, y)


# In[ ]:


xgb_preds = np.expm1(model_xgb.predict(X_test))


# Also we can use lasso regression for prediction

# In[ ]:


lasso.fit(X_train, y)
lass_pred = np.expm1(lasso.predict(X_test))


# You can use the average of the prediction of the two models, but unfortunately the improvement in the accuracy of the model in this case did not affect. So i used only xgb model for submission.

# In[ ]:


preds = xgb_preds


# In[ ]:


solution = pd.DataFrame({"id":test.Id, "SalePrice":preds})
solution.to_csv("sol5.csv", index = False)

