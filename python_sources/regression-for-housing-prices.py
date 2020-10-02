#!/usr/bin/env python
# coding: utf-8

# # Regression Models for Housing Prices 
# 
# By Akhil Jalan
# 
# # Load/Clean Data
# 
# Let's take a look at our dataset, and do some simple cleaning/reorganization for easy study later.

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[ ]:


import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


sns.set_style('whitegrid')


# In[ ]:


dat_path = '/kaggle/input/house-prices-advanced-regression-techniques/'


# In[ ]:


train_raw = pd.read_csv(f'{dat_path}train.csv')
test_raw = pd.read_csv(f'{dat_path}test.csv')


# In[ ]:


sample_sub = pd.read_csv(f'{dat_path}sample_submission.csv')


# In[ ]:


train_raw.shape


# In[ ]:


test_raw.shape


# In[ ]:


train_raw.dropna(axis=1)


# In[ ]:


test_raw.dropna(axis=1)


# ### How to deal with NaN values? 
# 
# It looks like quite a few columns are dropped when we drop Null values. However, we don't necessarily need to drop the entire column if only a few of the values are null - we can just as well drop a few rows if those are the only ones for which the column is null. 

# Let's get the counts of null values by column.

# In[ ]:


plt.rcParams['figure.figsize'] = [12, 8]
plt.rcParams['font.size'] = 16


# In[ ]:


plt.bar(np.arange(test_raw.shape[1]), test_raw.isna().sum() / len(test_raw))
plt.ylabel('Fraction of Rows with NaN Value')
plt.xlabel('Column Index')


# Let's do the same for the training data. 

# In[ ]:


plt.bar(np.arange(train_raw.shape[1]), train_raw.isna().sum() / len(train_raw))
plt.ylabel('Fraction of Rows with NaN Value')
plt.xlabel('Column Index')


# **Result**: It looks like there are quite a few columns where only a small fraction of the rows are null. whereas there are some for which a substantial portion (> 15 percent) are missing. Let's drop the columns where at least 15% of values are missing, and then retain the rest of the columns, but drop the rows for which a value is missing there. 

# In[ ]:


test_raw.shape


# In[ ]:


test = test_raw.dropna(thresh=test_raw.shape[0]*0.9, axis=1)
train = train_raw.dropna(thresh=train_raw.shape[0]*0.9, axis=1)


# In[ ]:


test.shape


# In[ ]:


train.shape


# In[ ]:


train = train.dropna(axis=0)


# In[ ]:


train.shape


# ### Convert Categorical Data
# 
# Quite a few of our variables are categorical (as opposed to numerical). Let's take a look at these and see if we can create one-hot vectors out of them without too much trouble.

# In[ ]:


train.columns.values


# In[ ]:


train['Exterior2nd'].dtype.name


# Let's gather up all the column names whose datatype is "object" (and so NOT numeric). 

# In[ ]:


categorical_train_cols = [col_name for col_name in train.columns.values if train[col_name].dtype.name == 'object']


# In[ ]:


categorical_train_cols += ['MSSubClass']


# In[ ]:


categorical_test_cols = [col_name for col_name in test.columns.values if test[col_name].dtype.name == 'object']


# In[ ]:


categorical_test_cols += ['MSSubClass']


# In[ ]:


numeric_cols = [col_name for col_name in train.columns.values if col_name not in categorical_train_cols]


# We can use the `get_dummies` function in Pandas to convert this categorical data into one-hot vectors. 

# In[ ]:


train_df = pd.concat((train[numeric_cols], pd.concat([
    pd.get_dummies(train[col_name], prefix = f'{col_name}') for col_name in categorical_train_cols
], axis = 1)), axis = 1)


# In[ ]:


numeric_test_cols = [col_name for col_name in test.columns.values if col_name not in categorical_test_cols]


# In[ ]:


test_df = pd.concat((test[numeric_test_cols], pd.concat([
    pd.get_dummies(test[col_name], prefix = f'{col_name}') for col_name in categorical_test_cols
], axis = 1)), axis = 1)


# In[ ]:


train_df


# In[ ]:


test_df


# ### Odds and Ends
# 
# * We need to split the train data into train, test, and validation data. Our "test_df" as of now has no training labels, because it's used for submission - so the name 'test' is a bit of a misnomer. We'll need a 3-way split because we're going to be comparing several regression models. Since the test data will be used for hyperparameter selection at the individualized model level (e.g. the regularization weight for LASSO), we need an entirely separate validation dataset at the level of comparing models. 
# 
# * We need to separate the predictive features ("X") from the sale price feature ("Y")
# 
# * We should eliminate the additonal columns present in the train data that are not in the test data. No point in using these features for prediction if they are not part of the test data. 

# In[ ]:


extra_train_cols = set(train_df.columns.values).difference(set(test_df.columns.values))


# In[ ]:


extra_test_cols = set(test_df.columns.values).difference(set(train_df.columns.values))


# In[ ]:


extra_train_cols.remove('SalePrice')


# In[ ]:


train_df = train_df.drop(columns = extra_train_cols)
test_df = test_df.drop(columns = extra_test_cols)


# In[ ]:


train_X = train_df.copy().drop(columns = ['SalePrice', 'Id'])
train_Y = train_df['SalePrice'].copy()


# In[ ]:


from sklearn.model_selection import train_test_split


# In[ ]:


train_X, test_X_all, train_Y, test_Y_all = train_test_split(train_X, train_Y, train_size=0.7, shuffle=True, random_state = 42)


# In[ ]:


train_X.shape


# In[ ]:


test_X_all.shape


# In[ ]:


submission_test_df = test_df.copy()


# In[ ]:


test_X, validation_X, test_Y, validation_Y = train_test_split(test_X_all, test_Y_all, train_size=0.6, shuffle=True)


# In[ ]:


test_X.shape


# In[ ]:


validation_X.shape


# # Data Exploration

# ## Distribution of Target Prices

# In[ ]:


plt.hist(train_Y, bins = 40)
plt.title('Distribution of Sale Prices for Train Data')


# The high-priced outliers will skew a model that is trained on squared errors. So, we should log-normalize sale prices to account for this. 

# In[ ]:


train_log_Y = np.log(train_Y)


# In[ ]:


plt.hist(train_log_Y, bins = 40)
plt.title('Distribution of (Log-Scaled) Sale Prices for Train Data')


# ## Feature Correlations

# In[ ]:


correlations_series = train_df.corrwith(train_log_Y, method='pearson').dropna()


# In[ ]:


correlations_series


# In[ ]:


sorted(correlations_series)


# In[ ]:


plt.bar(np.arange(len(correlations_series)), sorted(correlations_series))
plt.title('Correlation of Individual Features with Target Variable (LogSalePrice)')
plt.ylabel('Correlation (Pearson R)')
plt.xlabel('Feature Index Number');


# **Takeaway**: 
# 
# 1. Most features have fairly low correlation so we can impose a fairly strong condition on the number of features used without losing much signal. For regression, this involves a sparsity constraint on the feature vector. For decision trees, this means a maximum depth constraint. 
# 
# 2. We have an almost perfect balance between negatively and positively corelated features. This is useful for decision trees especially, as it enables us to split downwards (that is, predict a lower price) and upwards as we traverse the tree. 

# # Test Models 
# 
# ## Ordinary Least Squares
# 
# The simplest regression technique, and a good baseline for further testing. 

# In[ ]:


from numpy.linalg import lstsq, norm


# In[ ]:


# set rcond = -1 to use higher precision than the default
lstsq_weights, residuals, train_rank, train_sing_values = lstsq(train_X, train_log_Y, rcond= -1)


# Least squares regression uses a **Mean Squared Error** loss function. Mathematically, the vector $y$ is the target variable (log Sale Price) and the matrix $X$ is a collection of feature vectors, so that each house corresponds to a unique row. We solve for $w$ in the equation:
# 
# $$y = Xw$$ 
# 
# The resulting estimator is denoted $\hat w$, and the loss is then: 
# 
# $$ \mathcal{L}(\hat w) = \frac{1}{n} \left\lVert y - X \hat w \right\rVert_2$$
# 
# Where $n$ is the number of data points. 

# In[ ]:


lstsq_train_loss = norm(train_X.dot(lstsq_weights) - train_log_Y)**2 / len(train_X)


# In[ ]:


lstsq_train_loss


# Note that this is not the error in predicting housing precises, but rather the *log* of those prices. Here's the loss when we undo the logarithms. 

# In[ ]:


norm(np.exp(train_X.dot(lstsq_weights)) - train_Y) / len(train_X)


# So an average prediction error of about $500, which isn't bad. But we could certainly do better! Finally, let's get the test loss.

# In[ ]:


test_log_Y = np.log(test_Y)


# In[ ]:


lstsq_test_loss = norm(test_X.dot(lstsq_weights) - test_log_Y) / len(test_log_Y) 


# In[ ]:


lstsq_test_loss


# In[ ]:


norm(np.exp(test_X.dot(lstsq_weights)) - test_Y) / len(test_log_Y) 


# ## Random Forest

# In[ ]:


from sklearn.ensemble import RandomForestRegressor


# In[ ]:


rf_regressor = RandomForestRegressor(n_estimators = 100, max_depth = 10, random_state = 42)


# In[ ]:


rf_regressor.fit(train_X, train_log_Y)


# In[ ]:


rf_train_loss = norm(rf_regressor.predict(train_X) - train_log_Y) / len(train_X)


# In[ ]:


rf_test_loss = norm(rf_regressor.predict(test_X) - test_log_Y) / len(test_Y)


# In[ ]:


rf_train_loss


# In[ ]:


rf_test_loss


# Good! A test loss of $0.0104$ is a big improvement over the previous $0.0155$. However, notice that there is an even bigger difference between train/test loss here than in the previous model of OLS, indicating some overfitting. 

# ## Dealing with Overfitting
# 
# ### Weaker Random Forest
# 
# Let's use a much weaker RF model with just 10 trees and a max depth of 4. 

# In[ ]:


weak_rf_regressor = RandomForestRegressor(n_estimators = 10, max_depth = 4, random_state = 42)


# In[ ]:


weak_rf_regressor.fit(train_X, train_log_Y)


# In[ ]:


weak_rf_train_loss = norm(weak_rf_regressor.predict(train_X) - train_log_Y) / len(train_X)


# In[ ]:


weak_rf_test_loss = norm(weak_rf_regressor.predict(test_X) - test_log_Y) / len(test_X)


# In[ ]:


weak_rf_train_loss


# In[ ]:


weak_rf_test_loss


# While the train loss increases as we would expect, this doesn't help since the test loss actually increased slightly! We already have a fairly weak random forest regressor which does not solve the overfitting issue - let's try regularization for linear regression next.

# ### Sparse Regression - Is it worth it? 

# First, let's look at what OLS ended up doing for individual feature weights. The idea behind sparse/regularized regression is to ensure that *only a few features are used*, so we need to make sure this isn't already happening.

# In[ ]:


plt.bar(np.arange(len(lstsq_weights)), sorted(np.abs(lstsq_weights)))
plt.title('Feature Weights for Ordinary Least Squares Regression')
plt.ylabel('Coefficient (Absolute Value)');


# Indeed, it looks like there isn't really a bifurcation of features weights, and instead a significant portion are used with relatively large weight. Nevertheless, it looks like a significant portion of the weight is in the top 50-100 most important features.

# ### LASSO Regression
# 
# LASSO regression is a particular kind of linear regression in which the $\ell_1$-norm of the weight vector is part of the loss function. Hence there is an incentive to assign lower weights to features, which is what we hope to achieve in order to overcome overfitting.

# In[ ]:


from sklearn.linear_model import Lasso


# In[ ]:


lasso_model = Lasso(alpha = 1.0, normalize = True, fit_intercept = True, tol=1e-6, random_state = 42)


# In[ ]:


lasso_model.fit(train_X, train_log_Y)


# In[ ]:


lasso_train_loss = norm(lasso_model.predict(train_X) - train_log_Y) / len(train_X)
lasso_test_loss = norm(lasso_model.predict(test_X) - test_log_Y) / len(test_X)


# In[ ]:


lasso_train_loss


# In[ ]:


lasso_test_loss


# Now we're getting somewhere - the train and test loss are relatively close! Unfortunately the test loss is quite high, so we probably overshot a bit. Let's try a weaker regularization (e.g. lower $\alpha$)

# In[ ]:


lasso_model_001 = Lasso(alpha = 0.01, normalize = True, fit_intercept = True, tol=1e-6, random_state = 42)


# In[ ]:


lasso_model_001.fit(train_X, train_log_Y)


# In[ ]:


lasso_train_loss_001 = norm(lasso_model_001.predict(train_X) - train_log_Y) / len(train_X)
lasso_test_loss_001 = norm(lasso_model_001.predict(test_X) - test_log_Y) / len(test_X)


# In[ ]:


lasso_train_loss_001


# In[ ]:


lasso_test_loss_001


# We get a very slight improvement, but LASSO is still not competitive with the random forest model. Let's try an extremely weak regularization of $\alpha = 10^{-4}$ and see if we get anywhere.

# In[ ]:


lasso_model_1e4 = Lasso(alpha = 0.0001, normalize = True, fit_intercept = True, tol=1e-6, random_state = 42)


# In[ ]:


lasso_model_1e4.fit(train_X, train_log_Y)


# In[ ]:


lasso_train_loss_1e4 = norm(lasso_model_1e4.predict(train_X) - train_log_Y) / len(train_X)
lasso_test_loss_1e4 = norm(lasso_model_1e4.predict(test_X) - test_log_Y) / len(test_X)


# In[ ]:


lasso_train_loss_1e4


# In[ ]:


lasso_test_loss_1e4


# In[ ]:


lstsq_test_loss


# Great! Looks like we saw a real improvement with the lower $\alpha$. Now LASSO outperforms OLS on the test loss, although it still is outperformed by random forest models. Let's compare feature weights for OLS and the weakest LASSO to see what the practical effect is in terms of feature importance. 

# In[ ]:


lasso_weights = lasso_model_1e4.coef_


# In[ ]:


plt.scatter(np.abs(lstsq_weights), np.abs(lasso_weights), s = 10, marker='o');
plt.xlabel('Feature Weight for Least Squares')
plt.ylabel('Feature Weight for LASSO')
plt.title('Feature Weights for Regression with/without Regularization');


# **How to interpret this plot**: This mainly tells use that LASSO assigned much lower weights to most features. The almost-solid line at the $x$-axis tells use that many nonzero features in OLS were set to zero for LASSO. Despite this, LASSO has a better test error! Therefore we found that correcting for the overfitting was worth it. 
# 
# **Comparison to Random Forest**: Interestingly, while we were able to get a useful improvement for linear regression when using regularization, our weaker random forest model had worse test error. This might indicate a *double descent* type phenomenon, or just that we didn't search hyperparameter space well enough.

# # Model Selection and Result Submission

# So far we have tested 4 regression models: 
# 
# 1. Ordinary Least Squares Regression 
# 
# 2. LASSO Regression (AKA Sparse Linear Regression) 
# 
# 3. Random Forest Regression 
# 
# 4. Low-Depth RF Regression (Random Forest with Stronger Sparsity Constraints)
# 
# To select the best-performing model, we need to test on our *validation data* to avoid data incest. 

#  ## Compute Validation Error

# In[ ]:


validation_log_Y = np.log(validation_Y)


# In[ ]:


lasso_validation_loss_1e4 = norm(lasso_model_1e4.predict(validation_X) - validation_log_Y) / len(validation_X)
weak_rf_regressor_validation_loss = norm(weak_rf_regressor.predict(validation_X) - validation_log_Y) / len(validation_X)
rf_regressor_validation_loss = norm(rf_regressor.predict(validation_X) - validation_log_Y) / len(validation_X)


# In[ ]:


lstsq_validation_loss = norm(validation_X.dot(lstsq_weights) - validation_log_Y) / len(validation_X)


# In[ ]:


lasso_validation_loss_1e4


# In[ ]:


weak_rf_regressor_validation_loss


# In[ ]:


rf_regressor_validation_loss


# In[ ]:


lstsq_validation_loss


# # Conclusion/Insights: Random Forest Wins
# 
# Although regularized models like LASSO or weaker random forest did reduce the gap between train/test loss, it appears that a random forest model performed best. In terms of insights, what this tells us is that **model strength overcomes overfitting**. Of course, this doesn't answer the question of **how many features we really need** for this problem. To answer this, we need to look at feature weights for the best-performing random forest model.

# ## Why is Random Forest Better?
# 
# One hypothesis for why random forest is better is that one really need a lot of features to get all the signal in the data. This is why regularization might fail. Is this true? Let's take a look at feature weights for the random forest regressor.

# In[ ]:


plt.bar(np.arange(len(rf_regressor.feature_importances_)), rf_regressor.feature_importances_)
plt.ylabel('Feature Importance (Sums to 1)')
plt.title('Feature Importance for Random Forest Regressor')


# Surprisingly, the majority of the weight goes to a single feature! But beyond the top 2, it looks like there are several smaller features making a contribution.
# 
# How many features do we need to get to 80 or 90 percent weight?

# In[ ]:


rf_weights = rf_regressor.feature_importances_


# In[ ]:


plt.plot(np.arange(len(rf_weights)), np.cumsum(sorted(rf_weights, reverse=True)), marker='^')
plt.title('Cumulative Feature Weight for Random Forest')
plt.ylabel('Sum of Feature Weights')
plt.xlabel('Index of feature weight')


# As we can see, while it only takes a few features to get to around 90% feature weight, we need to get the top 100 or so features to reach roughly 99% of the full feature weights. Hence the *long-tailed* distribution of feature weights explains *why the random forest outperforms other models*. Its additional depth and number of trees allows it to account for the fine-grained signal in these additional features.

# ## Which Features Matter? 

# In[ ]:


top_feature_indices = np.where(rf_weights > 0.01)


# In[ ]:


train_X.columns.values[top_feature_indices]


# In[ ]:


rf_weights[top_feature_indices]


# In[ ]:


plt.bar(np.arange(len(top_feature_indices[0])), rf_weights[top_feature_indices])
plt.xticks(np.arange(len(top_feature_indices[0])), train_X.columns.values[top_feature_indices], rotation=60);
plt.ylabel('Feature Weight')
plt.title('Weights for Top Features in Random Forest Regressor');


# The two strongest features are thus: 
#     
# * *OverallQual: Rates the overall material and finish of the house*
#     
# * *GrLivArea: Above grade (ground) living area square feet*
# 
# Both make sense. The first feature is a sort of human-based assessment which corresponds to the quality of the house. The second is square footage, a common metric used in home sales which no doubt influences advertising and negotations. 
# 
# Interestingly, all of the top features seem to be numeric (as opposed to categorical) variables. Perhaps this tells us soemthing about the *psychology of home pricing* - namely, that people prefer to base prices on seemingly objective metrics like square footage, number of cars that can be fit in the garage,the year built, and so on.

# # Prepare Submission

# In[ ]:


sample_submission_df = pd.read_csv(f'{dat_path}sample_submission.csv')


# In[ ]:


submission_test_df.shape


# In[ ]:


train_X.shape


# In[ ]:


submission_X = submission_test_df.drop(columns = ['Id'])


# We need to fill in missing values in the submission DF. Let's use the mean of each feature.

# In[ ]:


feature_means = np.mean(train_X, axis=0)


# In[ ]:


submission_X_no_nan = submission_X.fillna(value=feature_means)


# In[ ]:


submission_X_no_nan.shape


# In[ ]:


submission_Y_predict = np.exp(rf_regressor.predict(submission_X_no_nan))


# In[ ]:


submission_df_final = pd.concat((submission_test_df['Id'], pd.Series(submission_Y_predict)), axis = 1)


# In[ ]:


submission_df_final.rename(columns = {0: 'SalePrice'}, inplace=True)


# In[ ]:


submission_df_final


# In[ ]:





# In[ ]:





# In[ ]:


submission_df_final.to_csv('house_prices_submission.csv')

