#!/usr/bin/env python
# coding: utf-8

# This is one of the first few Kaggle competitions I've participated in so it's quite special to me as it helped kick off my data science journey. I've taken inspiration from the following awesome notebook(s):-  
# 
# - [Stacked Regressions to predict House Prices](https://www.kaggle.com/serigne/stacked-regressions-top-4-on-leaderboard/notebook) by **Serigne**

# In[ ]:


# First, let's import the necessary stuff and load our training and test data
import pandas as pd
import numpy as np
import seaborn as sns
import scipy
sns.set()
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


# suppressing all warnings for readability (I wouldn't do this unless I'm really really sure...)
import warnings
warnings.filterwarnings("ignore")


# In[ ]:


train = pd.read_csv("../input/train.csv")
test = pd.read_csv("../input/test.csv")


# In[ ]:


train.info()


# In[ ]:


# firstly, we do not need the ID column in both the train and test datasets so we'll go ahead and drop them
train.drop('Id', axis = 1, inplace = True)
test_ids = test['Id']
test.drop('Id', axis = 1, inplace = True)


# ## Exploratory Data Analysis

# In[ ]:


# after studying the dataset, it is clear that there are quite a few outliers in several of the predictors
# but given that we only have 1460 rows to play with, it doesn't make sense to remove all of them
# So, for now, I am only removing the outliers from the 'ground living area' predictor

fig, axes = plt.subplots(nrows = 1, ncols = 2)
sns.scatterplot(x = train.GrLivArea, y = train.SalePrice, data = train, ax = axes[0])
axes[0].set_xlim(0, 4000)
axes[0].set_ylim(0, 650000)
axes[0].set_title("Before dropping outliers")
train = train[~(np.abs(train.GrLivArea - train.GrLivArea.mean()) > (3.5 * train.GrLivArea.std()))]
print(f"Updated shape of the dataset: {train.shape}")
sns.scatterplot(x = train.GrLivArea, y = train.SalePrice, data = train, ax = axes[1], color = "brown")
axes[1].set_xlim(0, 4000)
axes[1].set_ylim(0, 650000)
axes[1].set_title("After dropping outliers")
plt.tight_layout()


# ## Now, we'll analyze the response variable

# In[ ]:


# let's study the distribution of the response
sns.distplot(train.SalePrice)

# qq-plot
fig = plt.figure()
res = scipy.stats.probplot(train['SalePrice'], plot = plt)
plt.show()


# In[ ]:


# we see that the response data is skewed
# it would be preferable to have normality in our data, so let's do that first
# for this purpose, a natural log of the data seems to be doing the trick

train['SalePrice'] = np.log(train['SalePrice'])
fig = plt.figure()
res = scipy.stats.probplot(train['SalePrice'], plot = plt)
plt.show()


# ## Time for feature engineering

# In[ ]:


# to aid is better in this process, we would be combining the train and test data
# this would prove beneficial in some cases. For instance, when we are imputing missing data

ntrain = len(train)
y = train.SalePrice
combined = pd.concat([train, test], ignore_index = True)
combined.drop(columns='SalePrice', inplace=True)
print(f"Shape of combined data: {combined.shape}")


# In[ ]:


# let's analyze the missing data now
(combined.isna().sum().sort_values(ascending = False) / len(combined)) * 100


# In[ ]:


# we see that PoolQC, MiscFeature, Alley, Fence, FireplaceQu have a substantial number of missing values
# it doesn't make sense to try to impute/fill them as it'd be an inaccurate representation anyways
# so let's go ahead and drop them

combined.drop(['PoolQC', 'MiscFeature', 'Alley', 'Fence', 'FireplaceQu'], inplace = True, axis = 1)
print(f"Update shape of the combined dataset: {combined.shape}")


# In[ ]:


# for lot frontage, since we can roughly expect streets in a neighbourhood to have similar lot areas
# we can impute the missing values based on this grouping

combined.LotFrontage = combined.groupby('Neighborhood').LotFrontage.apply(lambda x: x.fillna(x.median()))


# In[ ]:


# the following will be replaced with "None"
for col in ('GarageType', 'GarageFinish', 'GarageQual', 'GarageCond', 'MSSubClass', 'MasVnrType', 'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2'):
    combined[col] = combined[col].fillna('None')


# In[ ]:


# the following will be replaced with a zero
for col in ('GarageYrBlt', 'GarageArea', 'GarageCars', 'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF','TotalBsmtSF', 'BsmtFullBath', 'BsmtHalfBath', 'MasVnrArea'):
    combined[col] = combined[col].fillna(0)


# In[ ]:


# the following categorical predictors will be replaced by the mode (most frequently occuring class)
combined['MSZoning'] = combined['MSZoning'].fillna(combined['MSZoning'].mode()[0])
combined['Electrical'] = combined['Electrical'].fillna(combined['Electrical'].mode()[0])
combined['KitchenQual'] = combined['KitchenQual'].fillna(combined['KitchenQual'].mode()[0])
combined['Exterior1st'] = combined['Exterior1st'].fillna(combined['Exterior1st'].mode()[0])
combined['Exterior2nd'] = combined['Exterior2nd'].fillna(combined['Exterior2nd'].mode()[0])
combined['SaleType'] = combined['SaleType'].fillna(combined['SaleType'].mode()[0])


# In[ ]:


# for function, the data description says NA means 'typical'
combined["Functional"] = combined["Functional"].fillna("Typical")

# and finally, for Utilities, we might as well get rid of it as almost all of the entries have the same value for it
# so it won't really aid us in the modeling process
combined.drop('Utilities', axis = 1, inplace = True)
print(f"The final dimensions of the combined data {combined.shape}")


# ## Aaanddd, that's a goodbye to missing values!
# 
# ### Now, let's do further feature engineering

# In[ ]:


# let's now transform some of the numerical variables that are really categorical
combined['MSSubClass'] = combined['MSSubClass'].apply(str)
combined['OverallCond'] = combined['OverallCond'].astype(str)


# In[ ]:


# given how important the total sq. footage is in determining a house price, we are creating a TotalSF predictor
combined['TotalSF'] = combined['TotalBsmtSF'] + combined['1stFlrSF'] + combined['2ndFlrSF']


# In[ ]:


# let's check for skewness in our numerical data
numeric_feats = combined.dtypes[combined.dtypes != "object"].index

# Check the skew of all numerical features
skewed_feats = combined[numeric_feats].apply(lambda x: x.skew()).sort_values(ascending = False)
skewness = pd.DataFrame({'Skew': skewed_feats})
skewness


# In[ ]:


# with the above info, let's perform a box-cox transformation to bring data as close as possible to a gaussian distribution
from scipy.special import boxcox1p

skewness = skewness[abs(skewness) > 0.75]
skewed_features = skewness.index
lam = 0.15
for feat in skewed_features:
    combined[feat] = boxcox1p(combined[feat], lam)


# In[ ]:


# finally, let's assign dummy variables to our categorical data (algorithms like linear regression need this)
combined = pd.get_dummies(combined)
combined.shape


# In[ ]:


# let's split our datasets back to training and testing
train = combined[:ntrain]
test = combined[ntrain:]


# ## Now, we can finally start modeling

# In[ ]:


# so we have the predictors in 'train' and the response in 'y'
# since we've already taken log of the response, it would suffice if we just calculated the RMSE
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import RobustScaler
from sklearn.pipeline import make_pipeline
y.reset_index(drop = True, inplace = True)

def cv_rmsle(model):
    cv_score_array = np.sqrt(-cross_val_score(model, train, y, cv = 5, scoring = "neg_mean_squared_error"))
    return cv_score_array.mean()


# ### Linear Regression

# In[ ]:


from sklearn.linear_model import LinearRegression

lm_rmsle = cv_rmsle(LinearRegression())
print(f"RMSLE for Linear Regression: [{lm_rmsle}]")


# ### LASSO

# In[ ]:


# for LASSO, we first need to determine the value of alpha
# we will use LassoCV for that purpose

from sklearn.linear_model import Lasso, LassoCV

lassocv_model = LassoCV(cv = 5, random_state = 1)
lassocv_model.fit(train, y)
best_alpha = lassocv_model.alpha_

lasso_model = make_pipeline(RobustScaler(), Lasso(alpha = best_alpha, random_state = 1))
lasso_rmsle = cv_rmsle(lasso_model)
print(f"RMSLE for LASSO (L1 Regularization): [{lasso_rmsle}]")


# ### Ridge Regression

# In[ ]:


from sklearn.linear_model import RidgeCV, Ridge

ridgecv_model = make_pipeline(RobustScaler(), RidgeCV(alphas = np.logspace(-10, 10, 100)))
ridge_rmsle = cv_rmsle(ridgecv_model)
print(f"RMSLE for Ridge Regression (L2 Regularization): [{ridge_rmsle}]")


# ### Elastic Net

# In[ ]:


from sklearn.linear_model import ElasticNet, ElasticNetCV

enetcv_model = ElasticNetCV(l1_ratio = np.arange(0.1, 1, 0.1), cv = 5, random_state = 1)
enetcv_model.fit(train, y)
best_l1_ratio = enetcv_model.l1_ratio_
best_alpha = enetcv_model.alpha_

enet_model = make_pipeline(RobustScaler(), ElasticNet(alpha = best_alpha, l1_ratio = best_l1_ratio, random_state = 1))
enet_rmsle = cv_rmsle(enet_model)
print(f"RMSLE for Elastic Net (L1 and L2 Regularization): [{enet_rmsle}]")


# ### Moving on to ensemble methods..
# 
# ### Gradient Boosting

# In[ ]:


from sklearn.ensemble import GradientBoostingRegressor

# after much trial and error, I arrived at the hyperparameters used in gradient boosting
gboost_model = GradientBoostingRegressor(loss = 'huber', learning_rate = 0.1, n_estimators = 3000, max_depth = 1, random_state = 1)
gboost_rmsle = cv_rmsle(gboost_model)
print(f"RMSLE for Gradient Boosting: [{gboost_rmsle}]")


# ### Adaptive Boosting (with LASSO as the estimator)

# In[ ]:


from sklearn.ensemble import AdaBoostRegressor

# after much trial and error, I arrived at the hyperparameters used in adaptive boosting
adaboost_lasso_model = AdaBoostRegressor(lasso_model, n_estimators = 50, learning_rate = 0.001, random_state = 1)
adaboost_lasso_rmsle = cv_rmsle(adaboost_lasso_model)
print(f"RMSLE for ADA Boosting (with LASSO): [{adaboost_lasso_rmsle}]")


# ### Adaptive Boosting (with Elastic Net as the estimator)

# In[ ]:


from sklearn.ensemble import AdaBoostRegressor

# after much trial and error, I arrived at the hyperparameters used in adaptive boosting
adaboost_enet_model = AdaBoostRegressor(enet_model, n_estimators = 50, learning_rate = 0.001, random_state = 1)
adaboost_enet_rmsle = cv_rmsle(adaboost_enet_model)
print(f"RMSLE for ADA Boosting (with Elastic Net): [{adaboost_enet_rmsle}]")


# ### Random Forests

# In[ ]:


from sklearn.ensemble import RandomForestRegressor

rforest_rmsle = cv_rmsle(RandomForestRegressor(random_state = 1))
print(f"RMSLE for Random Forests: [{rforest_rmsle}]")


# ### Extremely randomized trees

# In[ ]:


from sklearn.ensemble import ExtraTreesRegressor

etrees_rmsle = cv_rmsle(ExtraTreesRegressor(random_state = 1))
print(f"RMSLE for extremely randomized trees: [{etrees_rmsle}]")


# ### PCA
# 
# ### Let's see if reducing dimensionality has a positive effect on the predictions

# In[ ]:


from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler

pca = make_pipeline(RobustScaler(), PCA(n_components = 3, random_state = 1))
pca.fit(train.transpose())
print(f"Proportion of variance explained by the components: {pca.steps[1][1].explained_variance_ratio_}")

# we are using 3 components in this case
p_comps = pca.steps[1][1].components_.transpose()

pca_lm_rmsle = np.sqrt(-cross_val_score(LinearRegression(), p_comps, y, cv = 5, scoring = "neg_mean_squared_error")).mean()
print(f"RMSLE for Linear Regression after PCA reduction: [{pca_lm_rmsle}]")


# So PCA reduction didn't really help our case. So we are now done with all our models so it's time to have some fun.
# 
# Let's try combining some of these models and see if we get better results

# In[ ]:


from sklearn.model_selection import cross_val_predict

# we also need to define a function that returns cross-validated predictions for the data
def cv_pred(model):
    return cross_val_predict(model, train, y, cv = 5)


# We will try out various combinations of our top 3 models which are:-
# 1. Adaptive boosting with LASSO (`adaboost_lasso_model`)
# 2. Adaptive boosting with Elastic Net (`adaboost_enet_model`)
# 3. LASSO (`lasso_model`)

# In[ ]:


adaboost_enet_pred = cv_pred(adaboost_enet_model)
adaboost_lasso_pred = cv_pred(adaboost_lasso_model)
lasso_pred = cv_pred(lasso_model)


# In[ ]:


# now let's take the exponential of the responses to bring them back to their original scale
adaboost_enet_pred = np.exp(adaboost_enet_pred)
adaboost_lasso_pred = np.exp(adaboost_lasso_pred)
lasso_pred = np.exp(lasso_pred)


# In[ ]:


from sklearn.metrics import mean_squared_log_error
y_actual = np.exp(y)

rmsle_summary = pd.DataFrame(columns = ['ADA_ENet', 'ADA_Lasso', 'Lasso', 'RMSLE'])
rmsle_summary = rmsle_summary.append({'ADA_ENet':1, 'ADA_Lasso':0, 'Lasso':0, 'RMSLE':adaboost_enet_rmsle}, ignore_index = True)
rmsle_summary = rmsle_summary.append({'ADA_ENet':0, 'ADA_Lasso':1, 'Lasso':0, 'RMSLE':adaboost_lasso_rmsle}, ignore_index = True)
rmsle_summary = rmsle_summary.append({'ADA_ENet':0, 'ADA_Lasso':0, 'Lasso':1, 'RMSLE':lasso_rmsle}, ignore_index = True)

for i in np.arange(0.1, 0.9, 0.1).tolist():
    for j in np.arange(0.1, 1 - i, 0.1).tolist():
            final_preds = round(i, 1)*adaboost_enet_pred + round(j, 1)*adaboost_lasso_pred + round(1 - (i + j), 1)*lasso_pred
            rmsle = np.sqrt(mean_squared_log_error(y_actual, final_preds))
            rmsle_summary = rmsle_summary.append({'ADA_ENet':round(i, 1), 'ADA_Lasso':round(j, 1), 'Lasso':round(1 - (i + j), 1), 'RMSLE':rmsle}, ignore_index = True)
            
print(rmsle_summary)


# Ahh... all that effort and we still see that ADA boosting with LASSO gives us the best results!
# 
# ### Submission time!

# In[ ]:


adaboost_lasso_model.fit(train, y_actual)
submission_preds = adaboost_lasso_model.predict(test)

results = pd.DataFrame({'Id':test_ids, 'SalePrice':submission_preds})
results.to_csv("submission.csv", index = False)

