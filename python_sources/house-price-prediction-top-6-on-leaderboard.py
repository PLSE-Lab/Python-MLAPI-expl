#!/usr/bin/env python
# coding: utf-8

# ## Description
# 
# 
# Ask a home buyer to describe their dream house, and they probably won't begin with the height of the basement ceiling or the proximity to an east-west railroad. But this playground competition's dataset proves that much more influences price negotiations than the number of bedrooms or a white-picket fence.
# 
# With 79 explanatory variables describing (almost) every aspect of residential homes in Ames, Iowa, this competition challenges you to predict the final price of each home.

# ## Goal
# The goal is to predict the Sales Price for the test dataset with the given features.

# ## Steps-
#     1. Introduction
#     2. Read Data
#     3. Explore Data
#         a. Data size
#         b. Strtucture of Data
#     4. Exploratory Data Analysis & Visualization
#         a. Check Missing Values
#         b. Correlation between missing values and Sales Price
#         c. Check numerical varibale
#         d. Temporal variables and correlation with Sales Price
#         e. Discrete variables and correlation with Sales Price
#         f. Continuous variables, skeweness and outliers
#         g. Categorial variables and correlation with Sales Price
#     5. Missing value Imputation
#     6. Handle Rare Categorial Features
#     7. Label Encoding
#     8. Scaling
#     9. Train models
#         a. Lasso Regression
#         b. Elastic Net Regression
#         c. Kernel Ridge Regression
#         d. Support Vector Regression
#         e. Gradient Boosting Regression
#         f. XGBoost Regression
#         g. Light GBM Regression
#         h. Random Forest Regression
#     10. Stack models
#     11. Visualize model scores

# # Import packages

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
from sklearn.linear_model import ElasticNet, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import mean_squared_error
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.svm import SVR


# # Read Data

# In[ ]:


# Display all columns of a dataframe
pd.pandas.set_option("display.max_columns", None)


# #### Dataset for House Price Prediction is from below URL:
# #### https://www.kaggle.com/c/house-prices-advanced-regression-techniques/data    

# In[ ]:


train = pd.read_csv("../input/house-prices-advanced-regression-techniques/train.csv")
train.head()


# In[ ]:


test = pd.read_csv("../input/house-prices-advanced-regression-techniques/test.csv")
test.head()


# # Explore Data

# In[ ]:


print("Train Dataset: ", train.shape)
print("Test Dataset: ", test.shape)


# In[ ]:


train.describe()


# In[ ]:


train.info()


# In[ ]:


# Sales Price Distribution
sns.set_style("white")
sns.set_color_codes(palette='deep')
f, ax = plt.subplots(figsize=(6,5))
sns.distplot(train['SalePrice'], color="b");
ax.xaxis.grid(False)
ax.set(ylabel="Frequency")
ax.set(xlabel="Sales Price")
ax.set(title="Home Sales Price Distribution")
sns.despine(trim=True, left=True)
plt.show()


# As we see here that the SalePrice is skewed towards right. 
# And it is a problem because most of the ML models don't perform well with skewed/un-normally distributed data. 
# So we have ton apply a log(x) tranform to fix the skew.

# In[ ]:


# Skew and kurt
print("Skewness: %f" % train['SalePrice'].skew())
print("Kurtosis: %f" % train['SalePrice'].kurt())


# In[ ]:


plt.figure(figsize=(20,10))
sns.heatmap(data=train.corr(), cmap="Blues", square=True)
plt.show()


# # Exploratiory Data Analysis

# ### Check Missing Values

# In[ ]:


# Get features with missing values
features_with_na = [feature for feature in train.columns if train[feature].isnull().sum() > 0]


# In[ ]:


# Print missing features and its percentage in train dataset
for feature in features_with_na:
    print(feature, np.round(train[feature].isnull().mean(), 4), "% missing values")


# #### As we see there are many missing values in the train dataset, so lets check the relationship between missing values and the sales price.

# In[ ]:


for feature in features_with_na:
    data = train.copy()
    #Create a variable that indicates 1 if the values is missing and 0 otherwise.
    data[feature] = np.where(data[feature].isnull(), 1, 0)
    
    # Plot bar graph of median SalesPrice for values missing or present in train dataset
    data.groupby(feature)['SalePrice'].median().plot.bar()
    plt.xlabel(feature)
    plt.ylabel("SalePrice")
    plt.title(feature)
    plt.show()


# As we see here for most houses where features are missing the Sales Price is comparatevily low.
# Why? We will know in a while.

# ### Numerical Variables

# In[ ]:


numerical_features = [feature for feature in train.columns if train[feature].dtype != 'O']
print("Number of numerical features: ", len(numerical_features))
train[numerical_features].head()


# ### Temporal Variables (Date-time variables)
# <p> In the above train dataset we have 4 temporal variables</p>

# In[ ]:


temporal_features = [feature for feature in numerical_features if 'Year' in feature or 'Yr' in feature]
print("Number of temporal features: ", len(temporal_features))
train[temporal_features].head()


# <p> Lets see the relation between temporal variables and SalesPrice.

# In[ ]:


for feature in temporal_features:
    data = train.copy()
    
    data.groupby(feature)['SalePrice'].median().plot()
    plt.xlabel(feature)
    plt.ylabel('SalePrice')
    plt.title(feature)
    plt.show()


# <p>The first 3 plots here look fine as the recent the year house is built/remodling done/garage build, the higher the SalesPrice.
# But in the 4th plot Sales Price is decreasing as the Year is increasing. Ideally SalesPrice should increase with every passing year.</p>

# <p> So lets see the relation between the first 3 year variables and the Year Sold

# In[ ]:


for feature in temporal_features:
    data = train.copy()
    
    if feature != 'YrSold':
        data[feature] = data['YrSold'] - data[feature]
        plt.scatter(data[feature], data['SalePrice'])
        plt.xlabel(feature)
        plt.ylabel('SalePrice')
        plt.title(feature)
        plt.show()


# <p> So above scatter plot indicates: 
# 1. The lesser the difference between house YrSold and house year built/remodling done/garagebuilt, the higher the Sales Price.
# 2. When Sales Price is less then it means the house is old with no/not recent alterations done.

# So now we also know that "Houses where faeture values are missing have comparatively low price", because no remodelling or feature enhancements are done recently.

# ### Discrete Variables

# In[ ]:


discrete_features = [feature for feature in numerical_features if len(train[feature].unique()) <=25 
                     and feature not in temporal_features + ['Id']]
print("Length of discrete features: ", len(discrete_features))
train[discrete_features].head()


# In[ ]:


for feature in discrete_features:
    data = train.copy()
    
    data.groupby(feature)['SalePrice'].median().plot.bar()
    plt.xlabel(feature)
    plt.ylabel('SalePrice')
    plt.title(feature)
    plt.show()


# ### Continuous Variables

# In[ ]:


continuous_features = [feature for feature in numerical_features if feature not in discrete_features + temporal_features + ['Id']]
print("Length of continuous features: ", len(continuous_features))
train[continuous_features].head()


# In[ ]:


for feature in continuous_features:
    data = train.copy()
    
    data[feature].hist(bins=25)
    plt.xlabel(feature)
    plt.ylabel("Count")
    plt.title(feature)
    plt.show()


# <p> As the continuous variables are all skewed, we will use logarithmic transformation to visualize.

# In[ ]:


for feature in continuous_features:
    data = train.copy()
    if 0 in data[feature].unique():
        pass
    else:
        data[feature] = np.log(data[feature])
        plt.scatter(data[feature], data['SalePrice'])
        plt.xlabel(feature)
        plt.ylabel('Sale Price')
        plt.title(feature)
        plt.show()


# <p>While converting variables using logarithmic transformation we see there are only 5 skewed variables - LotFrontage, LotArea, 
# 1stFlrSF, GrLivArea and SalePrice which has non-zero values. </p>

# ### Outliers

# In[ ]:


for feature in continuous_features:
    data = train.copy()
    if 0 in data[feature].unique():
        pass
    else:
        data[feature] = np.log(data[feature])
        data.boxplot(column=feature)
        plt.title(feature)
        plt.show()


# ### Categorial Variables

# In[ ]:


categorial_features = [feature for feature in train.columns if train[feature].dtypes == 'O']
print(categorial_features)


# In[ ]:


train[categorial_features].head()


# In[ ]:


for feature in categorial_features:
    print("Feature {} has {} unique values".format(feature, len(train[feature].unique())))


# <p>Plot categorial features with target variable

# In[ ]:


for feature in categorial_features:
    data = train.copy()
    data.groupby(feature)['SalePrice'].median().plot.bar()
    plt.xlabel(feature)
    plt.ylabel('Sale Price')
    plt.title(feature)
    plt.show()


# ## Missing value imputation

# #### Categorical Variables

# In[ ]:


train[categorial_features].head()


# In[ ]:


categorial_with_nan = [feature for feature in categorial_features if train[feature].isnull().sum() > 0]
print(categorial_with_nan)
for feature in categorial_with_nan:
    print("Feature {}, has {}% missing values in train dataset", (feature, np.round(train[feature].isnull().mean(), 4)))


# In[ ]:


for feature in categorial_with_nan:
    train[feature].fillna('Missing', inplace=True)


# In[ ]:


categorial_with_nan = [feature for feature in categorial_features if test[feature].isnull().sum() > 0]
print(categorial_with_nan)
for feature in categorial_with_nan:
    print("Feature {}, has {}% missing values in test dataset", (feature, np.round(test[feature].isnull().mean(), 4)))


# In[ ]:


for feature in categorial_with_nan:
    test[feature].fillna('Missing', inplace=True)


# In[ ]:


train[categorial_features].head()


# In[ ]:


test[categorial_features].head()


# In[ ]:


print("Train Dataset Categorial Features:",train[categorial_features].shape)
print("Test Dataset Categorial Features:",test[categorial_features].shape)


# #### Numeric Variables

# In[ ]:


print(numerical_features)


# In[ ]:


numerical_with_nan = [feature for feature in numerical_features if train[feature].isnull().sum() > 0]
print(numerical_with_nan)
for feature in numerical_with_nan:
    print("Feature {} has {}% missing values in train datset", (feature,np.round(train[feature].isnull().mean(), 4)))


# Replace the missing values in train set with median since there are outliers

# In[ ]:


for feature in numerical_with_nan:
    train[feature].fillna(train[feature].median(), inplace=True)


# In[ ]:


train.head()


# In[ ]:


numerical_with_nan = [feature for feature in numerical_features if feature not in ['SalePrice'] and test[feature].isnull().sum() > 0]
print(numerical_with_nan)
for feature in numerical_with_nan:
    print("Feature {} has {}% missing values in test datset", (feature,np.round(test[feature].isnull().mean(), 4)))


# Replace the missing values in test set with median since there are outliers

# In[ ]:


for feature in numerical_with_nan:
    test[feature].fillna(test[feature].median(), inplace=True)


# In[ ]:


test.head()


# In[ ]:


print("Train Dataset", train.shape)
print("Test Dataset", test.shape)


# #### Temporal Variables

# In[ ]:


train[temporal_features].head()


# In[ ]:


for feature in ['YearBuilt', 'YearRemodAdd', 'GarageYrBlt']:
    train[feature] = train['YrSold'] - train[feature]
    test[feature] = test['YrSold'] - test[feature]


# In[ ]:


train[temporal_features].head()


# In[ ]:


test[temporal_features].head()


# ##### Since the numeric variables are skewed, we will perform log normal distribution.

# In[ ]:


train.head()


# In[ ]:


num_non_zero_skewed_features_train_set = ['LotFrontage', 'LotArea', '1stFlrSF', 'GrLivArea', 'SalePrice']
train[num_non_zero_skewed_features_train_set].head()


# In[ ]:


for feature in num_non_zero_skewed_features_train_set:
    train[feature] = np.log(train[feature])


# We may assume the same numeric features will be skewed in test set as well.

# In[ ]:


num_non_zero_skewed_features_test_set = ['LotFrontage', 'LotArea', '1stFlrSF', 'GrLivArea']
test[num_non_zero_skewed_features_test_set].head()


# In[ ]:


for feature in num_non_zero_skewed_features_test_set:
    test[feature] = np.log(test[feature])


# In[ ]:


train[num_non_zero_skewed_features_train_set].head()


# In[ ]:


test[num_non_zero_skewed_features_test_set].head()


# ##### Handle rare categorical features - which are present in less than 1% of the observations.

# In[ ]:


train[categorial_features].head()


# In[ ]:


print(train.shape)
print(test.shape)


# In[ ]:


print(len(categorial_features))
print(len(numerical_features))


# In[ ]:


remaining_features = [feature for feature in train.columns if feature not in categorial_features + numerical_features]
print(remaining_features)


# # Create Dummies

# In[ ]:


train1 = train.copy()
test1 = test.copy()


# In[ ]:


data = pd.concat([train1,test1], axis=0)
train_rows = train1.shape[0]

for feature in categorial_features:
    dummy = pd.get_dummies(data[feature])
    for col_name in dummy.columns:
        dummy.rename(columns={col_name: feature+"_"+col_name}, inplace=True)
    data = pd.concat([data, dummy], axis = 1)
    data.drop([feature], axis = 1, inplace=True)

train1 = data.iloc[:train_rows, :]
test1 = data.iloc[train_rows:, :] 


# In[ ]:


train1.head()


# In[ ]:


test1.head()


# In[ ]:


print("Train",train1.shape)
print("Test",test1.shape)


# # Scaling

# In[ ]:


from sklearn.preprocessing import MinMaxScaler, RobustScaler

scaling_features = [feature for feature in train1.columns if feature not in ['Id', 'SalePrice']]
scaling_features


# In[ ]:


print(len(scaling_features))


# In[ ]:


train1[scaling_features].head()


# In[ ]:


scaler = RobustScaler()
scaler.fit(train1[scaling_features])


# In[ ]:


X_train = scaler.transform(train1[scaling_features])
X_test = scaler.transform(test1[scaling_features])


# In[ ]:


print("Train", X_train.shape)
print("Test", X_test.shape)


# In[ ]:


y_train = train1['SalePrice']


# In[ ]:


X = pd.concat([train1[['Id','SalePrice']].reset_index(drop=True), pd.DataFrame(X_train, columns = scaling_features)], axis =1)
print(X.shape)
X.head()


# # Train Model

# ### Setup cross validation strategy

# In[ ]:


n_folds = 12

def rmsle_cv(model):
    kf = KFold(n_folds, shuffle=True, random_state=42).get_n_splits(X_train)
    rmse= np.sqrt(-cross_val_score(model, X_train, y_train, scoring="neg_mean_squared_error", cv = kf))
    return(rmse)

def rmsle(y_train, y_pred):
    return np.sqrt(mean_squared_error(y_train, y_pred))


# In[ ]:


lasso = Lasso(alpha =0.0005, random_state=0)
elasticNet = ElasticNet(alpha=0.0005, l1_ratio=.9, random_state=0)
kernelRidge = KernelRidge(alpha=0.6, kernel='polynomial', degree=2, coef0=2.5)
svr = SVR(C= 20, epsilon= 0.008, gamma=0.0003)
gradientBoosting = GradientBoostingRegressor(n_estimators=3000, learning_rate=0.05,
                                   max_depth=4, max_features='sqrt',
                                   min_samples_leaf=15, min_samples_split=10, 
                                   loss='huber', random_state =0)
xgb = XGBRegressor(colsample_bytree=0.4603, gamma=0.0468, 
                             learning_rate=0.05, max_depth=3, 
                             min_child_weight=1.7817, n_estimators=2200,
                             reg_alpha=0.4640, reg_lambda=0.8571,
                             subsample=0.5213, silent=1,
                             random_state =0, nthread = -1)
lgbm = LGBMRegressor(objective='regression',num_leaves=5,
                              learning_rate=0.05, n_estimators=720,
                              max_bin = 55, bagging_fraction = 0.8,
                              bagging_freq = 5, feature_fraction = 0.2319,
                              feature_fraction_seed=9, bagging_seed=9,
                              min_data_in_leaf =6, min_sum_hessian_in_leaf = 11, random_state=0)
randomForest = RandomForestRegressor(n_estimators=1200,
                          max_depth=15,
                          min_samples_split=5,
                          min_samples_leaf=5,
                          max_features=None,
                          oob_score=True,
                          random_state=0)


# In[ ]:


scores ={}


# In[ ]:


score = rmsle_cv(lasso)
print("Lasso:: Mean:",score.mean(), " Std:", score.std())
scores['lasso'] = (score.mean(), score.std())
lasso_model = lasso.fit(X_train, y_train)
y_pred_lasso = lasso_model.predict(X_train)
rmsle(y_train,y_pred_lasso)


# In[ ]:


score = rmsle_cv(elasticNet)
print("ElasticNet:: Mean:",score.mean(), " Std:", score.std())
scores['elasticNet'] = (score.mean(), score.std())
elasticNet_model = elasticNet.fit(X_train, y_train)
y_pred_elasticNet = elasticNet_model.predict(X_train)
rmsle(y_train,y_pred_elasticNet)


# In[ ]:


score = rmsle_cv(kernelRidge)
print("KernelRidge:: Mean:",score.mean(), " Std:", score.std())
scores['kernelRidge'] = (score.mean(), score.std())
kernelRidge_model = kernelRidge.fit(X_train, y_train)
y_pred_kernelRidge = kernelRidge_model.predict(X_train)
rmsle(y_train,y_pred_kernelRidge)


# In[ ]:


score = rmsle_cv(svr)
print("SVR:: Mean:",score.mean(), " Std:", score.std())
scores['svr'] = (score.mean(), score.std())
svr_model = svr.fit(X_train, y_train)
y_pred_svr = svr_model.predict(X_train)
rmsle(y_train,y_pred_svr)


# In[ ]:


score = rmsle_cv(gradientBoosting)
print("GradientBoostingRegressor:: Mean:",score.mean(), " Std:", score.std())
scores['gradientBoosting'] = (score.mean(), score.std())
gradientBoosting_model = gradientBoosting.fit(X_train, y_train)
y_pred_gradientBoosting = gradientBoosting_model.predict(X_train)
rmsle(y_train,y_pred_gradientBoosting)


# In[ ]:


score = rmsle_cv(xgb)
print("XGBRegressor:: Mean:",score.mean(), " Std:", score.std())
scores['xgb'] = (score.mean(), score.std())
xgb_model = xgb.fit(X_train, y_train)
y_pred_xgb = xgb_model.predict(X_train)
rmsle(y_train,y_pred_xgb)


# In[ ]:


score = rmsle_cv(lgbm)
print("LGBMRegressor:: Mean:",score.mean(), " Std:", score.std())
scores['lgbm'] = (score.mean(), score.std())
lgbm_model = lgbm.fit(X_train, y_train)
y_pred_lgbm = lgbm_model.predict(X_train)
rmsle(y_train,y_pred_lgbm)


# In[ ]:


score = rmsle_cv(randomForest)
print("RandomForestRegressor:: Mean:",score.mean(), " Std:", score.std())
scores['randomForest'] = (score.mean(), score.std())
randomForest_model = randomForest.fit(X_train, y_train)
y_pred_randomForest = randomForest_model.predict(X_train)
rmsle(y_train,y_pred_randomForest)


# # Stack Models

# In[ ]:


def ensemble_models(X):
    return ((0.1 * lasso_model.predict(X)) +
            (0.1 * elasticNet_model.predict(X)) +
            (0.1 * kernelRidge_model.predict(X)) +
            (0.1 * svr_model.predict(X)) +
            (0.2 * gradientBoosting_model.predict(X)) + 
            (0.1 * xgb_model.predict(X)) +
            (0.2 * lgbm_model.predict(X)) +
            (0.1 * randomForest_model.predict(X)))


# In[ ]:


averaged_score = rmsle(y_train, ensemble_models(X_train))
scores['averaged'] = (averaged_score, 0)
print('RMSLE averaged score on train data:', averaged_score)


# In[ ]:


def stack_models(X):
    return ((0.7 * ensemble_models(X)) +
            (0.15 * lasso_model.predict(X)) +
#             (0.1 * elasticNet_model.predict(X)) +
#             (0.1 * gradientBoosting_model.predict(X)) + 
            (0.15 * xgb_model.predict(X))
#             (0.15 * lgbm_model.predict(X))
           )


# In[ ]:


stacked_score = rmsle(y_train, stack_models(X_train))
scores['stacked'] = (stacked_score, 0)
print('RMSLE stacked score on train data:', stacked_score)


# # Visualize model scores

# In[ ]:


sns.set_style("white")
fig = plt.figure(figsize=(20, 10))

ax = sns.pointplot(x=list(scores.keys()), y=[score for score, _ in scores.values()], markers=['o'], linestyles=['-'])
for i, score in enumerate(scores.values()):
    ax.text(i, score[0] + 0.002, '{:.4f}'.format(score[0]), horizontalalignment='left', size='large', color='black', weight='semibold')

plt.ylabel('Score', size=20, labelpad=12.5)
plt.xlabel('Regression Model', size=20, labelpad=12.5)
plt.tick_params(axis='x', labelsize=13.5)
plt.tick_params(axis='y', labelsize=12.5)
plt.title('Regression Model Scores', size=20)
plt.show()


# In[ ]:


test_predict = np.exp(stack_models(X_test))
print(test_predict[:5])


# In[ ]:


sub = pd.DataFrame()
sub['Id'] = test['Id']
sub['SalePrice'] = test_predict
sub.to_csv('submission.csv',index=False)


# In[ ]:


sub1 = pd.read_csv('submission.csv')
sub1.head()


# **Don't forget to upvote if you like the kernel.**

# In[ ]:




