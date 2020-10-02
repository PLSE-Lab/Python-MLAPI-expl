#!/usr/bin/env python
# coding: utf-8

# # 1. Introduction

# The main goal of this project is to predict a sale price for each Id in the test data of Ames, Iowa housing dataset. To develope the predictive model, training and test dataset of Ames house sale are given. In this project, one can learn identifying the numerical and categorical features, identifing missing data, filling numerical and categorical missing data, the use of pairplot to identify outliers, how to eliminate ouliers from the dataset, checking skewness of the numerical dataset distribution, how to normalize skewed numerical dataset, numerical feature selection using correlation between each feature and between the target SalePrice and the features. In this case, we will use heatmap for visualization. I have consedered models: LinearRegression, Lasso, and Ridge regression models. Moreover, KFold cross-validation, parameter tuning of Ridge and Lasso models, predicting the target value "SalePrice" of the test data, and submiting the final resuls are included. 

# **Note:** At this first run, Lasso and Ridge regressions achieve mean rmse of the training data with 10 fold for cross-validation  ***0.12*** . However, the test dataset rmse is  ***0.139***  and I want to improve this rmse. Any question, comment, suggestion for feature engineering and corrections are welocome !!!!**
# 
# References: This notebook use as reference notebooks from juliencs, Pedro Marcelino, Alexander Papiu, meikegw and Sergne

# ## 1. 1 Import libraries

# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from scipy import stats
from scipy.stats import norm, skew


# # 1.2 Get Data

# Retriving the dataset

# In[ ]:


train = pd.read_csv("../input/train.csv")
test = pd.read_csv("../input/test.csv")


# In[ ]:


train.head()


# In[ ]:


test.head()


# Partial look at the traing and test dataset. As we can see above, our dataset includes NaN (for missing data), numerical and categorical variables

# ### Continues variables

# In[ ]:


train.describe()


# ### Categorical variables

# In[ ]:


train.describe(include = ['O'])


# The Id in the dataset is not important for developing a predictive model. The Id will be saved for  result submission and we will drop Id from the dataset for the preprocessing. 

# In[ ]:


# Drop feature that does not contribute developing the predictive model
test_Id = test['Id']
train.drop('Id', axis = 1, inplace = True)
test.drop('Id', axis = 1, inplace = True)


# In[ ]:


train.shape, test.shape


# # 2. Missing Data

# **show_missing** method given below is a method that extracts the number of missing data from each feature and calculates the percentage of these missing data

# In[ ]:


def show_missing(df):
    num_missing = df.isnull().sum().sort_values(ascending=False)
    missing_percent = (100*df.isnull().sum()/df.isnull().count()).sort_values(ascending=False)
    missing_num_percent = pd.concat([num_missing, missing_percent], axis=1, 
                                    keys=['num_missing', 'missing_percent'])
    return missing_num_percent


# If there is missing data more than __5%__, I am dropping the feature from further consideration in developing the predictive model

# In[ ]:


missing_num_percent = show_missing(train)
train.drop(missing_num_percent[(missing_num_percent['missing_percent'] > 10)].index, axis = 1, inplace = True)
missing_num_percent = show_missing(test)
test.drop(missing_num_percent[(missing_num_percent['missing_percent'] > 10)].index,  axis = 1, inplace = True)


# In[ ]:


train.shape, test.shape


# After dropping the features with missing data more than 5%, we have 68 features for developing the predictive model

# In[ ]:


missing_num_percent = show_missing(train)
missing_num_percent.head(20)


# Now, we have 13 features with less than 10% missing data. The missing data ranges from 0.07% to 5.55%. If we look the missing data, the top 5 missing data are related to the basement. 

# ## 2.1 Transform numerical variables to categorical features

# In[ ]:


train.select_dtypes(exclude = ["object"]).columns


# Transforming some of the numerical features alike into categorical features because they are in reality categorical variables. Example: OverallQual, OverallCond, YearBuilt, YearRemodAdd, Fireplaces

# In[ ]:


train['GarageYrBlt'] = train['GarageYrBlt'].apply(str)
train['OverallQual'] = train['OverallQual'].apply(str)
train['OverallCond'] = train['OverallCond'].apply(str)
train['YearBuilt'] = train['YearBuilt'].apply(str)
train['YearRemodAdd'] = train['YearRemodAdd'].apply(str)
train['Fireplaces'] = train['Fireplaces'].apply(str)


# ## 2.2 Fill missing data

# In[ ]:


# These are categorical features with missing data 
cat_miss_cols =['BsmtExposure', 'BsmtFinType1', 'BsmtFinType2','BsmtQual', 
                'BsmtCond', 'GarageType', 'GarageYrBlt', 'GarageFinish',
                'GarageQual', 'GarageCond', 'MasVnrType' ,'Electrical']


# **explor_max_index** and **impute_missing** methods are methods that extract the most frequent value of a categorical feature and replacing the missing data from the same feature by the most frequent value, respectively. 

# In[ ]:


# Looking at categorical values
def explor_max_index(df, cols):
    df[cols].value_counts().max()
    return df[cols].value_counts().idxmax()


# In[ ]:


def impute_missing(df, cols, value):
    df.loc[df[cols].isnull(), cols] = value


# In[ ]:


# In this for loop the missing data is replaced using the most frequent data 
# of the categorical feature
for item in cat_miss_cols:
    freq_value = explor_max_index(train, item)
    impute_missing(train, item, freq_value)


# In[ ]:


# there was only one numerical feature with missing data and the 
# missing values is replaced with mean value
train['MasVnrArea'].fillna(train['MasVnrArea'].mean(), inplace = True)


# **Check if all missing data are replenished**

# In[ ]:


missing_num_percent = show_missing(train)
missing_num_percent.head()


# # 3. Feature selection

# Feature selection, variable selection, attribute selection or variable subset selection, is a process of selecting relevant features (variables, predictors) to build a machine learning predictive model. Feature selection techniques important for :
# - simplification of predictive model
# - shorter training times,
# -  avoiding the curse of dimensionality,
# - enhancing generalization by reducing overfitting(reduction of variance) (wikipedia)
# 
# To study the data distribution and feature selection using correlation, let us first differentiate the numerical features from categorical features 

# In[ ]:


cat_features = train.select_dtypes(include = ["object"]).columns # categorical features
num_features = train.select_dtypes(exclude = ["object"]).columns # numerical features
#num_features = num_features.drop("SalePrice")
print("All features : " + str(train.shape))
print("Numerical features : " + str(len(num_features)))
print("Categorical features : " + str(len(cat_features)))
train_num = train[num_features]
train_cat = train[cat_features]
print(num_features)


# We have 31 numerical features and 43 categorical features

# ## 3.1 The numerical feature selection

# > - Correlation matrix can be used for feature selection: For visualization, we can use heatmap from seaborn 
# - Then to study the pattern of the relationship between features and the target variable and within each other, we will plot the most correlated features using pairplot (seaborn)

# In[ ]:


colormap = plt.cm.viridis
plt.figure(figsize = (16,16))
plt.title('Correlation between Features', y=1.05, size = 20)
sns.heatmap(train_num.corr(),
            linewidths=0.1, 
            center = 0,
            vmin = -1,
            vmax = 1, 
            annot = True,
            square = True, 
            fmt ='.2f', 
            annot_kws = {'size': 10},
            cmap = colormap, 
            linecolor ='white');


# In[ ]:


# From the heatmap the following features have insignificant correlation with the target value 'SalePrice'
low_corr_coeff = ['MSSubClass', 'BsmtFinSF2', 'LowQualFinSF', 'BsmtHalfBath','BsmtUnfSF', 
                  '3SsnPorch','PoolArea', 'MiscVal', 'MoSold', 'YrSold',
                  'KitchenAbvGr','EnclosedPorch', 'ScreenPorch', 'LotArea','BsmtFinSF1','HalfBath']
# drop the features which have minimal corelation with the target variable
train_num = train_num.drop(train_num[low_corr_coeff], axis = 1)
train_num.shape


# Use heatmap to screenout the remaining features with high correlation to each other. If two features are highly corellated, by considering one of the features we are considering implicitly both. From droping one of them, we wouldnt loss informatution **

# In[ ]:


colormap = plt.cm.viridis
plt.figure(figsize=(16,16))
plt.title('Correlation between Features', y=1.05, size = 20)
sns.heatmap(train_num.corr(),
            linewidths=0.1, 
            center = 0,
            vmin = -1,
            vmax= 1, 
            annot=True,
            square=True, 
            fmt='.2f', 
            annot_kws={'size': 10},
            cmap=colormap, 
            linecolor='white');


# From this heatmap, we can see that TotRmsAbvGrd vs GrLivArea (corr = 0.83), FullBath vs GrLivArea (corr = 0.63),  2ndFlrSF vs GrLivArea (corr = 0.69), GarageCars vs GarageArea (corr = 0.88), BsmtFinSF1 vs BsmtFullBath (corr = 0.65), are highlly correlated. we will use one of the features

# In[ ]:


feature_high_corr_coeff = ['TotRmsAbvGrd', 'FullBath',  '2ndFlrSF', 
                  'GarageCars', 'BsmtFullBath']
train_num = train_num.drop(train_num[feature_high_corr_coeff], axis = 1)
train_num.shape


# The final heatmap

# In[ ]:


colormap = plt.cm.viridis
plt.figure(figsize=(16,16))
plt.title('Correlation between Features', y=1.05, size = 20)
sns.heatmap(train_num.corr(),
            linewidths=0.1, 
            center = 0,
            vmin = -1,
            vmax= 1, 
            annot=True,
            square=True, 
            fmt='.2f', 
            annot_kws={'size': 10},
            cmap=colormap, 
            linecolor='white');


# Now, using the selected features, we can visualize the pattern of the relationship between each feature and with the SalePrice through pairplot as shown below

# In[ ]:


sns.set()
cols = ['MasVnrArea', 'TotalBsmtSF', '1stFlrSF', 'GrLivArea', 'BedroomAbvGr', 'GarageArea', 'WoodDeckSF', 'OpenPorchSF', 'SalePrice']
sns.pairplot(train_num[cols], size = 2.5)
plt.show();


# ## 3.2 Outliers

# As we can see from the pairplot, MasVnrArea (1 data) BsmtFinSF1 (1 data ), TotalBsmtSF (1 data), GrLivArea (2 data ) have outliers. For example, there are two sales with high living room area but less price. The orginal dataset provider (author) recommend to remove houses with greater than 4000 sq ft. We have additional 2 sales which have area > 4000sq.ft but they are inline with the data pattern. We will remove these 5 data entries from each feature. 

# In[ ]:


num_cat = [train_num, train_cat]
train = pd.concat(num_cat, axis = 1)
print(train.shape)

train.drop(train[(train['GrLivArea'] > 4000) & (train['SalePrice'] < 200000)].index, inplace = True)
train.drop(train[(train['MasVnrArea']  > 1500)].index, inplace = True)
train.drop(train[(train['TotalBsmtSF'] > 5000)].index, inplace = True)

train_num = train[cols]
train_cat = train[cat_features] ;


# ***Note: *** While I was working with the numerical data, I have observed that when an outlier such as a data that satisfy 'GrLivArea' > 4000 is droped, the whole observation is eliminated and hence it results in a miss match between the number of observation in the categorical dataset and the numarical data. Therfore, to avoid this miss match, I have combined them first followed by dropping the outliers.
# 
# 
# 

# In[ ]:


train_num.shape, train_cat.shape


# To look the data pattern after eliminating the outlier data, let us plot those features affected by the outliers using the pairplot.

# In[ ]:


sns.set()
cols = ['GrLivArea', 'MasVnrArea', 'TotalBsmtSF','SalePrice']
sns.pairplot(train_num[cols], size = 2.5)
plt.show();


# The outiers are eliminated from the dataset

# # 4. Numerical data distribution

# ## 4.1 Target variable distribution

# We use distplot from seaborn to plot the distributuin of the **target variable: SalePrice** 

# In[ ]:


sns.distplot(train_num['SalePrice']);


# From the figure, we can see that the sale price distribution doesnt fellow a normal distributiion and it is positive (right) skewed. Skewness is a measure of symmetry. skewed mean it lacks symmetry. Kurtosis is a measure whether the data distribution is right-tailed or left-tailed relative to the normal distribution. -ve values for skewness shows the data distribution is skewed to left and +ve value indicate skewd to right.   

# In[ ]:


print("Skewness of SalePrice Distribution is : {:.4f}"
      .format(train['SalePrice'].skew()))

print("Kurtosis of SalePrice Distribution is : {:.4f}"
      .format(train['SalePrice'].kurt()))


# SalePrice is skewed to the right. heavy tailed towards right. For normal probability plot, the data distribution should fellow the diagonal which it represents the normal distibution. Let us check the SalePrice data distribution

# In[ ]:


sns.distplot(train_num['SalePrice'], fit = norm);
fig = plt.figure()
res = stats.probplot(train_num['SalePrice'], plot = plt);


# SalePrice doesnt fellow normal distribution. It shows peakedness, postive skewnes and it doesnt fellow the diagonal line from the probplot . The linear models are effective if the data is normally distributed. Therefor, we need to trasform the SalePrice data into normally distributed. One method that we can use to transform the skewed Saleprice distribution to normal distribution is the log transformation 

# In[ ]:


train_num["SalePrice"] = np.log(train_num["SalePrice"]) 
# check the new distribution 
sns.distplot(train_num['SalePrice'], fit = norm);
fig = plt.figure()
res =stats.probplot(train_num['SalePrice'], plot = plt);


# As we can see from the figure, now the data appears to fellow normal distribution. The skewness is corrected but not 100%

# ## 4.2 Skewness for the  rest of the features

# In[ ]:


y_train = train_num['SalePrice']
num_featurs = train_num.columns.drop("SalePrice")
train_num = train_num[num_featurs]
train_num.info()


# In[ ]:


skewed_features = train_num[num_featurs].apply(lambda x: skew(x)).sort_values(ascending = False)
print('skewness of the numerical features of the training data: \n')
skewness = pd.DataFrame({'skew' : skewed_features})
skewness


# As we can see from the table, the features are positively skewed i.e, skewed to the right. However, most of the features are soft skewed. To transoform these skewd features to normal distribution, we can use the box-cox power transformation because some of the features data includ zero values. 

# In[ ]:


from scipy.special import boxcox1p


# In[ ]:


lam = 0.25
skewed_features = ['MasVnrArea', 'OpenPorchSF', 'WoodDeckSF', 'GrLivArea', '1stFlrSF', 'TotalBsmtSF'] 
# these are the features that have morethan 0.75 skewness
for features in skewed_features:
    train_num[features] = boxcox1p(train_num[features], lam)
  


# In[ ]:


train_num_cat = [train_num, train_cat]
train = pd.concat(train_num_cat, axis = 1)
train.shape


# # 5 Preprocessing of the test dataset

# We dealt with the training data preprocessing and we will fellow similar procedure for the test dataset. The number of numerical and categorical features will be same.

# In[ ]:


train_cat.columns # the test dataset will have similar features


# In[ ]:


test_cat_features = ['MSZoning', 'Street', 'LotShape', 'LandContour', 'Utilities',
       'LotConfig', 'LandSlope', 'Neighborhood', 'Condition1', 'Condition2',
       'BldgType', 'HouseStyle', 'OverallQual', 'OverallCond', 'YearBuilt',
       'YearRemodAdd', 'RoofStyle', 'RoofMatl', 'Exterior1st', 'Exterior2nd',
       'MasVnrType', 'ExterQual', 'ExterCond', 'Foundation', 'BsmtQual',
       'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2', 'Heating',
       'HeatingQC', 'CentralAir', 'Electrical', 'KitchenQual', 'Functional',
       'Fireplaces', 'GarageType', 'GarageYrBlt', 'GarageFinish', 'GarageQual',
       'GarageCond', 'PavedDrive', 'SaleType', 'SaleCondition']


# 'OverallQual', 'OverallCond', 'YearBuilt', 'YearRemodAdd', and 'Fireplaces' are categorical features but they are represented  as numerical features in the orginal dataset. So, let us first transofrm these features to categorical features

# In[ ]:


test['GarageYrBlt'] = test['GarageYrBlt'].apply(str)
test['OverallQual'] = test['OverallQual'].apply(str)
test['OverallCond'] = test['OverallCond'].apply(str)
test['YearBuilt'] = test['YearBuilt'].apply(str)
test['YearRemodAdd'] = test['YearRemodAdd'].apply(str)
test['Fireplaces'] = test['Fireplaces'].apply(str)


# The numerical features are similar to the training dataset numerical features

# In[ ]:


train_num.columns # the test dataset will have similar features


# In[ ]:


test_num_features = ['MasVnrArea', 'TotalBsmtSF', '1stFlrSF', 
                     'GrLivArea', 'BedroomAbvGr','GarageArea', 
                     'WoodDeckSF', 'OpenPorchSF']


# In[ ]:


test_cat = test[test_cat_features]
test_num = test[test_num_features]


# In[ ]:


test_cat.shape, test_num.shape


# ## 5.1  Filling missing value of test dataset 

# Starting with categorical features fellowed by filling the numerical missing data

# In[ ]:


# Determine the number of missing data and their percentage of the categorical features
missing_num_percent = show_missing(test_cat)
missing_num_percent.head(20)


# In[ ]:


# these features are features with missing data
cat_miss_cols =['GarageCond','GarageQual', 'GarageFinish', 'GarageType', 'MSZoning',
                'MasVnrType', 'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1',
                'BsmtFinType2', 'KitchenQual','Functional', 'SaleType', 'Utilities', 
                'Exterior2nd', 'Exterior1st' ]


# The missing data of the categorical features are replaced by the most frequent feature in the fellowing for loop

# In[ ]:


for item in cat_miss_cols:
    freq_value = explor_max_index(test_cat, item)
    impute_missing(test_cat, item, freq_value);


# Similary the missing data of the numerical features are replaced by the mean of in the fellowing cell

# In[ ]:


# first determine the number of missing data and their percentage
missing_num_percent = show_missing(test_num)
missing_num_percent.head()


# In[ ]:


test_num['MasVnrArea'].dtype, test_num['TotalBsmtSF'].dtype, test_num['GarageArea'].dtype 


# Three of the features with missing value are floating data type

# In[ ]:


test_num['MasVnrArea'].fillna(test_num['MasVnrArea'].mean(), inplace = True)
test_num['TotalBsmtSF'].fillna(test_num['TotalBsmtSF'].mean(), inplace = True)
test_num['GarageArea'].fillna(test_num['GarageArea'].mean(), inplace = True);


# **Now let us check if we have missing data from the test dataset**

# In[ ]:


missing_num_percent = show_missing(test_num)
missing_num_percent.head()


# In[ ]:


missing_num_percent = show_missing(test_cat)
missing_num_percent.head()


# ## 5.2 Skewness of the Test Dataset

# In[ ]:


skewed_features = test_num[test_num_features].apply(lambda x: skew(x)).sort_values(ascending = False)
print('skewness of the numerical features of the test data: \n')
skewness = pd.DataFrame({'skew' : skewed_features})
skewness


# **Transforming the numerical test data features with skewness to normal distribution****

# In[ ]:


lam = 0.25
skewed_features = ['MasVnrArea', 'OpenPorchSF', 'WoodDeckSF', 'GrLivArea', '1stFlrSF', 'TotalBsmtSF']
for features in skewed_features:
    test_num[features] = boxcox1p(test_num[features], lam)


# In[ ]:


test_num_cat = [test_num, test_cat]
test = pd.concat(test_num_cat, axis = 1)
test.shape


# ## 5.3 Get Dummies

# In[ ]:


n_train_observations = train.shape[0]
n_test_observations = test.shape[0]
train_test_data = pd.concat((train, test)).reset_index(drop=True)
print(" The combined dataset size is : {}".format(train_test_data.shape))


# In[ ]:


train_test_data_dummies = pd.get_dummies(train_test_data)


# In[ ]:


train_test_data_dummies.shape


# # 6. Model Selection,  Parameter tunning and Propose the model

# **split data into training and test set**

# In[ ]:


train = train_test_data_dummies[:n_train_observations]
test = train_test_data_dummies[n_train_observations:]
y_train = y_train
X_train = train
X_test = test

X_train.shape,  y_train.shape,  X_test.shape


# ## 6.1 Import Libraries of Machine Learning Algorithms

# In[ ]:


from sklearn.linear_model import LinearRegression,Ridge, ElasticNet, Lasso,LassoCV, BayesianRidge, LassoLarsIC
from sklearn.ensemble import RandomForestRegressor,  GradientBoostingRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin, clone
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.grid_search import GridSearchCV
import xgboost as xgb;


# In[ ]:


#Validation function
n_folds = 10
def RMSLE(model):
    kf = KFold(n_folds, shuffle=True, random_state=42).get_n_splits(X_train.values)
    rmse= np.sqrt(-cross_val_score(model, X_train.values, y_train, scoring="neg_mean_squared_error", cv = kf))
    return rmse


# ## 6.2 Linear Regression

# In[ ]:


lr = LinearRegression()
lr.fit(X_train, y_train)
y_pred_lr = np.exp(lr.predict(X_test))
lr_score = lr.score(X_train, y_train)*100
rmse = RMSLE(lr).mean()
print("Linear regression model accuracy score = {:.2f} and  mean rmse = {:.3f}"
          .format(lr_score, rmse))


# ## 6.3 Ridge Regression

# The main regularization parameter to balance the overfitting and underfitting of our ridge model is alpha. 

# ##### alpha parameter tunning

# In[ ]:


alphas = [0.001, 0.01, 0.1, 1, 5, 10, 20, 50, 75, 100]
for alpha in alphas:
    rg = Ridge(alpha = alpha)
    rg.fit(X_train, y_train)
    rg_score = rg.score(X_train, y_train)
    rmse = RMSLE(rg).mean()
    print("For alpha = {} the Ridge model accuracy score = {:.2f} and  mean rmse = {:.3f}"
          .format(alpha, rg_score*100, rmse))


# In[ ]:


alpha = 10
rg = Ridge(alpha = alpha)
rg.fit(X_train, y_train)
y_pred_rg = np.exp(rg.predict(X_test))
rg_score = rg.score(X_train, y_train)
rmse = RMSLE(rg).mean()
print("For alpha = {} the Ridge model accuracy score = {:.2f} and  mean rmse = {:.3f}"
          .format(alpha, rg_score*100, rmse))
print(y_pred_rg[:10])


# ## 6.4 Lasso Regression

# In[ ]:


lasr = Lasso()
lasr.fit(X_train, y_train)
y_pred_lasr = np.exp(lasr.predict(X_test))
lasr_score = lasr.score(X_train, y_train)
print("Lasso Accuracy  score = {:.2f}".format(lasr_score*100))
RMSLE(lasr)


# ##### alpha parameter tunning

# In[ ]:


alpha = [0.001, 0.005, 0.01, 0.015, 0.1, 0.5, 1, 5, 10, 20, 50, 75, 100]
lasCV = LassoCV(alphas = alpha, cv = n_folds)
lasCV.fit(X_train, y_train)
lasCV_score = lasCV.score(X_train, y_train)
plt.plot(lasCV.alphas_, lasCV.mse_path_)
plt.title('LassoCV')
plt.xlabel('alphas')
plt.ylabel(' Mean square error for test set in each kfold (mse_path)')
print(lasCV_score*100)
print(RMSLE(lasCV))


# alpha = 0.001 gives a better accuracy which is 92.11% and the root-mean square error (RMSE) is 0.12.

# In[ ]:


alpha = 0.001
lasr = Lasso(alpha = alpha)
lasr.fit(X_train, y_train)
y_pred_lasr = np.exp(lasr.predict(X_test))
lasr_score = lasr.score(X_train, y_train)
print("Ridge Accuracy score = {:.2f} at alpha = {}".format(lasr_score*100, alpha))
print("RMSE = {} ".format(RMSLE(lasr)))
coef = pd.Series(lasr.coef_, index = X_train.columns)
print("Lasso selected " + str(sum(coef !=0)) + ' features and eliminates ' + str(sum(coef == 0)) + " features")


# In[ ]:


print(y_pred_lasr[:10])


# ## 6.5 Result submission 

# In[ ]:


submission = pd.DataFrame(data= {'Id' : test_Id, 'SalePrice': y_pred_lasr})


# In[ ]:


submission.shape


# In[ ]:


submission.to_csv('submission_Lasso.csv', index=False)


# In[ ]:


import xgboost as xgb
Xgbreg = xgb.XGBRegressor(
                 colsample_bytree=0.2,
                 gamma=0.0,
                 learning_rate=0.01,
                 max_depth=4,
                 min_child_weight= 1.5,
                 n_estimators=6000,                                                                  
                 reg_alpha=0.9,
                 reg_lambda=0.6,
                 subsample=0.2,
                 random_state=42,
                 silent=1)

Xgbreg.fit(X_train, y_train)

y_pred_Xgbreg = np.exp(Xgbreg.predict(X_test))
Xgbreg_score = Xgbreg.score(X_train, y_train)
print("XGBoost Accuracy  score = {:.2f}".format(Xgbreg_score*100))
RMSLE(Xgbreg)


# In[ ]:


rmse = RMSLE(Xgbreg).mean()
print(rmse)


# In[ ]:


submission = pd.DataFrame(data= {'Id' : test_Id, 'SalePrice': y_pred_Xgbreg})
submission.to_csv('submission_XGBoost.csv', index=False)

