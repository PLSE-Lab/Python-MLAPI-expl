#!/usr/bin/env python
# coding: utf-8

# ## GUIDE FOR REGRESSION PROBLEMS

# In this notebook I will solve a regression problem. **The goal for this notebook is to give a idea of the steps we need to follow to solve a regression problem** and to show some interesting techniques to do it.

# The problem we will be solving is this:  [House Prices: Advanced Regression Techniques](https://www.kaggle.com/c/house-prices-advanced-regression-techniques/overview)

# In this notebook I put together different concepts I found in the following notebooks. Kudos to these guys for the great work:
# * [Stacked Regressions : Top 4% on LeaderBoard](https://www.kaggle.com/serigne/stacked-regressions-top-4-on-leaderboard)
# * [Simple approach to predict house price](https://www.kaggle.com/shubhammahajan3110/simple-approach-to-predict-house-price)
# * [XGBoost + Lasso](https://www.kaggle.com/humananalog/xgboost-lasso)
# 

# Any comments or suggestions are very welcome.

# Ok, we begin by importing a set of packages we'll be needing:

# In[ ]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")


# Importing the data:

# In[ ]:


df_train = pd.read_csv('../input/house-prices-advanced-regression-techniques/train.csv')
df_test = pd.read_csv('../input/house-prices-advanced-regression-techniques/test.csv')


# # STEP 0 : EXPLORE TARGET VARIABLE

# First of all, let's analyze the target column 'SalePrice'

# In[ ]:


sns.distplot(df_train.SalePrice)


# In[ ]:


sns.boxplot(df_train['SalePrice'])


# We can see that the distribution is skewed to the right (positive skew). Let's take a look at the [skewness and kurtosis](https://codeburst.io/2-important-statistics-terms-you-need-to-know-in-data-science-skewness-and-kurtosis-388fef94eeaa):

# In[ ]:


print('Skewness: ',df_train['SalePrice'].skew())
print('Kurtosis: ',df_train['SalePrice'].kurt())


# In order to address this problem we can apply a [Box-Cox](https://blog.minitab.com/blog/applying-statistics-in-quality-projects/how-could-you-benefit-from-a-box-cox-transformation) transformation. In this case I will use a Box-Cox with lambda= 0 wich is basically a logarithmic transformation:

# In[ ]:


df_train['SalePrice']=np.log1p(df_train['SalePrice'])

sns.distplot(df_train['SalePrice'])


# We find a couple of outliers when we plot SalePrice against the top 2 most correlated feature, 'GrLivArea'. Let's plot them and drop those rows

# In[ ]:


fig, ax = plt.subplots()
ax.scatter(x = df_train['GrLivArea'], y = df_train['SalePrice'])
plt.ylabel('SalePrice', fontsize=13)
plt.xlabel('GrLivArea', fontsize=13)
plt.show()


# In[ ]:


df_train.drop(df_train[df_train['GrLivArea']>4000].index, inplace=True)


# Now that we have tranformed the target variable, we can put together train a test data

# In[ ]:


ntrain = df_train.shape[0] #we do this because we are going to concatenate train and test and we will need this later
ntest = df_test.shape[0] 

y_train = df_train.SalePrice.values

df_all = pd.concat((df_train, df_test)).reset_index(drop=True)
df_all.drop(['SalePrice'], axis=1, inplace=True)

df=df_all


# # STEP 1 : EXPLORATORY DATA ANALYSIS (EDA)

# Overview of the dataset 

# In[ ]:


def overview(df):
    print('SHAPE: ', df.shape)
    print('columns: ', df.columns.tolist())
    col_nan=df.columns[df.isnull().any()].tolist()
    print('columns with missing data: ',df[col_nan].isnull().sum())

overview(df)


# It looks like we have a lot of columns with missing data. Let's rank the variables with the hisghest % of missing data

# In[ ]:


all_data_na = (df.isnull().sum() / len(df)) * 100
all_data_na = all_data_na.drop(all_data_na[all_data_na == 0].index).sort_values(ascending=False)[:30]
missing_data = pd.DataFrame({'Missing Ratio' :all_data_na})
missing_data


# Let's see how variables correlate with each other

# In[ ]:


corrmat = df_train.corr()
corrmat['SalePrice'].sort_values(ascending=False)


# Let's take a look at the 10 variables that are more correlated to the target

# In[ ]:


import matplotlib.pyplot as plt#visualization
k = 10 #number of variables for heatmap
cols = corrmat.nlargest(k, 'SalePrice')['SalePrice'].index
cm = np.corrcoef(df_train[cols].values.T)
plt.figure(figsize=(10,10))
sns.set(font_scale=1.25)
hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 15}, yticklabels=cols.values, xticklabels=cols.values)
plt.show()


# Let's take a look at the values of all the **categorical** variables

# In[ ]:


cat_cols = df.select_dtypes('object').columns.tolist()

for i in cat_cols:
    print(df[i].value_counts())


# Let's take a look at the values of all the **numerical** variables

# In[ ]:


int_cols =  df.select_dtypes(['int64','float64']).columns.tolist()
df[int_cols].describe().T


# # STEP 2 : DATA PREPROCESSING

# Looking at the data description of the data we see that most of the missing values are not really missin values, but actual null categories or zeros (for categorical and numerical features). **For example, PoolQC, is the Pool Quality, so having a missing value here means that the house doesn't have a pool, so we can fill the missing values with 'None'**. 
# * **Categorical (not really missing data)**: fill missing values with **'None'**
# * **Categorical (actually missing data)**: fill missing values with **mode**
# * **Numerical**: fill missing values with **0**
# 
# There are a couple of exceptions.

# In[ ]:


def handling_missing(df):
    cols_none=['PoolQC','MiscFeature','Alley','Fence','FireplaceQu','GarageFinish','GarageQual','GarageCond',
               'GarageType','BsmtExposure','BsmtCond','BsmtQual','BsmtFinType2','BsmtFinType1','Exterior2nd',
               'Exterior1st']
    for i in cols_none:
        df[i] = df[i].fillna('None')
    
    cols_zero=['GarageYrBlt','BsmtHalfBath','BsmtFullBath','MasVnrArea','TotalBsmtSF','BsmtFinSF2','BsmtFinSF1',
               'BsmtUnfSF']
    for i in cols_zero:
        df[i] = df[i].fillna(0)
    
    cols_mode=['MasVnrType','MSZoning','Utilities','SaleType','GarageArea','GarageCars','KitchenQual','Electrical']
    for i in cols_mode:
        df[i] = df[i].fillna(df[i].mode()[0])
    
    df["Functional"] = df["Functional"].fillna("Typ") #tells you to do this in the data description
    
    df['LotFrontage'] = df['LotFrontage'].fillna(df['LotFrontage'].mean()) 
    
    df=df.drop(['Id'],axis=1) # Let's drop the Id column while we are at it
    
    return df


# In[ ]:


df=handling_missing(df)


# Quality measurements are stored as text but we can convert them to numbers where a higher number means higher quality.

# In[ ]:


qual_dict = {'None': 0, "Po": 1, "Fa": 2, "TA": 3, "Gd": 4, "Ex": 5}

df["ExterQual"] = df["ExterQual"].map(qual_dict).astype(int)
df["ExterCond"] = df["ExterCond"].map(qual_dict).astype(int)
df["BsmtQual"] = df["BsmtQual"].map(qual_dict).astype(int)
df["BsmtCond"] = df["BsmtCond"].map(qual_dict).astype(int)
df["HeatingQC"] = df["HeatingQC"].map(qual_dict).astype(int)
df["KitchenQual"] = df["KitchenQual"].map(qual_dict).astype(int)
df["FireplaceQu"] = df["FireplaceQu"].map(qual_dict).astype(int)
df["GarageQual"] = df["GarageQual"].map(qual_dict).astype(int)
df["GarageCond"] = df["GarageCond"].map(qual_dict).astype(int)


# We split the Year features into 7 groups of 20 years

# In[ ]:


# Divide up the years between 1871 and 2010 in slices of 20 years.
year_map = pd.concat(pd.Series("YearBin" + str(i+1), index=range(1871+i*20,1891+i*20)) for i in range(0, 7))

df['YearBuilt']=df['YearBuilt'].map(year_map)
df['YearRemodAdd']=df['YearRemodAdd'].map(year_map)
df['GarageYrBlt']=df['GarageYrBlt'].map(year_map)
df['YrSold']=df['YrSold'].map(year_map)


# These two variables are categorical instead of numeric

# In[ ]:


cols_numcat=['MSSubClass','MoSold']

for i in cols_numcat:
    df[i]=df[i].astype('object')


# We apply a logarithmic transformation to numeric features with high skewness

# In[ ]:


from scipy.stats import skew

numeric_features = df.dtypes[df.dtypes != "object"].index

numeric_df=df[numeric_features]
  
skewed = df[numeric_features].apply(lambda x: skew(x.dropna().astype(float)))
skewed = skewed[skewed > 0.75]
skewed = skewed.index

df[skewed]=np.log1p(df[skewed])


# We normalize the numeric variables

# In[ ]:


from sklearn.preprocessing import StandardScaler

std=StandardScaler()
scaled=std.fit_transform(df[numeric_features])
scaled=pd.DataFrame(scaled,columns=numeric_features)

df_original=df.copy()
df=df.drop(numeric_features,axis=1)

df=df.merge(scaled,left_index=True,right_index=True,how='left')


# We encode the categorical variables 

# In[ ]:


df=pd.get_dummies(df)


# # STEP 3 : MODELLING

# In[ ]:


train = df[:ntrain]
test = df[ntrain:]


# In[ ]:


#this is the metric we use to validate the model
from sklearn.metrics import mean_squared_error
def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))


# After trying out varios options, this one seems to get the best result. It's a **blend of 3 different models**

# In[ ]:


from sklearn.linear_model import ElasticNetCV, LassoCV, RidgeCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import KFold, cross_val_score
import xgboost as xgb

alphas_ridge = [14.5, 14.6, 14.7, 14.8, 14.9, 15, 15.1, 15.2, 15.3, 15.4, 15.5]
alphas_lasso = [5e-05, 0.0001, 0.0002, 0.0003, 0.0004, 0.0005, 0.0006, 0.0007, 0.0008]

kfolds = KFold(n_splits=10, shuffle=True, random_state=42)

# Kernel Ridge Regression : made robust to outliers
ridge = make_pipeline(RobustScaler(), RidgeCV(alphas=alphas_ridge, cv=kfolds))

# LASSO Regression : made robust to outliers
lasso = make_pipeline(RobustScaler(), LassoCV(max_iter=1e7, 
                    alphas=alphas_lasso,random_state=42, cv=kfolds))

# XGBOOST Regressor : The parameters of this model I took from another notebook
regr = xgb.XGBRegressor(
                 colsample_bytree=0.2,
                 gamma=0.0,
                 learning_rate=0.01,
                 max_depth=4,
                 min_child_weight=1.5,
                 n_estimators=7200,                                                                  
                 reg_alpha=0.9,
                 reg_lambda=0.6,
                 subsample=0.2,
                 seed=42,
                 silent=1)


lasso_model = lasso.fit(train,y_train)
ridge_model = ridge.fit(train,y_train)
xgb_model = regr.fit(train,y_train)

# model blending function using fitted models to make predictions
def blend_models(X):
    return ((xgb_model.predict(X)) + (lasso_model.predict(X)) + (ridge_model.predict(X)))/3

y_pred=blend_models(train)

print("blend score on training set: ", rmse(y_train, y_pred))


# # STEP 4 : PREPARING THE SUBMISSION FILE

# In[ ]:


y_pred_blend = blend_models(test)
y_pred_exp_blend = np.exp(y_pred_blend)

pred_df_blend = pd.DataFrame(y_pred_exp_blend, index=df_test["Id"], columns=["SalePrice"])
pred_df_blend.to_csv('output.csv', header=True, index_label='Id')


# It doesnt perform as well on the test data so there's some overfitting to fix. It should get you around 0.12
