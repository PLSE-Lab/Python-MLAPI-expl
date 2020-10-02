#!/usr/bin/env python
# coding: utf-8

# # Abstract
# 
# I conduct a machine learning experiment on the Ames Housing dataset. Through an iteration of experiments with different preprocessing methods and models, I achieve the final mean r2 score of 0.8745 and the final RMSLE score of 0.11812, ranking at top 20%.

# # [](http://)Imports 

# In[ ]:


#Core packages
import pandas as pd
import numpy as np

#Data Profiling
import pandas_profiling as pp

#Visualisation
import matplotlib.pyplot as plt
import seaborn as sns

#sklearn
import category_encoders as ce
from sklearn.preprocessing import PowerTransformer, RobustScaler
from sklearn import model_selection
from sklearn.linear_model import LinearRegression, ElasticNet,ElasticNetCV

#XGBoost
from xgboost import XGBRegressor


# # Data
# 

# In[ ]:


train = pd.read_csv("../input/house-prices-advanced-regression-techniques/train.csv")
test = pd.read_csv("../input/house-prices-advanced-regression-techniques/test.csv")

print(test.shape)
print(train.shape)


# # Exploratory Data Analysis & Preprocessing

# First, I use the `pandas_profiling` package to have an overview of the data, checking for their types

# In[ ]:


featuresDF = train.drop(['Id','SalePrice'],axis=1)   
featuresDF.profile_report(style={'full_width':True})


# Through this overview, I notice that many features have a large amount of missing values. On careful examination, it comes into light that some of the missing values actually mean `None` and `0` for their respective categorial and numerical features. I would fill these NAs with such values accordingly.
# 
# Some of the features are highly skewed, which will require power transforming.
# 
# The `MSSubClass` is numerical, even though it is supposed to be categorical, and will be transformed accordingly.

# ## Filling Missing Values

# ### Categorical
# 
# As mentioned above, some categorical features have missing values that mean `None`. I will fill these NAs with such value. Other missing values will be filled with the mode of their respective features. I fill the missing values of the test set with the mode of the training set to prevent information leakage.

# In[ ]:


# Filling 'None' for missing values
for col in ('Alley','Utilities','MasVnrType','BsmtQual','BsmtCond','BsmtExposure',
            'BsmtFinType1','BsmtFinType2','Electrical','FireplaceQu','GarageType',
            'GarageFinish','GarageQual','GarageCond','PoolQC','Fence','MiscFeature'):
    train.loc[:,col] = train.loc[:,col].fillna('None')
    test.loc[:,col] = test.loc[:,col].fillna('None')

# Filling the mode value for the features actual missing values
for col in ('MSZoning','Exterior1st','Exterior2nd','KitchenQual','SaleType','Functional'):
    train.loc[:,col] = train.loc[:,col].fillna(train.loc[:,col].mode()[0])
    test.loc[:,col] = test.loc[:,col].fillna(train.loc[:,col].mode()[0])


# ### Numerical
# 
# Similarly, some numerical features have missing values that mean `0`. I will perform the same process as with the cateogrical features.

# In[ ]:


# Filling '0' for missing values
for col in ('2ndFlrSF','MasVnrArea','BsmtFinSF1','BsmtFinSF2','BsmtUnfSF',
            'BsmtHalfBath','GarageCars','TotalBsmtSF', 'BsmtFullBath'):
    train.loc[:,col] = train.loc[:,col].fillna(0)
    test.loc[:,col] = test.loc[:,col].fillna(0)

# Filling the mean value for the missing values
train.loc[:,'LotFrontage'] = train.loc[:,'LotFrontage'].fillna(train.loc[:,'LotFrontage'].mean())
test.loc[:,'LotFrontage'] = test.loc[:,'LotFrontage'].fillna(train.loc[:,'LotFrontage'].mean())


# ## Removing Outliers
# 
# There are outliers as identified by the author of the dataset. These outliers are values of the feature `GrLivArea` that are larger than 4000 in the training set. I will remove those outliers.

# In[ ]:


train = train.loc[train.loc[:,'GrLivArea']<4000,:]
idx_train = train.shape[0] #keep track of the training inde
idx_train


# ## Correlation Heatmap
# 
# After filling the missing values, I will graph a heatmap to observe the correlation between every pair of features. After obtaining the pairs of features with high correlation, I would remove the feature with lower correlation to our training output in those pairs. Excluding highly correlated features from our model will prevent overfitting.

# In[ ]:


corrMatrix = train.drop('Id',axis=1).corr()
plt.figure(figsize=[30,15])
heatmap = sns.heatmap(corrMatrix, annot=True, cmap='Blues')


# From the heatmap, I identify pairs of features with high correlation (>.65): 
# `[(GarageArea, GarageCars) (1stFlrSF,TotalBstmSF), (TotRmsAbvGrd, GrLivArea), (TotRmsAbvGr, BedroomAbvGrd),
# (GarageYrBuilt, YearBuilt), (2ndFlrSF,GrLivArea), (BsmtFullBath, BsmtFinSF1)]`
# 
# From these pairs, I remove the features with lower correlation to `SalePrice`

# From the heatmap, I identify pairs of features with high correlation (>.80): 
# `[(GarageArea, GarageCars) (1stFlrSF,TotalBstmSF), (TotRmsAbvGrd, GrLivArea), (GarageYrBuilt, YearBuilt)]`
# 
# From these pairs, I remove the features with lower correlation to `SalePrice`

# In[ ]:


df = pd.concat([train,test], sort=False)
df.drop(['TotRmsAbvGrd','GarageArea', 'GarageYrBlt', '1stFlrSF'], axis=1, inplace=True)
df.shape


# ## Transformation
# 

# As mentioned above, `MSSubClass` feature needs to be transformed into categorical type

# In[ ]:


# transforming MSSubClass to category type
df.loc[:,'MSSubClass'] = df.loc[:,'MSSubClass'].astype('category')
print(df.loc[:,'MSSubClass'].dtype)

#splitting the dataset in test id, the training set and the testing set
test_id = df.iloc[idx_train:,0]
df.drop(['Id'], axis=1, inplace=True)

train = df.iloc[:idx_train,:]
test = df.iloc[idx_train:,:]

print(test.shape)
print(train.shape)


# In[ ]:


#saving predictor columns and label column
predictors = list(df.columns.drop(['SalePrice']))
label = 'SalePrice'

#splitting the dataset
X_train = df.iloc[:idx_train, :].loc[:, predictors]
y_train = df.iloc[:idx_train, :].loc[:, label]
X_test = df.iloc[idx_train:, :].loc[:, predictors]

print(X_train.shape)
print(y_train.shape)
print(X_test.shape)


# Some numerical features are highly skewed. I will use a Jeo-Johnson `PowerTransformer` to transform these features into a more normal distribution.

# In[ ]:


#obtaining numerical features with absolute skewness > .5
skewed_cols_bool = np.absolute(df.select_dtypes(include=['int','float']).skew(axis = 0)) > .5
skewed_cols = skewed_cols_bool.loc[skewed_cols_bool].index.drop('SalePrice').tolist()

#applying a power transformer to skweded cols
pt = PowerTransformer()
X_train.loc[:,skewed_cols] = pt.fit_transform(X_train.loc[:,skewed_cols])
X_test.loc[:,skewed_cols] = pt.transform(X_test.loc[:,skewed_cols])

# print(X_test.shape)
# print(X_train.shape)
# print(X_train.head())
# print(skewed_cols)


# Transforming all categorical features into one-hot encoding

# In[ ]:


#one-hot encoding
df = pd.concat([X_train,X_test], sort=False)
df = pd.get_dummies(df)

X_train = df.iloc[:idx_train, :]
X_test = df.iloc[idx_train:, :]

print(df.shape)
print(X_train.shape)
print(X_test.shape)


# Normalizing all features using `RobustScaler` will hypothetically handle any remaining outliers that are not captured in the process

# In[ ]:


scaler=RobustScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

print(X_train.shape)
print(X_test.shape)


# Log transforming the label is empirically proven to provide a better relationship between the output and the predictors in this dataset

# In[ ]:


#log transforming the label
temp = np.log1p(y_train) # creating a temp series to avoid chained indexing
y_train = temp


# # Model

# I choose to use an `ElasticNetCV` model for this experiment. I set the `l1_ratio` to be `[.1, .3, .5, .7, .8, .85, .9, .95, .99, 1]`, with most values closer to 1.0 so that the model is closer to a Lasso model. I also do a 10-fold CV to prevent overfitting. Finally, I transform the predictions exponentially to return its final results.

# In[ ]:


elasticNet = ElasticNetCV(l1_ratio = [.1, .3, .5, .7, .8, .85, .9, .95, .99, 1], cv=10, n_jobs=-1)
elasticNet.fit(X_train, y_train)
predictions = elasticNet.predict(X_test)
predictionsExp = np.exp(predictions)

predictionsExp.shape


# In[ ]:


submission = pd.DataFrame({
    "Id": test_id,
    "SalePrice": predictionsExp
})
submission.to_csv('submission_ElasticNetCVFinal.csv', index=False)

# Prepare CSV
print(submission.head())

