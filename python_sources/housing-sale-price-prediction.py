#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import warnings
warnings.filterwarnings("ignore")


# In[ ]:


import os
print(os.listdir("../input"))


# In[ ]:


train_df = pd.read_csv("../input/train.csv")
test_df = pd.read_csv("../input/test.csv")


# In[ ]:


train_df.head()


# In[ ]:


sale_price = train_df["SalePrice"]


# In[ ]:


test_df.head()


# In[ ]:


train_df.shape, test_df.shape


# In[ ]:


train_df.info()


# In[ ]:


test_df.info()


# In[ ]:


train_df.describe(include="all") # include both categorical and numerical features


# In[ ]:


train_df.drop("Id", axis =1,inplace = True)
test_id = test_df["Id"]
test_df.drop("Id", axis =1,inplace = True)


# In[ ]:


train_df.shape,test_df.shape


# # Analysis of Target Variable:

# In[ ]:


from scipy import stats


# In[ ]:


plt.subplots(figsize=(12,9))
sns.distplot(train_df['SalePrice'], fit=stats.norm)


# In the above plot we can see that the sale price is positively skewed. Let's apply log transformation to make it more normal

# In[ ]:


train_df['SalePrice'] = np.log1p(train_df['SalePrice']) # log transform to make it more "normal"


# In[ ]:


plt.subplots(figsize=(12,9))
sns.distplot(train_df['SalePrice'], fit=stats.norm)


# Now, it's more symmetric / normal.

# In[ ]:


all_df = pd.concat((train_df,test_df)).reset_index(drop = True)


# In[ ]:


all_df.drop(['SalePrice'],axis =1,inplace = True)


# In[ ]:


all_df.shape


# # Exploration of numeric features:

# In[ ]:


numeric_cols = list(all_df._get_numeric_data().columns) 


# In[ ]:


print((numeric_cols))


# In[ ]:


numeric_cols_df = all_df[['MSSubClass', 'LotFrontage', 'LotArea', 'OverallQual', 'OverallCond', 'YearBuilt', 'YearRemodAdd', 
                       'MasVnrArea', 'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', '1stFlrSF', '2ndFlrSF', 
                       'LowQualFinSF', 'GrLivArea', 'BsmtFullBath', 'BsmtHalfBath', 'FullBath', 'HalfBath', 'BedroomAbvGr',
                       'KitchenAbvGr', 'TotRmsAbvGrd', 'Fireplaces', 'GarageYrBlt', 'GarageCars', 'GarageArea', 
                       'WoodDeckSF', 
                       'OpenPorchSF', 'EnclosedPorch', '3SsnPorch', 'ScreenPorch', 'PoolArea', 'MiscVal', 'MoSold', 
                       'YrSold']]


# In[ ]:


numeric_cols_df.shape


# In[ ]:


numeric_cols_df.hist(bins =20, figsize = (20,20))
plt.show()


# Above plots show distribution of each numeric feature's value. Here we can identify more categorical features and have already been label encoded with numrical values. Also, some of the freatures are normally distributed.

# In[ ]:


for i in train_df[numeric_cols].columns:
    plt.figure(figsize = (8,4))
    plt.scatter( train_df[numeric_cols][i],train_df["SalePrice"])
    plt.xlabel(i)
    plt.ylabel("SalePrice")


# In the above scatter plot, correlation between target variable and each of the numeric features can be seen.

# In[ ]:


corr = train_df[train_df._get_numeric_data().columns].corr()
plt.subplots(figsize=(20,9))
sns.heatmap(corr)


# In[ ]:


# filtering out only highly co-related features
top_feature = corr.index[abs(corr['SalePrice']>0.5)]
plt.subplots(figsize=(12, 8))
top_corr = train_df[train_df._get_numeric_data().columns][top_feature].corr()
sns.heatmap(top_corr, annot=True)
plt.show()


# In[ ]:


train_df["OverallQual"].unique() # listing values for highest correlated feature with target variable


# In[ ]:


sns.barplot(train_df.OverallQual, train_df.SalePrice) # plotting the highest correlated feature


# An increasing trend in SalePrice can be observed with increasing Quality. This is what we expect.

# In[ ]:


plt.figure(figsize=(18, 8))
sns.boxplot(x=train_df.OverallQual, y=train_df.SalePrice)


# In[ ]:


col = ['SalePrice', 'OverallQual', 'GrLivArea', 'GarageCars', 'TotalBsmtSF', 'FullBath', 'TotRmsAbvGrd', 'YearBuilt',
       '1stFlrSF']
sns.set(style='ticks')
sns.pairplot(train_df[col], height=3, kind='reg')


# # Handling missing data in numeric cols:

# In[ ]:


data_nan = (numeric_cols_df.isnull().sum() / len(numeric_cols_df)) * 100
data_nan = data_nan.drop(data_nan[data_nan == 0].index).sort_values(ascending=False)[:20]
missing_data = pd.DataFrame({'Missing Ratio' :data_nan})
missing_data


# In[ ]:


plt.figure(figsize = (8,5))
sns.barplot(x=data_nan.index, y=data_nan)
plt.xlabel('Features', fontsize=15)
plt.xticks(rotation= 'vertical' )
plt.ylabel('Percent of missing values', fontsize=15)
plt.title('Percent missing data by feature', fontsize=15)


# In[ ]:


all_df["LotFrontage"] = all_df.groupby("Neighborhood")["LotFrontage"].transform(lambda x: x.fillna(x.median()))


# In[ ]:


all_df["MasVnrArea"] = all_df["MasVnrArea"].fillna(0)


# In[ ]:


all_df["GarageYrBlt"] = all_df["GarageYrBlt"].fillna(0)


# In[ ]:


all_df["BsmtFinSF2"] = all_df["BsmtFinSF2"].fillna(0)


# In[ ]:


all_df['BsmtFinSF1'] = all_df["BsmtFinSF1"].fillna(0)


# In[ ]:


all_df["BsmtHalfBath"] = all_df["BsmtHalfBath"].fillna(0)


# In[ ]:


all_df['BsmtFullBath'] = all_df["BsmtFullBath"].fillna(0)


# In[ ]:


all_df['GarageArea'] = all_df["GarageArea"].fillna(0)


# In[ ]:


all_df["GarageCars"] = all_df["GarageCars"].fillna(0)


# In[ ]:


all_df["TotalBsmtSF"] = all_df["TotalBsmtSF"].fillna(0)


# In[ ]:


all_df["BsmtUnfSF"] = all_df["BsmtUnfSF"].fillna(0)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# # Exploration of Categorical Features:

# In[ ]:


categ_col = list(set(all_df.columns.unique()) - set(numeric_cols))


# In[ ]:


all_df.shape


# In[ ]:


print(len(categ_col))


# In[ ]:


print((categ_col))


# In[ ]:


categ_col_df = all_df[['Neighborhood', 'MSZoning', 'RoofStyle', 'SaleCondition', 'HouseStyle', 'Utilities', 'LandContour',
                        'MasVnrType', 'Functional', 'Condition1', 'KitchenQual', 'ExterQual', 'PoolQC', 'Foundation',
                        'Heating', 'LotConfig', 'GarageCond', 'LandSlope', 'Street', 'Exterior2nd', 'BsmtQual', 
                        'Exterior1st', 'GarageFinish', 'BsmtExposure', 'GarageType', 'HeatingQC', 'CentralAir', 
                        'PavedDrive', 'SaleType', 'BsmtCond', 'RoofMatl', 'Alley', 'LotShape', 'BldgType', 'BsmtFinType1', 
                        'GarageQual', 'Electrical', 'Fence', 'MiscFeature', 'ExterCond', 'FireplaceQu', 'BsmtFinType2',
                        'Condition2']]


# In[ ]:


for i in categ_col_df:
    train_df.boxplot("SalePrice",i, rot = 30, figsize = (12,6))


# # Handling missing data in categorical cols:

# In[ ]:


data_nan1 = (categ_col_df.isnull().sum() / len(categ_col_df)) * 100
data_nan1 = data_nan1.drop(data_nan1[data_nan1 == 0].index).sort_values(ascending=False)[:30]
missing_data = pd.DataFrame({'Missing Ratio' :data_nan1})
missing_data


# In[ ]:


plt.figure(figsize = (8,5))
plt.xticks(rotation =90)
sns.barplot(x=data_nan1.index, y=data_nan1)
plt.xlabel('Features', fontsize=15)
plt.ylabel('Percent of missing values', fontsize=15)
plt.title('Percent missing data by feature', fontsize=15)


# In[ ]:


all_df["PoolQC"] = all_df["PoolQC"].fillna("None")


# In[ ]:


all_df["MiscFeature"] = all_df["MiscFeature"].fillna("None")


# In[ ]:


all_df["Alley"] = all_df["Alley"].fillna("None")


# In[ ]:


all_df["Fence"] = all_df["Fence"].fillna("None")


# In[ ]:


all_df["FireplaceQu"] = all_df["FireplaceQu"].fillna("None")


# In[ ]:


for col in ('GarageType', 'GarageFinish', 'GarageQual', 'GarageCond'):
    all_df[col] = all_df[col].fillna('None')


# In[ ]:


for col in ('BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2'):
    all_df[col] = all_df[col].fillna('None')


# In[ ]:


all_df["MasVnrType"] = all_df["MasVnrType"].fillna("None")


# In[ ]:


all_df["MSZoning"] = all_df["MSZoning"].fillna("None")


# In[ ]:


all_df["Exterior1st"] =all_df["Exterior1st"].fillna("None")


# In[ ]:


all_df["Exterior2nd"] = all_df["Exterior2nd"].fillna("None")


# In[ ]:


all_df["Functional"] = all_df["Functional"].fillna("None")


# In[ ]:


all_df["SaleType"] = all_df["SaleType"].fillna("None")


# In[ ]:


all_df["KitchenQual"] = all_df["KitchenQual"].fillna("None")


# In[ ]:


all_df['Electrical'] = all_df['Electrical'].fillna(all_df['Electrical'].mode()[0])


# In[ ]:


all_df = all_df.drop(['Utilities'], axis=1)


# In[ ]:


all_df.isnull().values.sum()

Label Encoding:
# In[ ]:


from sklearn.preprocessing import LabelEncoder


# In[ ]:


cols = ('Neighborhood', 'MSZoning', 'RoofStyle', 'SaleCondition', 'HouseStyle', 'LandContour',
                        'MasVnrType', 'Functional', 'Condition1', 'KitchenQual', 'ExterQual', 'PoolQC', 'Foundation',
                        'Heating', 'LotConfig', 'GarageCond', 'LandSlope', 'Street', 'Exterior2nd', 'BsmtQual', 
                        'Exterior1st', 'GarageFinish', 'BsmtExposure', 'GarageType', 'HeatingQC', 'CentralAir', 
                        'PavedDrive', 'SaleType', 'BsmtCond', 'RoofMatl', 'Alley', 'LotShape', 'BldgType', 'BsmtFinType1', 
                        'GarageQual', 'Electrical', 'Fence', 'MiscFeature', 'ExterCond', 'FireplaceQu', 'BsmtFinType2',
                        'Condition2')


# In[ ]:


for c in cols:
    label = LabelEncoder() 
    label.fit(list(all_df[c].values)) 
    all_df[c] = label.transform(list(all_df[c].values))


# In[ ]:


all_df.shape


# In[ ]:


all_df.head()


# In[ ]:


ntrain = train_df.shape[0]
ntrain


# In[ ]:


ntest = test_df.shape[0]
ntest


# In[ ]:


new_train_df = all_df[:ntrain].copy()


# In[ ]:


new_train_df["SalePrice"] = sale_price


# In[ ]:


new_train_df['SalePrice'].head()


# In[ ]:


test_df = all_df[:ntest]
test_df.shape


# # Model Fitting:

# 1. Linear Regression:

# In[ ]:


X = new_train_df.drop("SalePrice", axis = 1)


# In[ ]:


Y = new_train_df["SalePrice"]


# In[ ]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=7)


# In[ ]:


from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(X_train, y_train)
print("Accuracy is", model.score(X_test, y_test)*100)


# 2. Random Forest:

# In[ ]:


from sklearn.ensemble import RandomForestRegressor
model = RandomForestRegressor(n_estimators=1000)
model.fit(X_train, y_train)
print("Accuracy is ", model.score(X_test, y_test)*100)


# 3. Gradient Boosting:

# In[ ]:


from sklearn.ensemble import GradientBoostingRegressor
GBR = GradientBoostingRegressor(n_estimators=100, max_depth=4)
GBR.fit(X_train, y_train)
print("Accuracy is ", GBR.score(X_test, y_test)*100)


# 4. Ridge Regression:

# In[ ]:


from sklearn.model_selection import cross_val_score
from sklearn.linear_model import Ridge
ridge = Ridge()
MSEs = cross_val_score(ridge, X, Y, cv=5)
print(MSEs)
mean_MSE = np.mean(MSEs)
print(mean_MSE)


# In[ ]:


#Test predictions


# In[ ]:





# In[ ]:


predicted_price = pd.Series(GBR.predict(test_df), name = "SalePrice")

submission_df = pd.concat([test_id,predicted_price], axis=1)


# In[ ]:


submission_df.head()


# In[ ]:


# filename = 'House SalePrice Predictions 1.csv'

# submission_df.to_csv(filename,index=False)

# print('Saved file: ' + filename)


# In[ ]:




