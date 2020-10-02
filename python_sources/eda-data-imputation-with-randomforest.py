#!/usr/bin/env python
# coding: utf-8

# In[62]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LassoCV
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import mean_squared_error, make_scorer

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[9]:


#bring in the six packs
df_train = pd.read_csv('../input/train.csv')
df_test = pd.read_csv('../input/test.csv')

datasets = [df_train, df_test]
df_train.head()


# Data fields
# 
# SalePrice - the property's sale price in dollars. This is the target variable that you're trying to predict.
# 
# These are some features we could look at the distrcutions of and try to transform to make them more normal. 
# * LotArea, LotFrontage, MasVnrArea, BsmtUnfSF, TotalBsmtSF, 1stFlrSF,  2ndFlrSF, BsmtFinSF1, BsmtFinSF2,  GarageArea, PoolArea

# In[10]:


numericFields = ['SalePrice', 'LotArea', 'LotFrontage', 'MasVnrArea', 'BsmtUnfSF', 'TotalBsmtSF', '1stFlrSF',  '2ndFlrSF', 'BsmtFinSF1', 'BsmtFinSF2',  'GarageArea', 'PoolArea']
for field in numericFields:
    plt.figure()
    sns.distplot(df_train.loc[~df_train[field].isnull()][field])


# Postively Skewed = SalePrice, LotArea, LotFrontage, MasVnrArea, BsmtUnfSF, 1stFlrSF
# The rest seem to be somewhat bimodal.

# In[11]:


numericFields = ['SalePrice', 'LotArea', 'LotFrontage', 'MasVnrArea', 'BsmtUnfSF', '1stFlrSF']
for field in numericFields:
    plt.figure()
    sns.distplot(np.log1p(df_train.loc[~df_train[field].isnull()][field]))


# SalePrice, LotArea, 1stFlrSF benefit from log1p, converting it more to a normal dist. 

# In[12]:


print(df_train['SalePrice'].skew()) # apply log to convert to normal dist.


# In[13]:


df_train['SalePrice'].describe()


# In[14]:


df_train['SalePrice'].isnull().sum() # no null values.


# In[15]:


df_train.BedroomAbvGr.describe()


# In[16]:


data = pd.concat([df_train['SalePrice'], df_train['BedroomAbvGr']], axis=1)
data.plot.scatter(x='BedroomAbvGr', y='SalePrice')


# Seems like Bedrooms doesn't have a clear relationship through this graph.

# In[17]:


data = pd.concat([df_train['SalePrice'], df_train['LotArea']], axis=1)
data.plot.scatter(x='LotArea', y='SalePrice')


# LotArea STRONGLY effects the housing price, as can be seen by this steep linear relationship. 

# In[18]:


var = 'TotalBsmtSF'
data = pd.concat([df_train['SalePrice'], df_train[var]], axis=1)
data.plot.scatter(x=var, y='SalePrice')


# In[19]:


data = pd.concat([df_train['SalePrice'], df_train['OverallQual']], axis=1)
f, ax = plt.subplots(figsize=(8, 6))
fig = sns.boxplot(x='OverallQual', y='SalePrice', data=data)


# In[20]:


corrmat = df_train.corr()
f, ax = plt.subplots(figsize=(12, 9))
sns.heatmap(corrmat, vmax=.8, square=True);


# In[21]:


cols = corrmat.nlargest(10, 'SalePrice').index
corrmat_small = df_train[cols].corr()
hm = sns.heatmap(corrmat_small)


# In[22]:


proportion_null = pd.DataFrame(df_train.isnull().sum() / df_train.isnull().count(), columns=['Prop Null']).sort_values(by="Prop Null", ascending=False).head(10)
proportion_null


# In[23]:


df_train.LotFrontage.describe()


# ## Pool Quality Missing Data
# Approach is to look at the PoolArea and PoolQC and see if they align. 
# If square footage tells us whether there is a pool or not , we can impute values based on pool existance. 

# In[24]:


pool_stats = pd.concat([df_train.PoolArea, df_train.PoolQC], axis=1)
zero_pool = pool_stats[pool_stats.PoolArea == 0]
zero_pool.PoolQC.isnull().describe()


# Seems like all the PoolArea == 0 aligns with the PoolQC being Nan. Lets add an existance column.

# In[25]:


for dataset in datasets:
    dataset.HasPool = dataset.PoolArea != 0


# We could drop the related PoolQC, or fill it with a new quality category. Lets do this.

# In[26]:


print(set(df_train.PoolQC))
quality_map = {np.nan: 0, 'Fa': 1, 'Gd': 2, 'Ex': 3}
for dataset in datasets:
    dataset.PoolQC = dataset.PoolQC.map(quality_map)


# Ok done with PoolQC, lets move on to MiscFeature with 0.963014 empty.

# In[ ]:


print(set(df_train.MiscFeature))


# We can see there is a small amount of types of MiscFeature values, so we can just encode these and zero fill.

# In[27]:


feature_map = {np.nan: 0, 'TenC': 1, 'Gar2': 2, 'Othr': 3, 'Shed': 4}
for dataset in datasets:
    dataset.MiscFeature = dataset.MiscFeature.map(feature_map)


# Now, Alley.

# In[28]:


set(df_train.Alley)
alley_map = {np.nan: 0, 'Pave': 1, 'Grvl': 2}
for dataset in datasets:
    dataset.Alley = dataset.Alley.map(alley_map)


# In[29]:


set(df_train.Fence)
fence_map = {np.nan: 0, 'MnWw': 1, 'GdPrv': 2, 'GdWo': 3, 'MnPrv': 4}
for dataset in datasets:
    dataset.Fence = dataset.Fence.map(fence_map)


# Now lets look at FireplaceQu.

# In[30]:


df_train.groupby('FireplaceQu').count().Id.plot(kind='bar')
fireplace_qu_mapping = {np.nan: 0, 'Po': 1, 'Fa':2, 'TA': 3, 'Gd': 4, 'Ex': 5}
for dataset in datasets:
    dataset.FireplaceQu = dataset.FireplaceQu.map(fireplace_qu_mapping)


# 

# Now lets look at LotFrontage

# In[31]:


data = pd.concat([df_train.LotFrontage, df_train.SalePrice], axis=1)
data.plot.scatter(x="LotFrontage", y="SalePrice")


# It seems to have a decent correlation with SalePrice, so we should definitley work to keep it. Lets explore the values more in detail.

# In[32]:


print(df_train.LotFrontage.describe())
cols = corrmat.nlargest(10, 'LotFrontage').index
corrmat_small = df_train[cols].corr()
hm = sns.heatmap(corrmat_small)


# The highest LotFrontage correlators are with 1stFlrSF and LotArea, which makes a lot of sense considering both of these are area related. But they are still not super correlated. Maybe we could fill in a median value here based on Properties with similar 1stFlrSF.

# In[33]:


# for each property, if the lot frontage is null, then find the property with the closest lot area that is not null, and fill it with this value. 
def remove_lot_frontage_nulls(dataset):
    null_frontages = dataset.loc[dataset.LotFrontage.isnull()]
    non_null_frontages = dataset.loc[~dataset.LotFrontage.isnull()]
    new_frontages_rows = []
    for first_floor_sf in null_frontages['1stFlrSF']:
        df_sort = non_null_frontages.iloc[(non_null_frontages['1stFlrSF'] - first_floor_sf).abs().argsort()[:1]]
        new_frontages_rows.append(df_sort)
    new_frontages_rows = pd.concat(new_frontages_rows)
    new_frontages_rows.index = null_frontages.index
    lotFrontageNoNa = dataset.LotFrontage.dropna()
    print(lotFrontageNoNa.head(10))
    print(new_frontages_rows.LotFrontage.head(10))
    dataset.LotFrontage = pd.concat([lotFrontageNoNa, new_frontages_rows.LotFrontage])
    return dataset

df_train = remove_lot_frontage_nulls(df_train)
df_test = remove_lot_frontage_nulls(df_test)
datasets = [df_train, df_test]


# In[34]:


df_train.LotFrontage.describe()


# In[35]:


proportion_null = pd.DataFrame(df_train.isnull().sum() / df_train.isnull().count(), columns=['Prop Null']).sort_values(by="Prop Null", ascending=False).head(20)
proportion_null


# 
# 
# We can see we no longer have LotFrontage nulls! Also, through the describe above, we can see we haven't really made major changes to the summary statistics.

# In[36]:


set(df_train.GarageQual)
garage_qual_mapping = {np.nan: 0, 'Po': 1, 'Fa':2, 'TA': 3, 'Gd': 4, 'Ex': 5}
garage_fin_mapping = {np.nan: 0, 'RFn': 1, 'Unf': 2, 'Fin': 3}
garage_type_mapping = {np.nan: 0, 'CarPort': 1, 'Attchd': 2, 'Detchd': 3, '2Types': 4, 'Basment': 5, 'BuiltIn': 6}

for dataset in datasets:
    dataset.GarageQual = dataset.GarageQual.map(garage_qual_mapping)
    dataset.GarageCond = dataset.GarageCond.map(garage_qual_mapping)
    dataset.GarageFinish = dataset.GarageFinish.map(garage_fin_mapping)
    dataset.GarageType = dataset.GarageType.map(garage_type_mapping)


# In[37]:


print (set(df_train.BsmtFinType2))
bsmt_qual_mapping = {np.nan: 0, 'Po': 1, 'Fa':2, 'TA': 3, 'Gd': 4, 'Ex': 5}
bsmt_fin_types = {np.nan: 0, 'Rec': 1, 'Unf': 2, 'ALQ': 3, 'LwQ': 4, 'GLQ': 5, 'BLQ': 6}
bsmt_exposure = {np.nan:0, 'Mn': 1,'Gd': 2, 'No': 3, 'Av': 4}
for dataset in datasets:
    dataset.BsmtQual = dataset.BsmtQual.map(bsmt_qual_mapping)
    dataset.BsmtCond = dataset.BsmtCond.map(bsmt_qual_mapping)
    dataset.BsmtFinType1 = dataset.BsmtFinType1.map(bsmt_fin_types)
    dataset.BsmtFinType2 = dataset.BsmtFinType2.map(bsmt_fin_types)
    dataset.BsmtExposure = dataset.BsmtExposure.map(bsmt_exposure)


# In[38]:


sns.distplot(df_train.loc[~df_train.GarageYrBlt.isnull()].GarageYrBlt)
sns.distplot(df_train.loc[~df_train.YearBuilt.isnull()].YearBuilt)
data = pd.concat([df_train.GarageYrBlt, df_train.YearBuilt], axis=1)
corrmat = data.corr()
corrmat


# These are VERY correlated, so lets just take the YearBuilt to fill it in with.

# In[39]:


for dataset in datasets:
    dataset.GarageYrBlt.fillna(dataset.YearBuilt, inplace=True)


# In[40]:


for dataset in datasets:
    dataset.MasVnrType.fillna(dataset.groupby('MasVnrType').count().Id.idxmax(), inplace=True)


# In[41]:


sns.distplot(df_train.loc[~df_train.MasVnrArea.isnull()].MasVnrArea)
for dataset in datasets:
    dataset.MasVnrArea.fillna(dataset.MasVnrArea.median(), inplace=True)


# In[42]:


elec_types = {np.nan: 0, 'FuseA': 1, 'FuseF': 2, 'Mix': 3, 'SBrkr': 4, 'FuseP': 5}
set(df_train.Electrical)
for dataset in datasets:
    dataset.Electrical = dataset.Electrical.map(elec_types)


# In[51]:


y = df_train['SalePrice'] # Remember to inverse log the results!
X = df_train.drop('SalePrice', axis=1)
X = pd.get_dummies(X)
Xt = pd.get_dummies(df_test)
X, Xt = X.align(Xt, join='inner', axis=1)
X.fillna(X.mean(), inplace=True)
Xt.fillna(Xt.mean(), inplace=True)


# In[52]:


proportion_null = pd.DataFrame(X.isnull().sum() / X.isnull().count(), columns=['Prop Null']).sort_values(by="Prop Null", ascending=False).head(20)
proportion_null


# ## We have no more Nulls! Time to Evaluate.
# Judging by some of the other kernels out there I'm gonna start with the RandomForest regressor

# In[53]:


# Partition the dataset in train + validation sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 0)
print("X_train : " + str(X_train.shape))
print("X_test : " + str(X_test.shape))
print("y_train : " + str(y_train.shape))
print("y_test : " + str(y_test.shape))


# In[56]:


scorer = make_scorer(mean_squared_error, greater_is_better = False)

def rmse_cv(model, X, y):
    rmse= np.sqrt(-cross_val_score(model, X, y, scoring = scorer, cv = 10))
    return(rmse)


# In[60]:


model = RandomForestRegressor()
model.fit(X_train, y_train)
# Look at predictions on training and validation set
print("RMSE on Training set :", rmse_cv(model, X_train, y_train).mean())
print("RMSE on Test set :", rmse_cv(model, X_test, y_test).mean())
y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)

# Plot residuals
plt.scatter(y_train_pred, y_train_pred - y_train, c = "blue", marker = "s", label = "Training data")
plt.scatter(y_test_pred, y_test_pred - y_test, c = "lightgreen", marker = "s", label = "Validation data")
plt.title("Linear regression")
plt.xlabel("Predicted values")
plt.ylabel("Residuals")
plt.legend(loc = "upper left")
plt.hlines(y = 0, xmin = 10.5, xmax = 13.5, color = "red")
plt.show()

# Plot predictions
plt.scatter(y_train_pred, y_train, c = "blue", marker = "s", label = "Training data")
plt.scatter(y_test_pred, y_test, c = "lightgreen", marker = "s", label = "Validation data")
plt.title("Linear regression")
plt.xlabel("Predicted values")
plt.ylabel("Real values")
plt.legend(loc = "upper left")
plt.plot([10.5, 13.5], [10.5, 13.5], c = "red")
plt.show()


# In[65]:


# 3* Lasso
lasso = LassoCV(alphas = [0.0001, 0.0003, 0.0006, 0.001, 0.003, 0.006, 0.01, 0.03, 0.06, 0.1, 
                          0.3, 0.6, 1], 
                max_iter = 50, cv = 10)
lasso.fit(X_train, y_train)
alpha = lasso.alpha_
print("Best alpha :", alpha)

print("Try again for more precision with alphas centered around " + str(alpha))
lasso = LassoCV(alphas = [alpha * .6, alpha * .65, alpha * .7, alpha * .75, alpha * .8, 
                          alpha * .85, alpha * .9, alpha * .95, alpha, alpha * 1.05, 
                          alpha * 1.1, alpha * 1.15, alpha * 1.25, alpha * 1.3, alpha * 1.35, 
                          alpha * 1.4], 
                max_iter = 50, cv = 10)
lasso.fit(X_train, y_train)
alpha = lasso.alpha_
print("Best alpha :", alpha)

print("Lasso RMSE on Training set :", rmse_cv(lasso, X_train, y_train).mean())
print("Lasso RMSE on Test set :", rmse_cv(lasso, X_test, y_test).mean())
y_train_las = lasso.predict(X_train)
y_test_las = lasso.predict(X_test)

# Plot residuals
plt.scatter(y_train_las, y_train_las - y_train, c = "blue", marker = "s", label = "Training data")
plt.scatter(y_test_las, y_test_las - y_test, c = "lightgreen", marker = "s", label = "Validation data")
plt.title("Linear regression with Lasso regularization")
plt.xlabel("Predicted values")
plt.ylabel("Residuals")
plt.legend(loc = "upper left")
plt.hlines(y = 0, xmin = 10.5, xmax = 13.5, color = "red")
plt.show()

# Plot predictions
plt.scatter(y_train_las, y_train, c = "blue", marker = "s", label = "Training data")
plt.scatter(y_test_las, y_test, c = "lightgreen", marker = "s", label = "Validation data")
plt.title("Linear regression with Lasso regularization")
plt.xlabel("Predicted values")
plt.ylabel("Real values")
plt.legend(loc = "upper left")
plt.plot([10.5, 13.5], [10.5, 13.5], c = "red")
plt.show()

# Plot important coefficients
coefs = pd.Series(lasso.coef_, index = X_train.columns)
print("Lasso picked " + str(sum(coefs != 0)) + " features and eliminated the other " +        str(sum(coefs == 0)) + " features")
imp_coefs = pd.concat([coefs.sort_values().head(10),
                     coefs.sort_values().tail(10)])
imp_coefs.plot(kind = "barh")
plt.title("Coefficients in the Lasso Model")
plt.show()


# In[66]:


model = lasso
preds = model.predict(Xt)


# In[68]:


submission = pd.DataFrame({"Id": Xt["Id"],"SalePrice": preds})
submission.loc[submission['SalePrice'] <= 0, 'SalePrice'] = 0
fileName = "submission.csv"
submission.to_csv(fileName, index=False)


# In[ ]:




