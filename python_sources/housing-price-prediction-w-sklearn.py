#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import sklearn
import sklearn.preprocessing as pp
import sklearn.model_selection as ms


# In[ ]:


df = pd.read_csv('../input/train.csv')
df = df.set_index('Id')

sdf = pd.read_csv('../input/test.csv')
sdf = sdf.set_index('Id')
df.head()


# In[ ]:


price = df.SalePrice
print("Average sale price: " + "${:,.2f}".format(price.mean()))


# In[ ]:


# Create a matrix of test and train data so that column creation for Categorical variables works correctly

df = df.drop('SalePrice', axis=1)
all_df = df.append(sdf)
all_df.shape


# In[ ]:


all_features = 'MSSubClass,MSZoning,LotFrontage,LotArea,Street,Alley,LotShape,LandContour,Utilities,LotConfig,LandSlope,Neighborhood,Condition1,Condition2,BldgType,HouseStyle,OverallQual,OverallCond,YearBuilt,YearRemodAdd,RoofStyle,RoofMatl,Exterior1st,Exterior2nd,MasVnrType,MasVnrArea,ExterQual,ExterCond,Foundation,BsmtQual,BsmtCond,BsmtExposure,BsmtFinType1,BsmtFinSF1,BsmtFinType2,BsmtFinSF2,BsmtUnfSF,TotalBsmtSF,Heating,HeatingQC,CentralAir,Electrical,1stFlrSF,2ndFlrSF,LowQualFinSF,GrLivArea,BsmtFullBath,BsmtHalfBath,FullBath,HalfBath,BedroomAbvGr,KitchenAbvGr,KitchenQual,TotRmsAbvGrd,Functional,Fireplaces,FireplaceQu,GarageType,GarageYrBlt,GarageFinish,GarageCars,GarageArea,GarageQual,GarageCond,PavedDrive,WoodDeckSF,OpenPorchSF,EnclosedPorch,3SsnPorch,ScreenPorch,PoolArea,PoolQC,Fence,MiscFeature,MiscVal,MoSold,YrSold,SaleType,SaleCondition'.split(',')
numeric_features = ['LotFrontage', 'LotArea', 'MasVnrArea', 'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF','LowQualFinSF','GrLivArea','BsmtFullBath','BsmtHalfBath','FullBath','HalfBath','BedroomAbvGr','KitchenAbvGr','TotRmsAbvGrd','TotalBsmtSF','Fireplaces', 'GarageCars', 'GarageArea','WoodDeckSF', 'OpenPorchSF', 'EnclosedPorch', '3SsnPorch', 'ScreenPorch', 'PoolArea', 'MiscVal']
categorical_features = [f for f in all_features if not(f in numeric_features)]

(len(all_features), len(categorical_features), len(numeric_features))


# In[ ]:


numeric_df = all_df[numeric_features]
numeric_df.shape


# In[ ]:


X = numeric_df.as_matrix()

# Impute missing

imp = pp.Imputer(missing_values='NaN', strategy='most_frequent', axis=0)
imp = imp.fit(X)
X = imp.transform(X)
X.shape


# In[ ]:


# Scale and Center
scaler = pp.StandardScaler()
scaler.fit(X)
X = scaler.transform(X)
X[0:5,:]


# In[ ]:


# deal w/Categorical

def process_categorical(ndf, df, categorical_features):
    for f in categorical_features:
        new_cols = pd.DataFrame(pd.get_dummies(df[f]))
        new_cols.index = df.index
        ndf = pd.merge(ndf, new_cols, how = 'inner', left_index=True, right_index=True)
    return ndf

numeric_df = pd.DataFrame(X)
numeric_df.index = all_df.index
combined_df = process_categorical(numeric_df, all_df, categorical_features)
combined_df.head()


# In[ ]:


X = combined_df.as_matrix()
X.shape


# In[ ]:


test_n = df.shape[0]
X_train = X[:test_n,:]
X_train, X_val, y_train, y_val = ms.train_test_split(X_train, price, test_size=0.3, random_state=0)
X_test = X[test_n:,:]

(X_train.shape, X_val.shape, X_test.shape)


# In[ ]:


from sklearn import linear_model

lr = linear_model.LinearRegression()
lr.fit(X_train, y_train)


# In[ ]:


print('R^2 is %f' % lr.score(X_val, y_val))


# In[ ]:


from sklearn.metrics import mean_squared_error

y_val_pred = lr.predict(X_val)

print('mean squared error is %s' %       '{:,.2f}'.format(mean_squared_error(y_val, y_val_pred)))


# In[ ]:


import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
plt.style.use('ggplot')

fig, ax = plt.subplots()

ax.plot(y_val, y_val_pred, 'b.')
ax.plot([y_val.min(), y_val.max()], [y_val.min(), y_val.max()], 'k--')
ax.set_xlabel('Measured')
ax.set_ylabel('Predicted')
plt.show()


# In[ ]:


pd.options.display.float_format = '{:,.4f}'.format
y_submit = lr.predict(X_test)
sdf['SalePrice'] = y_submit
sdf.to_csv('submission.csv', columns = ['SalePrice'])

