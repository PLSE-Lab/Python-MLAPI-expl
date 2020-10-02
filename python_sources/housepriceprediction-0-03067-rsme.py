#!/usr/bin/env python
# coding: utf-8

# House Price prediction with use of XGBoost

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sea
from sklearn.preprocessing import StandardScaler,LabelEncoder,MinMaxScaler
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn.impute import SimpleImputer


# In[ ]:


df_train=pd.read_csv('../input/house-prices-advanced-regression-techniques/train.csv')
df_train.shape


# Lets create a function to check for columns with null values
# 
# a function for imputing the null values. Null values will be replaced by Most frequent value for Catagorical Values and Median for Numerical Values

# In[ ]:


def checkForNull(df):
    colList=df.columns
    for col in colList:
        nullCount=pd.isnull(df[col]).sum()
        if(nullCount!=0):
            print("{}-->{}".format(col,nullCount))
            
def imputeData(df):
    colList=df.columns
    for col in colList:
        if(df[col].dtypes=='O'):
            imp = SimpleImputer(missing_values=np.nan, strategy='most_frequent')
            df_temp=imp.fit_transform(np.array(df[col]).reshape(-1,1))
            df[col]=LabelEncoder().fit_transform(df_temp)
        else:
            imp = SimpleImputer(missing_values=np.nan, strategy='median')
            df[col]=imp.fit_transform(np.array(df[col]).reshape(-1,1))


# In[ ]:


checkForNull(df_train)


# Out of total 1460 rows columns Alley,'FireplaceQu',PoolQC,Fence and MiscFeature dont have much data i.e have mostly null values. So we will remove them.

# In[ ]:


df_train=df_train.drop(['Id','Alley','FireplaceQu','PoolQC','Fence','MiscFeature'],axis=1,inplace=False)
df_train.head(5)


# Next task is to remove the outliners.
# 
# Outliners are the data points that shows abnormality with respect to other data points.
# 
# Lets plot each column and process the data which will involve the dropping of rows.

# In[ ]:


sea.scatterplot(df_train['LotFrontage'],df_train['SalePrice'])
plt.show()


# In[ ]:


df_train=df_train.drop(df_train.loc[df_train['LotFrontage']>250].index,axis=0)
df_train=df_train.drop(df_train.loc[df_train['LotArea']>100000].index,axis=0)
df_train=df_train.drop(df_train.loc[(df_train['OverallCond']==2) & (df_train['SalePrice']>300000)].index,axis=0)
df_train=df_train.drop(df_train.loc[df_train['LowQualFinSF']>550].index,axis=0)
df_train=df_train.drop(df_train.loc[(df_train['GrLivArea']>4000) & (df_train['SalePrice']<300000)].index,axis=0)
df_train=df_train.drop(df_train.loc[(df_train['OpenPorchSF']>500) & (df_train['SalePrice']<100000)].index,axis=0)
df_train=df_train.drop(df_train.loc[df_train['EnclosedPorch']>500].index,axis=0)
df_train=df_train.drop(df_train.loc[df_train['MiscVal']>3000].index,axis=0)
df_train.shape


# Again check for null values for remaining column

# In[ ]:


checkForNull(df_train)


# Impute the missing data

# In[ ]:


imputeData(df_train)


# Verify for no null values

# In[ ]:


checkForNull(df_train)


# Transform the target values

# In[ ]:


X=df_train.drop('SalePrice',axis=1)
scaler=MinMaxScaler().fit(np.array(df_train['SalePrice']).reshape(-1,1))
Y=scaler.transform(np.array(df_train['SalePrice']).reshape(-1,1))
#X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2,random_state=101)


# We will do feature selection using SelectKBest.
# 
# We will use the feature for prediction according to its importance.

# In[ ]:


from sklearn.feature_selection import SelectKBest,chi2
best=SelectKBest(chi2,k=70).fit(X,df_train['SalePrice'])
best_X=best.transform(X)

X_train,X_test,Y_train,Y_test=train_test_split(best_X,Y,test_size=0.1,random_state=101)


# We can see the selected features and sort them with their score.
# 
# -1 for sort in decreasing order.

# In[ ]:


indices = np.argsort(best.scores_)[::-1]
for i in indices:
    print("{}--->{}".format(df_train.columns[i],best.scores_[i]))


# Here comes the XGBoost with its tuned parameters

# In[ ]:


boost=XGBRegressor(n_estimators=150,learning_rate=0.09,max_depth=10,booster='gbtree',verbosity=0,n_jobs=-1,random_state=47)
boost.fit(X_train,Y_train)
np.sqrt(mean_squared_error(Y_train,boost.predict(X_train)))


# In[ ]:


np.sqrt(mean_squared_error(Y_test,boost.predict(X_test)))


# Similar changes for test data

# In[ ]:


df_test=pd.read_csv('../input/house-prices-advanced-regression-techniques/test.csv')
checkForNull(df_test)


# In[ ]:


df_test=df_test.drop(['Id','Alley','FireplaceQu','PoolQC','Fence','MiscFeature'],axis=1,inplace=False)
df_test.head(5)


# In[ ]:


imputeData(df_test)


# In[ ]:


checkForNull(df_test)


# After prediction we will inverse the transformation to get actual values.

# In[ ]:


output=boost.predict(best.transform(df_test))
out_transformed=scaler.inverse_transform(output.reshape(-1,1)).reshape(-1,)
out_transformed

