#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# In[ ]:


pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', -1)


# In[ ]:


import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')

from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, RobustScaler


# In[ ]:


df = pd.read_csv('../input/house-prices-advanced-regression-techniques/train.csv')
df_test = pd.read_csv('../input/house-prices-advanced-regression-techniques/test.csv')


# In[ ]:


df.head()


# In[ ]:


corr = df.corr()


# In[ ]:


ranking_L = abs(corr['SalePrice']).sort_values(ascending=False).copy()
ranking_L.head(20)


# In[ ]:


sns.scatterplot(df_re.GrLivArea, df_re.SalePrice);


# In[ ]:


sns.scatterplot(df_re.GarageYrBlt, df_re.SalePrice);


# In[ ]:


many_na = df.isnull().sum()/df.shape[0]


# In[ ]:


many_na


# In[ ]:


many_na_feat = list(many_na.loc[many_na > 0.8].keys())


# In[ ]:


many_na_feat


# In[ ]:


df.drop(many_na_feat,axis=1, inplace=True)


# In[ ]:


df_test.drop(many_na_feat,axis=1, inplace=True)


# In[ ]:


df.shape


# In[ ]:


df.info()


# In[ ]:


from sklearn.preprocessing import LabelEncoder


# In[ ]:


def pre_processing(df):
    df['MSSubClass'] = df.MSSubClass.astype('str')
    df_1 = df.select_dtypes(include=['object']).fillna("N/A")
    df_2 = df.select_dtypes(include=['int64', 'float64'])
    df_2 = df_2.fillna(df_2.median())
    df = pd.concat([df_1,df_2], axis=1)
    
    """
    df['YearBuilt'] = 2020 - df['YearBuilt']
    df['YearRemodAdd'] = 2020 - df['YearRemodAdd']
    df['GarageYrBlt'] = 2020 - df['GarageYrBlt']
    df['YrSold'] = 2020 - df['YrSold']

    df['LotFrontage'] = df['LotFrontage']/1000
    df['LotArea'] = df['LotArea']/1000
    df['MasVnrArea'] = df['MasVnrArea']/1000
    df['BsmtFinSF1'] = df['BsmtFinSF1']/1000
    df['BsmtFinSF2'] = df['BsmtFinSF2']/1000
    df['BsmtUnfSF'] = df['BsmtUnfSF']/1000
    df['TotalBsmtSF'] = df['TotalBsmtSF']/1000
    df['1stFlrSF'] = df['1stFlrSF']/1000
    df['2ndFlrSF'] = df['2ndFlrSF']/1000
    df['LowQualFinSF'] = df['LowQualFinSF']/1000
    df['GrLivArea'] = df['GrLivArea']/1000
    df['GarageArea'] = df['GarageArea']/1000
    df['WoodDeckSF'] = df['WoodDeckSF']/1000
    df['OpenPorchSF'] = df['OpenPorchSF']/1000
    df['EnclosedPorch'] = df['EnclosedPorch']/1000
    df['3SsnPorch'] = df['3SsnPorch']/1000
    df['ScreenPorch'] = df['ScreenPorch']/1000
    df['PoolArea'] = df['PoolArea']/1000
    df['MiscVal'] = df['MiscVal']/1000
    """
    
    return df


# In[ ]:


cat_feature = ['MSSubClass', 'MSZoning','Street','LotShape','LandContour','Utilities','LotConfig','LandSlope',
              'Neighborhood','Condition1','Condition2','BldgType','HouseStyle','RoofStyle','RoofMatl','Exterior1st',
              'Exterior2nd','MasVnrType','Foundation','Heating','CentralAir','Electrical','GarageType','PavedDrive',
              'SaleType','SaleCondition']
deg_feature = ['ExterQual','ExterCond','BsmtQual','BsmtCond','BsmtExposure','BsmtFinType1','BsmtFinType2',
              'HeatingQC','KitchenQual','Functional','FireplaceQu','GarageFinish','GarageQual','GarageCond']
num_feature = df.select_dtypes(include=['int64', 'float64']).columns.to_list()
num_feature = [e for e in num_feature if e not in ('SalePrice', 'Id', 'MSSubClass')]


# In[ ]:


dict_1 = {"Ex":6, "Gd":5, "TA":4, "Fa":3, "Po":2, "NA":1, "No":2, "Mn":3, "Av":4, "Unf":2, "LwQ":5 , "Rec":4, "BLQ":5,
         "ALQ":6, "GLQ":7, "Sal":1, "Sev":2, "Maj2":3, "Maj1":4, "Mod":5, "Min2":6, "Min1":7, "Typ":8, "RFn":3, "Fin":4,
          "MnWw":2, "GdWo":3, "MnPrv":4, "GdPrv":5, "N/A":0}

le = LabelEncoder()

def sk_encoder(df, testdata = 0):
    df_cat = df[cat_feature].apply(le.fit_transform)
    df['MSSubClass'] = df.MSSubClass.astype(int)
    df_deg = df[deg_feature].replace(dict_1).astype(int)
    df_num = df[num_feature]
    
     
    if testdata == 0:
        df_re = pd.concat([df['Id'], df_cat, df_deg, df_num,df['SalePrice']],axis=1)
    elif testdata == 1:
        df_re = pd.concat([df['Id'], df_cat, df_deg, df_num],axis=1)
    return df_re


# In[ ]:


df2 = pre_processing(df)
df_re = sk_encoder(df2, testdata = 0)


# In[ ]:


df_re.describe()


# In[ ]:


top_corr_features = corr.index[abs(corr["SalePrice"])>0.5]
plt.figure(figsize=(15,15))
g = sns.heatmap(df_re[top_corr_features].corr(),annot=False,cmap="RdYlGn")


# In[ ]:


top_features = top_corr_features.to_list()


# In[ ]:


df_re_1 = df_re.copy()


# In[ ]:


dataset = df_re_1.values
X_train = dataset[:,:df_re_1.shape[1]-1]
y_train = dataset[:,df_re_1.shape[1]-1]


# In[ ]:


X_train.shape


# In[ ]:


from sklearn.preprocessing import MinMaxScaler, StandardScaler


# In[ ]:


#scaler = MinMaxScaler()
#scaler.fit(X_train)
#X_train = scaler.transform(X_train)


# In[ ]:


from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import ExtraTreesClassifier


# In[ ]:


from sklearn import tree
from sklearn import ensemble
from sklearn.model_selection import cross_val_score


# In[ ]:


n_score = []

for i in range(10,60):
    
    clf = ExtraTreesClassifier(random_state = 1)
    clf = clf.fit(X_train, y_train)
    tb_model = SelectFromModel(clf, prefit=True, threshold=-np.inf, max_features =i)
    tb_train_x = tb_model.transform(X_train)
    
    regr0 = ensemble.RandomForestRegressor(n_estimators=500, min_samples_split = 2, random_state=1)
    score = cross_val_score(regr0, tb_train_x, y_train, cv=5)
    score = np.average(score)
    
    print([i,score])
    
    n_score.append([i,score])

n_score = np.array(n_score)


# In[ ]:


plt.title("Number of selected features vs score")
plt.xlabel('Number of selected features')
plt.ylabel('score')
plt.plot(n_score[:,0],n_score[:,1]);


# In[ ]:


clf = ExtraTreesClassifier(random_state = 1)
clf = clf.fit(X_train, y_train)
tb_model = SelectFromModel(clf, prefit=True, threshold=-np.inf, max_features =40)
tb_train_x = tb_model.transform(X_train)


# In[ ]:


regr0 = ensemble.RandomForestRegressor(n_estimators=500, min_samples_split = 2, random_state=1)
scores = cross_val_score(regr0, tb_train_x, y_train, cv=5)
print(scores)
print(np.average(scores))


# In[ ]:


import xgboost as xgb


# In[ ]:


model_xgb = xgb.XGBRegressor(n_estimators=500, max_depth=2, learning_rate=0.1) 
scores = cross_val_score(model_xgb, tb_train_x, y_train, cv=5)
print(scores)
print(np.average(scores))


# In[ ]:


df_test2 = pre_processing(df_test)
df_test_re = sk_encoder(df_test2, testdata = 1)


# In[ ]:


df_test_re.head()


# In[ ]:


dataset_test = df_test_re.values
X_test = dataset_test[:, :]


# In[ ]:


#X_test = scaler.transform(X_test)


# In[ ]:


X_test.shape


# In[ ]:


tb_test_x = tb_model.transform(X_test)


# In[ ]:


tb_test_x.shape


# In[ ]:


df_test_re.head()


# In[ ]:


regr0.fit(tb_train_x, y_train)
predictions = regr0.predict(tb_test_x)
pred_df = pd.DataFrame(predictions, columns=['SalePrice'])
result = pd.concat([df_test_re,pred_df], axis=1, join='outer', ignore_index=False, keys=None, sort = False)
result = result[['Id','SalePrice']]
print(result.tail(5))
result.to_csv('sample_submission_reg0.csv',index=False)


# In[ ]:


model_xgb.fit(tb_train_x, y_train)
predictions = model_xgb.predict(tb_test_x)
pred_df = pd.DataFrame(predictions, columns=['SalePrice'])
result = pd.concat([df_test_re,pred_df], axis=1, join='outer', ignore_index=False, keys=None, sort = False)
result = result[['Id','SalePrice']]
print(result.tail(5))
result.to_csv('sample_submission_model_xgb.csv',index=False)


# In[ ]:


from sklearn.ensemble import VotingRegressor
#r1 = svm.SVR(kernel='rbf', C=10000000, gamma= 'scale')
r2 = ensemble.RandomForestRegressor(n_estimators=500, min_samples_split = 2, random_state=1)
model_xgb = xgb.XGBRegressor(n_estimators=500, max_depth=2, learning_rate=0.1)
er = VotingRegressor([('rf', r2),('xgb', model_xgb)], weights =[1,2])
scores = cross_val_score(er, tb_train_x, y_train, cv=5)
print(scores)


# In[ ]:


er.fit(tb_train_x, y_train)
predictions = er.predict(tb_test_x)
pred_df = pd.DataFrame(predictions, columns=['SalePrice'])
result = pd.concat([df_test_re,pred_df], axis=1, join='outer', ignore_index=False, keys=None, sort = False)
result = result[['Id','SalePrice']]
print(result.tail(5))
result.to_csv('sample_submission_em_vo.csv',index=False)


# In[ ]:




