#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd


# In[ ]:


import warnings
warnings.filterwarnings('ignore')


# In[ ]:


pd.set_option('display.max_columns',500)
pd.set_option('display.max_rows',500)


# In[ ]:


import matplotlib.pyplot as plt
import seaborn as sns


# In[ ]:


import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor, ExtraTreesRegressor, GradientBoostingRegressor


# In[ ]:


from sklearn.linear_model import LinearRegression,LassoCV


# In[ ]:


from sklearn.model_selection import GridSearchCV, cross_val_score


# In[ ]:


from sklearn import preprocessing
from sklearn.preprocessing import PowerTransformer


# In[ ]:


data=pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/train.csv')


# In[ ]:


test=pd.read_csv("/kaggle/input/house-prices-advanced-regression-techniques/test.csv")


# In[ ]:


test.shape


# In[ ]:


test.head()


# In[ ]:


#data.drop('Id',1,inplace=True)


# In[ ]:


#test.drop('Id',1,inplace=True)


# In[ ]:


x= data.copy()


# In[ ]:


data.head()


# In[ ]:


test.head()


# In[ ]:


data[data['MasVnrType'].isnull()]


# In[ ]:


data['MasVnrType'].unique()


# In[ ]:


data.info()


# In[ ]:


data = data.fillna(np.nan)
test = test.fillna(np.nan)


# In[ ]:


a=data.select_dtypes(['object'])


# In[ ]:


a.info()


# In[ ]:


a.head()


# In[ ]:


for c in a.columns:
    print(c ,':', a[c].unique())
    print()


# In[ ]:





# In[ ]:





# In[ ]:


b=data.select_dtypes(['int64','float64'])


# In[ ]:


data.describe()


# In[ ]:


miss = data.columns[data.isnull().any()]


# In[ ]:


miss


# In[ ]:


data[miss].isnull().sum()


# In[ ]:


data[miss].info()


# In[ ]:


data[miss].isnull().sum()


# In[ ]:


data["Alley"] = data["Alley"].fillna("No")

data["MiscFeature"] = data["MiscFeature"].fillna("No")

data["Fence"] = data["Fence"].fillna("No")

data["PoolQC"] = data["PoolQC"].fillna("No")

data["FireplaceQu"] = data["FireplaceQu"].fillna("No")

test["Alley"] = test["Alley"].fillna("No")

test["MiscFeature"] = test["MiscFeature"].fillna("No")

test["Fence"] = test["Fence"].fillna("No")

test["PoolQC"] = test["PoolQC"].fillna("No")

test["FireplaceQu"] = test["FireplaceQu"].fillna("No")


# In[ ]:


data.loc[data['GarageArea']==0,['GarageCars','GarageYrBlt']]=0
data.loc[data['GarageCars']==0,['GarageType','GarageQual','GarageCond','GarageFinish']]='No'

test.loc[test['GarageArea']==0,['GarageCars','GarageYrBlt']]=0
test.loc[data['GarageCars']==0,['GarageType','GarageQual','GarageCond','GarageFinish']]='No'


# In[ ]:


data.loc[data['TotalBsmtSF']==0,['BsmtQual','BsmtCond','BsmtExposure','BsmtFinType1','BsmtFinType2']]='No'

#test.loc[test['TotalBsmtSF']==0,['BsmtQual','BsmtCond','BsmtExposure','BsmtFinType1','BsmtFinType2']]='No'


# In[ ]:


fig,axs=plt.subplots(2,2)
plt.subplots_adjust(hspace=0.4)
fig.set_size_inches(12,6)
sns.countplot('BsmtExposure',data=data,ax = axs[0][0]).set_title('bsmtexp - count')
sns.countplot('BsmtFinType2',data=data,ax=axs[0][1]).set_title('fintype2 - count')
sns.countplot('Electrical',data=data,ax=axs[1][0]).set_title('electrical - count')
sns.countplot('MasVnrType',data=data,ax=axs[1][1]).set_title('masvnrtype - count')


# In[ ]:


#dataset["SaleType"] = dataset["SaleType"].fillna("WD")
test["SaleType"] = test["SaleType"].fillna("WD")


# In[ ]:


#dataset["MSZoning"] = dataset["MSZoning"].fillna("RL")
test["MSZoning"] = test["MSZoning"].fillna("RL")


# In[ ]:


#dataset["KitchenQual"] = dataset["KitchenQual"].fillna("TA")
test["KitchenQual"] = test["KitchenQual"].fillna("TA")


# In[ ]:


data['BsmtExposure']=data['BsmtExposure'].fillna('No')

test['BsmtExposure']=test['BsmtExposure'].fillna('No')


# In[ ]:





# In[ ]:


data['Electrical']=data['Electrical'].fillna('SBrkr')

test['Electrical']=test['Electrical'].fillna('SBrkr')


# In[ ]:


data['MasVnrType']=data['MasVnrType'].fillna('No')
data.loc[data['MasVnrType']=='No','MasVnrArea']=0

test['MasVnrType']=test['MasVnrType'].fillna('No')
test.loc[test['MasVnrType']=='No','MasVnrArea']=0


# In[ ]:


test["BsmtCond"] = test["BsmtCond"].fillna("No")
test["BsmtQual"] = test["BsmtQual"].fillna("No")
test["BsmtFinType2"] = test["BsmtFinType2"].fillna("No")
test["BsmtFinType1"] = test["BsmtFinType1"].fillna("No")
test["Functional"] = test["Functional"].fillna("No")
test["Exterior1st"] = test["Exterior1st"].fillna("No")

test["Exterior2nd"] = test["Exterior2nd"].fillna("No")

test.loc[test["BsmtCond"] == "No","BsmtUnfSF"] = 0
test.loc[test["BsmtFinType1"] == "No","BsmtFinSF1"] = 0
test.loc[test["BsmtFinType2"] == "No","BsmtFinSF2"] = 0
test.loc[test["BsmtQual"] == "No","TotalBsmtSF"] = 0
test.loc[test["BsmtCond"] == "No","BsmtHalfBath"] = 0
test.loc[test["BsmtCond"] == "No","BsmtFullBath"] = 0
test["BsmtExposure"] = test["BsmtExposure"].fillna("No")


# In[ ]:


test["GarageType"] = test["GarageType"].fillna("No")
test["GarageFinish"] = test["GarageFinish"].fillna("No")
test["GarageQual"] = test["GarageQual"].fillna("No")
test["GarageCond"] = test["GarageCond"].fillna("No")
test.loc[test["GarageType"] == "No","GarageYrBlt"] = test["YearBuilt"][test["GarageType"]=="No"]
test.loc[test["GarageType"] == "No","GarageCars"] = 0
test.loc[test["GarageType"] == "No","GarageArea"] = 0
test["GarageArea"] = test["GarageArea"].fillna(test["GarageArea"].median())
test["GarageCars"] = test["GarageCars"].fillna(test["GarageCars"].median())
test["GarageYrBlt"] = test["GarageYrBlt"].fillna(test["GarageYrBlt"].median())


# In[ ]:


#["Utilities"] = data["Utilities"].fillna("AllPub")
test["Utilities"] = test["Utilities"].fillna("AllPub")


# In[ ]:


test.shape


# In[ ]:


data.shape


# In[ ]:


test.isnull().sum()


# In[ ]:


from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
for c in data.select_dtypes(['object']).columns:
    if data[c].nunique() <= 5:
        data[c]=le.fit_transform(data[c])
    else:
        cont = pd.get_dummies(data[c],prefix=c)
        data=pd.concat([cont,data],1)
        data.drop(c,1,inplace=True)
   


# In[ ]:


from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
for c in test.select_dtypes(['object']).columns:
    if test[c].nunique() <= 5:
        test[c]=le.fit_transform(test[c])
    else:
        cont = pd.get_dummies(test[c],prefix=c)
        test=pd.concat([cont,test],1)
        test.drop(c,1,inplace=True)   


# In[ ]:





# In[ ]:


test.head()


# In[ ]:


b.columns


# In[ ]:


b.columns[:-1]


# In[ ]:


dat = data.drop(b,1)
data=data[b.columns]

tat = test.drop(b.columns[:-1],1)
test=test[b.columns[:-1]]


# In[ ]:


from sklearn.preprocessing import Normalizer,RobustScaler


# In[ ]:


LotF = data["LotFrontage"]
data = data.drop("LotFrontage",axis= 1)

LotF1 = test["LotFrontage"]
test = test.drop("LotFrontage",axis= 1)


# In[ ]:


cop=data.copy()

cop1=test.copy()


# In[ ]:


pt = preprocessing.QuantileTransformer(output_distribution='normal')
data=pd.DataFrame(pt.fit_transform(data),columns=cop.columns)

pt = preprocessing.QuantileTransformer(output_distribution='normal')
test=pd.DataFrame(pt.fit_transform(test),columns=cop1.columns)


# In[ ]:


test.isnull().sum()


# In[ ]:


data=pd.DataFrame(Normalizer().fit_transform(data),columns=cop.columns)

test=pd.DataFrame(Normalizer().fit_transform(test),columns=cop1.columns)


# In[ ]:


pt = preprocessing.QuantileTransformer(output_distribution='normal')
data=pd.DataFrame(pt.fit_transform(data),columns=cop.columns)

pt = preprocessing.QuantileTransformer(output_distribution='normal')
test=pd.DataFrame(pt.fit_transform(test),columns=cop1.columns)


# In[ ]:


data=pd.concat([data,LotF,dat],1)

test=pd.concat([test,LotF1,tat],1)


# In[ ]:


X_train_LotF = data[LotF.notnull()] 
X_train_LotF.drop('LotFrontage',1,inplace=True)
Y_train_LotF = LotF[LotF.notnull()] 

X_train_LotF1 = test[LotF1.notnull()] 
X_train_LotF1.drop('LotFrontage',1,inplace=True)
Y_train_LotF1 = LotF1[LotF1.notnull()] 


# In[ ]:


test_LotF = data[LotF.isnull()]
test_LotF.drop('LotFrontage',1,inplace=True)

test_LotF1 = test[LotF1.isnull()]
test_LotF1.drop('LotFrontage',1,inplace=True)


# In[ ]:


test_LotF1.head()


# In[ ]:


lassocv = LassoCV(eps=1e-8)

cv_results = cross_val_score(lassocv,X_train_LotF,Y_train_LotF,cv=5,scoring="r2",n_jobs=4)
print(cv_results.mean())


lassocv = LassoCV(eps=1e-8)

cv_results = cross_val_score(lassocv,X_train_LotF1,Y_train_LotF1,cv=5,scoring="r2",n_jobs=4)
print(cv_results.mean())


# In[ ]:


lassocv.fit(X_train_LotF,Y_train_LotF)

LotF_pred = lassocv.predict(test_LotF)

LotF[LotF.isnull()] = LotF_pred

lassocv.fit(X_train_LotF1,Y_train_LotF1)

LotF_pred1 = lassocv.predict(test_LotF1)

LotF1[LotF1.isnull()] = LotF_pred1


# In[ ]:


cop=pd.concat([cop,LotF],1)

cop1=pd.concat([cop1,LotF1],1)


# In[ ]:


pt = preprocessing.QuantileTransformer(output_distribution='normal')
cop=pd.DataFrame(pt.fit_transform(cop),columns=cop.columns)

pt = preprocessing.QuantileTransformer(output_distribution='normal')
cop1=pd.DataFrame(pt.fit_transform(cop1),columns=cop1.columns)


# In[ ]:


cop=pd.DataFrame(Normalizer().fit_transform(cop),columns=cop.columns)

cop1=pd.DataFrame(Normalizer().fit_transform(cop1),columns=cop1.columns)


# In[ ]:



pt = preprocessing.QuantileTransformer(output_distribution='normal')
data=pd.DataFrame(pt.fit_transform(cop),columns=cop.columns)


pt = preprocessing.QuantileTransformer(output_distribution='normal')
test=pd.DataFrame(pt.fit_transform(cop1),columns=cop1.columns)


# In[ ]:


data=pd.concat([data,dat],1)
test=pd.concat([test,tat],1)


# In[ ]:


c = np.intersect1d(data.columns, test.columns)


# In[ ]:


from sklearn.model_selection import train_test_split


# In[ ]:


x=data.drop('SalePrice',1)
x=x[c]
y=data['SalePrice']


# In[ ]:


x_train,x_test,y_train,y_test = train_test_split(x,y,random_state=1,test_size=0.33)


# In[ ]:


x_train = x_train.loc[:,~x_train.columns.duplicated()]


# In[ ]:


from sklearn.linear_model import LinearRegression,Lasso,Ridge,ElasticNetCV
from sklearn.model_selection import train_test_split,cross_val_score
from sklearn.metrics import mean_squared_error,r2_score


# In[ ]:


def model1(alg,x_train,y_train):
    alg.fit(x_train,y_train)
    y_pred=alg.predict(x_test)
    mse=np.sqrt(mean_squared_error(y_test,y_pred))
    r_2=r2_score(y_test,y_pred)
    
    print('Mean_Squared_Error:' ,mse)
    print('r_square_value :',r_2)
    


# In[ ]:


model1(LinearRegression(),x_train,y_train)


# In[ ]:


model1(Lasso(alpha=0.05),x_train,y_train)


# In[ ]:


model1(Ridge(),x_train,y_train)


# In[ ]:


model1(ElasticNetCV(), x_train, y_train)


# In[ ]:


def model2(alg,x_train,y_train,cv=5):
    cv_results=cross_val_score(alg,x_train,y_train,cv=cv,scoring="neg_mean_squared_error")
    return (np.sqrt(-cv_results)).mean()


# In[ ]:


model2(LinearRegression(),x_train,y_train)


# In[ ]:


model2(Lasso(),x_train,y_train)


# In[ ]:


model2(Ridge(),x_train,y_train)


# In[ ]:


model2(ElasticNetCV(), x_train, y_train)


# In[ ]:


model_xgb = xgb.XGBRegressor(colsample_bytree=0.2, gamma=0.0, 
                             learning_rate=0.05, max_depth=6, 
                             min_child_weight=1.5, n_estimators=7200,
                             reg_alpha=0.9, reg_lambda=0.6,
                             subsample=0.2,seed=42, silent=1)

model2(model_xgb,x_train,y_train)


# In[ ]:



GBoost = GradientBoostingRegressor(n_estimators=3000, learning_rate=0.05,
                                   max_depth=4, max_features='sqrt',
                                   min_samples_leaf=15, min_samples_split=10, 
                                   loss='huber', random_state =5)
model2(GBoost,x_train,y_train)


# In[ ]:


def model3(alg,x,y,cv=5):
    cv_results=cross_val_score(alg,x_train,y_train,cv=cv,scoring="neg_mean_squared_error")
    return (np.sqrt(-cv_results)).mean()


# In[ ]:


model3(GBoost,x,y)


# In[ ]:





# In[ ]:


GBoost.fit(x,y)
Y_pred_GBoost = GBoost.predict(test[c])


# In[ ]:


test['SalePrice']=Y_pred_GBoost

