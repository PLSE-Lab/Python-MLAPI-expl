#!/usr/bin/env python
# coding: utf-8

# # Simple Imputation and Encoding

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import norm, skew


# In[ ]:


def one_hot_encode(df, label):
    onehot = pd.get_dummies(df[label],prefix=label)
    df.drop(label, axis=1,inplace = True)
    return df.join(onehot)


# In[ ]:


train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')
test_id = test['Id']
train_len = len(train)

train["SalePrice"] = np.log1p(train["SalePrice"])
y_train = train.SalePrice.values
all_df = pd.concat([train, test], keys=['train', 'test'])


# In[ ]:


all_df = all_df.drop(['Id','PoolQC'], axis=1)


# In[ ]:


all_df['Alley'] = all_df['Alley'].fillna('None')
all_df['MasVnrType'] = all_df['MasVnrType'].fillna('None')
all_df['BsmtQual'] = all_df['BsmtQual'].fillna(all_df['BsmtQual'].mode()[0])
all_df['BsmtCond'] = all_df['BsmtCond'].fillna(all_df['BsmtCond'].mode()[0])
all_df['FireplaceQu'] = all_df['FireplaceQu'].fillna('missing')
all_df['GarageType'] = all_df['GarageType'].fillna('missing')
all_df['GarageFinish'] = all_df['GarageFinish'].fillna('missing')
all_df['GarageQual'] = all_df['GarageQual'].fillna('missing')
all_df['GarageCond'] = all_df['GarageCond'].fillna('missing')
all_df['Fence'] = all_df['Fence'].fillna('missing')
all_df['Street'] = all_df['Street'].fillna('missing')
all_df['LotShape'] = all_df['LotShape'].fillna('missing')
all_df['LandContour'] = all_df['LandContour'].fillna('missing')
all_df['BsmtExposure'] = all_df['BsmtExposure'].fillna(all_df['BsmtExposure'].mode()[0])
all_df['BsmtFinType1'] = all_df['BsmtFinType1'].fillna('missing')
all_df['BsmtFinType2'] = all_df['BsmtFinType2'].fillna('missing')
all_df['CentralAir'] = all_df['CentralAir'].fillna('missing')
all_df['Electrical'] = all_df['Electrical'].fillna(all_df['Electrical'].mode()[0])
all_df['MiscFeature'] = all_df['MiscFeature'].fillna('missing')
all_df['MSZoning'] = all_df['MSZoning'].fillna(all_df['MSZoning'].mode()[0])    
all_df['Utilities'] = all_df['Utilities'].fillna('missing')
all_df['Exterior1st'] = all_df['Exterior1st'].fillna(all_df['Exterior1st'].mode()[0])
all_df['Exterior2nd'] = all_df['Exterior2nd'].fillna(all_df['Exterior2nd'].mode()[0])    
all_df['KitchenQual'] = all_df['KitchenQual'].fillna(all_df['KitchenQual'].mode()[0])
all_df["Functional"] = all_df["Functional"].fillna("Typ")
all_df['SaleType'] = all_df['SaleType'].fillna(all_df['SaleType'].mode()[0])
all_df['SaleCondition'] = all_df['SaleCondition'].fillna('missing')

flist = ['LotFrontage','LotArea','MasVnrArea','BsmtFinSF1','BsmtFinSF2','BsmtUnfSF',
                 'TotalBsmtSF','1stFlrSF','2ndFlrSF','LowQualFinSF','GrLivArea',
                 'BsmtFullBath','BsmtHalfBath','FullBath','HalfBath','BedroomAbvGr','KitchenAbvGr',
                 'TotRmsAbvGrd','Fireplaces','GarageCars','GarageArea','WoodDeckSF','OpenPorchSF',
                 'EnclosedPorch','3SsnPorch','ScreenPorch','PoolArea','MiscVal', 'GarageYrBlt']
for fl in flist:
    all_df[fl] = all_df[fl].fillna(0)


# In[ ]:


all_df = all_df.drop(['GarageCars'], axis=1)


# In[ ]:


def OrdinalQ(x):
    if(x=='Ex'):
        val = 0
    elif(x=='Gd'):
        val = 1
    elif(x=='TA'):
        val = 2
    elif(x=='Fa'):
        val = 3
    elif(x=='missing'):
        val = 4
    else:
        val = 5
    return val

all_df['ExterQual'] = all_df['ExterQual'].apply(OrdinalQ)
all_df['ExterCond'] = all_df['ExterCond'].apply(OrdinalQ)
all_df['KitchenQual'] = all_df['KitchenQual'].apply(OrdinalQ)
all_df['HeatingQC'] = all_df['HeatingQC'].apply(OrdinalQ)
all_df['BsmtQual'] = all_df['BsmtQual'].apply(OrdinalQ)
all_df['BsmtCond'] = all_df['BsmtCond'].apply(OrdinalQ)
all_df['FireplaceQu'] = all_df['FireplaceQu'].apply(OrdinalQ)
all_df['GarageQual'] = all_df['GarageQual'].apply(OrdinalQ)


# In[ ]:


def OrdinalSlope(x):
    if(x=='Gtl'):
        val = 0
    elif(x=='Mod'):
        val = 1
    elif(x=='Sev'):
        val = 2
    else:
        val = 3
    return val

all_df['LandSlope'] = all_df['LandSlope'].apply(OrdinalSlope)


# In[ ]:


def OrdinalGarageF(x):
    if(x=='Fin'):
        val = 0
    elif(x=='RFn'):
        val = 1
    elif(x=='Unf'):
        val = 2
    else:
        val = 3
    return val

all_df['GarageFinish'] = all_df['GarageFinish'].apply(OrdinalGarageF)


# In[ ]:


def OrdinalBsmtExp(x):
    if(x=='Gd'):
        val = 0
    elif(x=='Av'):
        val = 1
    elif(x=='Mn'):
        val = 2
    elif(x=='No'):
        val = 3
    else:
        val = 4
    return val

all_df['BsmtExposure'] = all_df['BsmtExposure'].apply(OrdinalBsmtExp)


# In[ ]:


def OrdinalFunc(x):
    if(x=='Typ'):
        val = 0
    elif(x=='Min1' or x=='Min2'):
        val = 1
    elif(x=='Mod'):
        val = 2
    elif(x=='Maj1' or x=='Maj2'):
        val = 3
    elif(x=='Sev'):
        val = 4
    else:
        val = 5
    return val

all_df['Functional'] = all_df['Functional'].apply(OrdinalFunc)


# In[ ]:


def OrdinalBsmtType(x):
    if(x=='GLQ'):
        val = 6
    elif(x=='ALQ'):
        val = 5
    elif(x=='BLQ'):
        val = 4
    elif(x=='Rec'):
        val = 3   
    elif(x=='LwQ'):
        val = 2
    elif(x=='Unf'):
        val = 1        
    else:
        val = 0
    return val

all_df['BsmtFinType1'] = all_df['BsmtFinType1'].apply(OrdinalBsmtType)
all_df['BsmtFinType2'] = all_df['BsmtFinType2'].apply(OrdinalBsmtType)


# In[ ]:


def OrdinalFence(x):
    if(x=='GdPrv'):
        val = 0
    elif(x=='MnPrv'):
        val = 1
    elif(x=='GdWo'):
        val = 2
    elif(x=='MnWw'):
        val = 3
    else:
        val = 4
    return val

all_df['Fence'] = all_df['Fence'].apply(OrdinalFence)


# In[ ]:


def OrdinalSaleCond(x):
    if(x=='Normal'):
        val = 0
    elif(x=='Abnorml'):
        val = 1
    elif(x=='AdjLand'):
        val = 2
    elif(x=='Alloca'):
        val = 3
    elif(x=='Family'):
        val = 4
    else:
        val = 5
    return val

all_df['SaleCondition'] = all_df['SaleCondition'].apply(OrdinalSaleCond)


# In[ ]:


def OrdinalLotShape(x):
    if(x=='Reg'):
        val = 0
    elif(x=='IR1'):
        val = 1
    elif(x=='IR2'):
        val = 2
    else:
        val = 3
    return val

all_df['LotShape'] = all_df['LotShape'].apply(OrdinalLotShape)


# In[ ]:


def OrdinalUtil(x):
    if(x=='AllPub'):
        val = 0
    elif(x=='NoSewr'):
        val = 1
    elif(x=='NoSeWa'):
        val = 2
    else:
        val = 3
    return val

all_df['Utilities'] = all_df['Utilities'].apply(OrdinalUtil)


# In[ ]:


all_df['CentralAir'] = all_df['CentralAir'].apply( lambda x: 0 if x == 'N' else 1) 
all_df['PavedDrive'] = all_df['PavedDrive'].apply( lambda x: 0 if x == 'Y' else 1)
all_df['Street'] = all_df['Street'].apply( lambda x: 0 if x == 'Pave' else 1) 


# In[ ]:


all_df = one_hot_encode(all_df.drop("SalePrice", axis=1), "Alley").join(all_df["SalePrice"])
all_df = one_hot_encode(all_df.drop("SalePrice", axis=1), "BldgType").join(all_df["SalePrice"])
all_df = one_hot_encode(all_df.drop("SalePrice", axis=1), "Condition1").join(all_df["SalePrice"])
all_df = one_hot_encode(all_df.drop("SalePrice", axis=1), "Condition2").join(all_df["SalePrice"])
all_df = one_hot_encode(all_df.drop("SalePrice", axis=1), "Electrical").join(all_df["SalePrice"])
all_df = one_hot_encode(all_df.drop("SalePrice", axis=1), "Exterior1st").join(all_df["SalePrice"])
all_df = one_hot_encode(all_df.drop("SalePrice", axis=1), "Exterior2nd").join(all_df["SalePrice"])
all_df = one_hot_encode(all_df.drop("SalePrice", axis=1), "Foundation").join(all_df["SalePrice"])
all_df = one_hot_encode(all_df.drop("SalePrice", axis=1), "GarageCond").join(all_df["SalePrice"])
all_df = one_hot_encode(all_df.drop("SalePrice", axis=1), "GarageType").join(all_df["SalePrice"])
all_df = one_hot_encode(all_df.drop("SalePrice", axis=1), "Heating").join(all_df["SalePrice"])
all_df = one_hot_encode(all_df.drop("SalePrice", axis=1), "HouseStyle").join(all_df["SalePrice"])
all_df = one_hot_encode(all_df.drop("SalePrice", axis=1), "LandContour").join(all_df["SalePrice"])
all_df = one_hot_encode(all_df.drop("SalePrice", axis=1), "LotConfig").join(all_df["SalePrice"])
all_df = one_hot_encode(all_df.drop("SalePrice", axis=1), "MSZoning").join(all_df["SalePrice"])
all_df = one_hot_encode(all_df.drop("SalePrice", axis=1), "MasVnrType").join(all_df["SalePrice"])
all_df = one_hot_encode(all_df.drop("SalePrice", axis=1), "MiscFeature").join(all_df["SalePrice"])
all_df = one_hot_encode(all_df.drop("SalePrice", axis=1), "Neighborhood").join(all_df["SalePrice"])
all_df = one_hot_encode(all_df.drop("SalePrice", axis=1), "RoofMatl").join(all_df["SalePrice"])
all_df = one_hot_encode(all_df.drop("SalePrice", axis=1), "RoofStyle").join(all_df["SalePrice"])
all_df = one_hot_encode(all_df.drop("SalePrice", axis=1), "SaleType").join(all_df["SalePrice"])


# In[ ]:


all_df.drop(['SalePrice'], axis=1, inplace=True)


# In[ ]:


from sklearn.kernel_ridge import KernelRidge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin, clone
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.metrics import mean_squared_error
import xgboost as xgb


# In[ ]:


train = all_df[:train_len]
test = all_df[train_len:]


# In[ ]:


def rmsle(y, y_pred):
    return np.sqrt(mean_squared_error(y, y_pred))


# In[ ]:


model_xgb = xgb.XGBRegressor(colsample_bytree=0.4603, gamma=0.0468, 
                             learning_rate=0.05, max_depth=3, 
                             min_child_weight=1.7817, n_estimators=2200,
                             reg_alpha=0.4640, reg_lambda=0.8571,
                             subsample=0.5213, silent=1,
                             random_state =7, nthread = -1)


# In[ ]:


model_xgb.fit(train, y_train)
xgb_train_pred = model_xgb.predict(train)
xgb_pred = np.expm1(model_xgb.predict(test))
print(rmsle(y_train, xgb_train_pred))


# In[ ]:


sub = pd.DataFrame()
sub['Id'] = test_id
sub['SalePrice'] = xgb_pred
sub.to_csv('submission.csv',index=False)

