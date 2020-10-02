#!/usr/bin/env python
# coding: utf-8

# **THanks to this Kernel.
# 
# I have used Kernel by this individual to learn a lot.**
# 
# https://www.kaggle.com/serigne/stacked-regressions-top-4-on-leaderboard

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

import seaborn as sns 
import matplotlib.pyplot as plt
from scipy.stats import norm
from sklearn.preprocessing import StandardScaler
from scipy import stats
import warnings
warnings.filterwarnings('ignore')
get_ipython().run_line_magic('matplotlib', 'inline')

# Any results you write to the current directory are saved as output.


# In[ ]:


df_train = pd.read_csv("../input/train.csv")
df_test=pd.read_csv("..//input/test.csv")


train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')

print(train['RoofMatl'].unique())
print(test['RoofMatl'].unique())


# In[ ]:


df_test.shape
#df_train.shape[1]


# In[ ]:


df_train.shape


# In[ ]:



df_train = df_train.drop(df_train[(df_train['GrLivArea']>4000) & (df_train['SalePrice']<300000)].index)

df_all=df_train.drop(['SalePrice'],axis=1).append(df_test)


# In[ ]:


df_train.drop(['SalePrice'],axis=1).append(df_test)[:1458].head()


# In[ ]:


#data_input=df_train.drop(['Id','SalePrice'],axis=1)
#df.drop(['B', 'C'], axis=1)
#data_output=df_train['SalePrice']
df_train.head()
df_test.head()

#df_train.drop(['SalePrice'],axis=1).append(df_test)


# In[ ]:


df_all.head()


# In[ ]:


total=df_all.isnull().sum()/df_all.isnull().count()
sum=df_all.isnull().sum()
#total.head
#sum.head
missing=pd.concat([total,sum],axis=1,keys=['Perc','Sum']).sort_values(by='Perc',ascending=False)
#missing.keys=['Percentage','Sum']
colstodrop=missing[missing['Sum']>0].index
#colstodrop
missing[missing['Sum']>0]
 
#data_input=data_input.drop(colstodrop,axis=1)
#data_test=df_test.drop(colstodrop,axis=1)
#data_input.head(5)


# PoolQC	0.996574	2909
# MiscFeature	0.964029	2814
# Alley	0.932169	2721
# Fence	0.804385	2348
# FireplaceQu	0.486468	1420
# LotFrontage	0.166495	486
# GarageYrBlt	0.054471	159
# GarageFinish	0.054471	159
# GarageQual	0.054471	159
# GarageCond	0.054471	159
# GarageType	0.053786	157
# BsmtExposure	0.028092	82
# BsmtCond	0.028092	82
# BsmtQual	0.027749	81
# BsmtFinType2	0.027407	80
# BsmtFinType1	0.027064	79
# MasVnrType	0.008222	24
# MasVnrArea	0.007879	23
# MSZoning	0.001370	4
# Functional	0.000685	2
# BsmtHalfBath	0.000685	2
# BsmtFullBath	0.000685	2
# Utilities	0.000685	2

# In[ ]:


df_all['MiscFeature']=df_all['MiscFeature'].fillna("None")
df_all['Alley']=df_all['Alley'].fillna("None")
df_all['Fence']=df_all['Fence'].fillna("None")
df_all['FireplaceQu']=df_all['FireplaceQu'].fillna("None")
df_all['GarageFinish']=df_all['GarageFinish'].fillna("None")
df_all['GarageQual']=df_all['GarageFinish'].fillna("None")
df_all['GarageType']=df_all['GarageType'].fillna("None")
df_all['BsmtCond']=df_all['BsmtCond'].fillna("None")
df_all['BsmtExposure']=df_all['BsmtExposure'].fillna("Nobase")
df_all['BsmtQual']=df_all['BsmtQual'].fillna("None")
df_all['BsmtFinType2']=df_all['BsmtFinType2'].fillna("None")
df_all['BsmtFinType1']=df_all['BsmtFinType1'].fillna("None")
df_all['GarageCond']=df_all['GarageCond'].fillna("None")
df_all['PoolQC']=df_all['PoolQC'].fillna("None")
df_all['MasVnrType']=df_all['MasVnrType'].fillna("None")
df_all['MasVnrArea']=df_all['MasVnrArea'].fillna(0)
df_all['Functional']=df_all['Functional'].fillna("Typ")
#df_all['Utilities']=df_all['Utilities'].fillna("AllPub")
df_all=df_all.drop(['Utilities'],axis=1)
df_all['MSZoning']=df_all['MSZoning'].fillna(df_all['MSZoning'].mode()[0])
df_all['Electrical']=df_all['Electrical'].fillna(df_all['Electrical'].mode()[0])
df_all["LotFrontage"] = df_all.groupby("Neighborhood")["LotFrontage"].transform(
    lambda x: x.fillna(x.median()))
for col in ['GarageYrBlt','GarageArea','GarageCars']:
    df_all[col]=df_all[col].fillna(0)
    
for col in ['BsmtFinSF1','BsmtFinSF2','BsmtUnfSF','TotalBsmtSF','BsmtFullBath','BsmtHalfBath']:
    df_all[col]=df_all[col].fillna(0)
    
for col in ['SaleType','Exterior1st','Exterior2nd','KitchenQual']:
    df_all[col]=df_all[col].fillna(df_all[col].mode()[0])
    
    
#Introduce some new fields


df_all['Total_sqr_footage'] = (df_all['BsmtFinSF1'] + df_all['BsmtFinSF2'] +
                                 df_all['1stFlrSF'] + df_all['2ndFlrSF'])

df_all['Total_Bathrooms'] = (df_all['FullBath'] + (0.5 * df_all['HalfBath']) +
                               df_all['BsmtFullBath'] + (0.5 * df_all['BsmtHalfBath']))

df_all['Total_porch_sf'] = (df_all['OpenPorchSF'] + df_all['3SsnPorch'] +
                              df_all['EnclosedPorch'] + df_all['ScreenPorch'] +
                              df_all['WoodDeckSF'])


# In[ ]:


df_all['Exterior1st'].mode()[0]


# In[ ]:


#test piece for doing qa on missing values
#df_all['Utilities'].mode()[0]
df_all.isna().sum().sort_values(ascending=False)


# In[ ]:


df_all.head()


# In[ ]:


df_all['BsmtFinSF1']


# In[ ]:


#colstodrop


# data_test=data_test.drop(['Id'],axis=1)
# data_test.shape

# data_test=pd.get_dummies(data_test)
# data_input=pd.get_dummies(data_input)

# In[ ]:


df_all.columns


# In[ ]:


#df_all['YrSold']=df_all['YrSold'].apply(str)
#df_all['MoSold']=df_all['MoSold'].apply(str)
#df_all['OverallQual']=df_all['OverallQual'].apply(str)
#df_all['OverallCond']=df_all['OverallCond'].apply(str)
#df_all['MSSubClass']=df_all['MSSubClass'].apply(str)
#df_all['YearBuilt']=df_all['YearBuilt'].apply(str)
#df_all['YearRemodAdd']=df_all['YearRemodAdd'].apply(str)
#df_all['GarageYrBlt']=df_all['GarageYrBlt'].apply(str)
df_all=df_all.drop(['Id'],axis=1)
df_all['Totalarea']=df_all['TotalBsmtSF'] + df_all['1stFlrSF'] + df_all['2ndFlrSF']


# In[ ]:


numericcols=df_all.dtypes[df_all.dtypes != object]
categorcols=df_all.dtypes[df_all.dtypes == object]


# In[ ]:


numericcols.index


# In[ ]:


categorcols.index


# In[ ]:


catcols=('FireplaceQu', 'BsmtQual', 'BsmtCond', 'GarageQual', 'GarageCond', 
        'ExterQual', 'ExterCond','HeatingQC', 'PoolQC', 'KitchenQual', 'BsmtFinType1', 
        'BsmtFinType2', 'Functional', 'Fence', 'BsmtExposure', 'GarageFinish', 'LandSlope',
        'LotShape', 'PavedDrive', 'Street', 'Alley', 'CentralAir',  'OverallCond', 
        'YrSold', 'MoSold','MSSubClass')
#'MSSubClass',
catcols


#     Label Encoding

# In[ ]:





# In[ ]:


df_all.shape



# In[ ]:


df_all.dtypes[df_all.dtypes == "object"].index


# In[ ]:


print(df_all.dtypes.unique())
print(df_all['MSSubClass'].dtypes)

df_all['MSSubClass']


# In[ ]:



df_all.shape
#df_all['MSSubClass'].dtype
#print(numeric_feats)
#df_all['MSSubClass']=str(df_all['MSSubClass'])
df_all['MSSubClass']=df_all['MSSubClass'].apply(str)


# In[ ]:


df_all['MSSubClass']


# In[ ]:



from scipy import stats
from scipy.stats import norm, skew #for some statistics

df_all.dtypes

#df_all=df_all.drop('SalePrice',axis=1)
numeric_feats = df_all.dtypes[df_all.dtypes != "object"].index

# Check the skew of all numerical features
skewed_feats = df_all[numeric_feats].apply(lambda x: skew(x.dropna())).sort_values(ascending=False)
print("\nSkew in numerical features: \n")
skewness = pd.DataFrame({'Skew' :skewed_feats})
skewness.head(10)
skewness=skewness[abs(skewness['Skew'])>0.75]
skewness
print(df_all.shape)


# In[ ]:


numeric_feats


# In[ ]:


df_all['MiscVal'].skew()


# In[ ]:


#skewness=skewness[abs(skewness)>1.25]
#skewness.count()

from scipy.special import boxcox1p
from scipy.stats import boxcox_normmax 
skewed_features = skewness.index

lam = 0.15
for feat in skewed_features:
    #df_all[feat] = boxcox1p(df_all[feat], lam)
    df_all[feat] = boxcox1p(df_all[feat], boxcox_normmax(df_all[feat] + 1))
    #df_all[feat] = np.log1p(df_all[feat])
    
 #   df_all[feat] += 1
  #df_all[feat] = np.log1p(df_all[feat])
#for feat in skewed_features:
#    df_all[feat]=np.log1p(df_all[feat])


# In[ ]:


# from sklearn.preprocessing import LabelEncoder
# for c in catcols:
#     lbl = LabelEncoder() 
#     lbl.fit(list(df_all[c].values)) 
#     d=c+"_e"
#     #print(d)
#     df_all[d] = lbl.transform(list(df_all[c].values)) 
#    #df_all[c]=np.log1p(df_all[c])


# In[ ]:


df_all.shape


# In[ ]:


def encode(frame, feature):
    ordering = pd.DataFrame()
    ordering['val'] = df_all[feature].unique()
    ordering.index = ordering.val
    ordering['spmean'] = frame[[feature, 'SalePrice']].groupby(feature).mean()['SalePrice']
    ordering = ordering.sort_values('spmean')
    ordering['ordering'] = range(1, ordering.shape[0]+1)
    ordering = ordering['ordering'].to_dict()
    
    for cat, o in ordering.items():
        df_all.loc[df_all[feature] == cat, feature+'_E'] = o
    
qual_encoded = []
for q in catcols:  
    encode(df_train, q)
    #df_train.append(q+'_E')
print(qual_encoded)


# In[ ]:


#df_train.SalePrice
#df_all['MSSubClass']
df_all.shape


# In[ ]:


# def encode2(frame, feature):
#     ordering = pd.DataFrame()
#     ordering['val'] = df_all[feature].unique()
#     ordering.index = ordering.val
#     ordering['spmean'] = frame[[feature, 'SalePrice']].groupby(feature).mean()['SalePrice']
#     ordering = ordering.sort_values('spmean')
#     ordering['ordering'] = range(1, ordering.shape[0]+1)
#     #print(ordering)
#     ordering = ordering['ordering'].to_dict()
#     #print(ordering.items())
#     #print(ordering)
    
#     for cat, o in ordering.items():
#         #df_all.loc[df_all[feature] == cat, feature+'_E'] = o
#         print(feature,o,cat)
#         #print(o)
        

# for q in catcols:  
#     encode2(df_train, q)
#     #df_train.append(q+'_E')
    
    
# for c in catcols:
#     lbl = LabelEncoder() 
#     lbl.fit(list(df_all[c].values)) 
#     df_all[c] = lbl.transform(list(df_all[c].values))


# In[ ]:


df_all.head()


# In[ ]:


df_train[['MSZoning','SalePrice']].groupby('MSZoning').mean()
#frame[[feature, 'SalePrice']].groupby(feature).mean()['SalePrice']


# In[ ]:


df_all.isna().sum().sort_values(ascending=False)
#df_all[df_all['GarageQual_E'].isna()].head()#['GarageQual']
#df_all[df_all['GarageQual_E'].isna().GarageQual].head()#['GarageQual']


# In[ ]:


#df_train.iloc[:,81:]


# In[ ]:


#df_all['3SsnPorch']
numeric_feats = df_all.dtypes[df_all.dtypes != "object"].index
numeric_feats

skewed_feats = df_all[numeric_feats].apply(lambda x: skew(x.dropna())).sort_values(ascending=False)
skewness = pd.DataFrame({'Skew' :skewed_feats}) 
#skewness=skewness[abs(skewness['Skew'])>1]
skewness


# In[ ]:


skewed_feats


# In[ ]:


df_all.shape


# In[ ]:





# In[ ]:


df_all['haspool'] = df_all['PoolArea'].apply(lambda x: 1 if x > 0 else 0)
df_all['has2ndfloor'] = df_all['2ndFlrSF'].apply(lambda x: 1 if x > 0 else 0)
df_all['hasgarage'] = df_all['GarageArea'].apply(lambda x: 1 if x > 0 else 0)
df_all['hasbsmt'] = df_all['TotalBsmtSF'].apply(lambda x: 1 if x > 0 else 0)
df_all['hasfireplace'] = df_all['Fireplaces'].apply(lambda x: 1 if x > 0 else 0)


# In[ ]:


df_all=pd.get_dummies(df_all).reset_index(drop=True)
df_all.shape


# In[ ]:


skewed_features


# In[ ]:


df_all.shape


# In[ ]:


df_all.columns


# In[ ]:


from sklearn.linear_model import ElasticNet, Lasso,  BayesianRidge, LassoLarsIC,LassoCV
from sklearn.ensemble import RandomForestRegressor,  GradientBoostingRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin, clone
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.metrics import mean_squared_error
#from sklearn.linear_model import mean_squared_error
import xgboost as xgb
import lightgbm as lgb


# In[ ]:


from sklearn.linear_model import LinearRegression


# In[ ]:


linreg = LinearRegression()


# In[ ]:


for cat in df_all.columns:
    print(cat)

print(df_train['RoofMatl'].unique())
print(df_train['RoofMatl'].unique())


# In[ ]:


df_all


# In[ ]:


data_input=df_all[:1458]
data_test=df_all[1458:]
data_output=np.log1p(df_train['SalePrice']).values
data_output.shape


# In[ ]:


data_test.shape
data_output


# In[ ]:


data_input.shape


# In[ ]:


data_test.shape


# In[ ]:


# for col in data_test.columns:
#     if data_test[col].sum() == 0:
#         data_input=data_input.drop(col,axis=1)
#         data_test =data_test.drop(col,axis=1)


# In[ ]:


print(data_input.shape)
print(data_test.shape)
print(data_output.shape)


# In[ ]:



#data_test=df_test.drop(['Id'],axis=1)
#data_test=pd.get_dummies(data_test)

#data_output=np.log1p(df_train['SalePrice'])
#data_input=df_train.drop(['Id','SalePrice'],axis=1)
#data_input=df_train.drop(colstodrop,axis=1)
#data_input=pd.get_dummies(data_input)
#data_=pd.get_dummies(data_input)
#


# In[ ]:


#traincols=data_test.columns
#testcols=data_input.columns
#traincols

#def common_member(a, b): 
#    a_set = set(a) 
#    b_set = set(b) 
#    if (a_set & b_set): 
#        return list(a_set & b_set) 
#    else: 
#        print("No common elements")  
           
#modelcols=common_member(traincols, testcols) 


# In[ ]:


#data_input=data_input[modelcols]
#modelcols
#data_input
print(data_input.shape)
print(data_output.shape)


# In[ ]:


#Validation function
n_folds = 5

def rmsle_cv(model):
    kf = KFold(n_folds, shuffle=True, random_state=42).get_n_splits(data_input.values)
    rmse= np.sqrt(-cross_val_score(model, data_input.values, data_output, scoring="neg_mean_squared_error", cv = kf))
    return(rmse)


# In[ ]:


data_output


# In[ ]:


linreg.fit(data_input, data_output)
#linreg_output=linreg.predict(data_input)
rmsle_cv(linreg).mean()


# In[ ]:


fig,ax=plt.subplots()
ax.scatter(x=data_output,y=linreg.predict(data_input))


# In[ ]:


linreg_output=linreg.predict(data_test)


# In[ ]:


linreg_output[abs(linreg_output)>15]


# In[ ]:


data_input.shape
type(data_input)


# In[ ]:


data_input.iloc[120:290]


# In[ ]:


# data_input.shape

# kf=KFold(5)
# kf.split(data_input,data_output)
# k=list(range(100))
# for x,y in kf.split(data_input,data_output):
#     print (x,y)
#     x=list(x)
#     print(data_input[x])
#     print(type(x))


# #i=0
# #j=np.zeros(20)
# #for i in range(0,20):
#     #print(i)
# #    j[i]=i
# #    i+=1
#     
# kf=KFold(5)
# #print(j)
# #print(j.shape)
# lasso = Lasso(alpha =0.0005, random_state=1)
# ENet =  ElasticNet(alpha=0.0005, l1_ratio=.9, random_state=1)
# 
# #print(da)
# print(data_input.shape[0])
# print(data_output.shape[0])
# predictions=np.zeros((data_input.shape[0],2))
# 
# i=0
# for x,y in kf.split(data_input,data_output):
#     lasso.fit(data_input.iloc[x],data_output.iloc[x])
#     ENet.fit(data_input,data_output)
# #     print(i)
# #     i+=1
# #     if(data_input.loc[x].isna()):
#     #print(x)
#     #output=data_input.loc[x]
#     #print(data_input.loc[x].isna().sum())
#     #data_input[data_input.isnull().any(axis=1)]
#     #print(data_output.loc[x].shape)
#     #print(y)
#     predictions[y,0]=lasso.predict(data_input.iloc[y])
#     predictions[y,1]=ENet.predict(data_input.iloc[y])
#     #predictions[y,1]=lasso.predict(data_input.loc[y])
#     #print(data_output.loc[x])
#     #data_input.loc[x] 
# #print(predictions)
# #xgb_output=model_xgb.predict(data_test)
# #rmsle_cv(model_xgb).mean()
# model_xgb = xgb.XGBRegressor(colsample_bytree=0.4603, gamma=0.0468, 
#                              learning_rate=0.05, max_depth=3, 
#                              min_child_weight=1.7817, n_estimators=2200,
#                              reg_alpha=0.4640, reg_lambda=0.8571,
#                              subsample=0.5213, silent=1,
#                              random_state =7, nthread = -1)
# #model_xgb.fit(data_input,data_output)
# model_xgb.fit(predictions,data_output)
# print(rmsle_cv(model_xgb).mean())
# print(rmsle_cv(model_xgb).std())
# predictions.shape    
# print(predictions)

# In[ ]:


# data_input.loc[data_input['1stFlrSF'].isnull()]
# output.isnull().sum()
# output[output['MSSubClass'].isnull()]
# data_input.iloc[523,:]


# In[ ]:


# data_input.shape
# data_output.shape

# #data_output.loc[290]
# print(predictions.shape)

# #predictions=np.zeros(df_train.shape[0])
# print(predictions.shape)
# print(data_output.shape)
# print(predictions)


#predictions=np.zeros((df_train.shape[0],1))


# In[ ]:


# data_input.head()
# data_output.head()
df_all.shape


# > 

# In[ ]:


#lasso = make_pipeline(RobustScaler(), Lasso(alpha =0.0005, random_state=1))



alphas2 = [5e-05, 0.0001, 0.0002, 0.0003, 0.0004, 0.0005, 0.0006, 0.0007, 0.0008]
kfolds = KFold(n_splits=10, shuffle=True, random_state=42)
lasso = make_pipeline(RobustScaler(), LassoCV(max_iter=1e7, alphas=alphas2, random_state=42, cv=kfolds))
#lasso = Lasso(alpha =0.0005, random_state=1)
lasso.fit(data_input,data_output)

lasso_output_train=lasso.predict(data_input)
lasso_output=lasso.predict(data_test)
print(rmsle_cv(lasso).mean())
print(rmsle_cv(lasso).std())


# In[ ]:


resid=abs(lasso_output_train-data_output)
out_mean=abs(lasso_output_train-data_output).mean()
out_std=abs(lasso_output_train-data_output).std()
z=(resid-out_mean)/out_std
#out_border
z=np.array(z)
#outliers=
outliers=np.where(abs(z)>abs(z).std()*1.25)[0]
outliers


# In[ ]:


#detect outliers

#np.where(resid > out_mean+3*out_std  )
#data_input=data_input.drop([outliers],axis=0)
#data_input
# data_input_drop_out=data_input.drop([30,   66,   88,  142,  185,  277,  308,  328,  410,  431,  462,
#         479,  495,  559,  580,  587,  627,  631,  657,  665,  680,  687,
#         709,  710,  713,  727,  773,  811,  873,  897,  967,  969, 1021,
#        1061, 1121, 1138, 1180, 1210, 1211, 1322, 1381, 1430, 1451])
# data_output_drop_out=np.delete(data_output,[30,   66,   88,  142,  185,  277,  308,  328,  410,  431,  462,
#         479,  495,  559,  580,  587,  627,  631,  657,  665,  680,  687,
#         709,  710,  713,  727,  773,  811,  873,  897,  967,  969, 1021,
#        1061, 1121, 1138, 1180, 1210, 1211, 1322, 1381, 1430, 1451],0)
# type(data_input)
# type(data_output)

data_input_drop_out=data_input.drop([   3,    4,   13,   17,   22,   24,   30,   35,   38,   48,   59,
         66,   70,   76,   88,   97,  107,  142,  154,  157,  169,  175,
        181,  185,  193,  198,  216,  217,  218,  223,  225,  238,  242,
        250,  251,  261,  268,  270,  275,  277,  286,  291,  308,  318,
        328,  329,  330,  335,  347,  348,  358,  365,  371,  377,  378,
        393,  397,  401,  410,  418,  431,  439,  441,  445,  451,  457,
        462,  473,  479,  488,  495,  503,  507,  512,  528,  534,  543,
        544,  545,  557,  558,  559,  580,  587,  588,  606,  607,  625,
        627,  629,  631,  638,  651,  657,  661,  665,  668,  679,  680,
        687,  704,  706,  709,  710,  713,  714,  715,  716,  725,  727,
        737,  739,  743,  746,  770,  771,  773,  788,  795,  796,  802,
        807,  808,  811,  854,  863,  873,  884,  895,  897,  901,  914,
        915,  922,  934,  939,  941,  944,  962,  967,  969,  971,  972,
        989,  999, 1021, 1025, 1029, 1045, 1048, 1055, 1061, 1064, 1067,
       1074, 1079, 1091, 1121, 1130, 1138, 1142, 1144, 1149, 1162, 1167,
       1177, 1178, 1180, 1182, 1183, 1184, 1199, 1201, 1204, 1210, 1211,
       1214, 1215, 1218, 1224, 1243, 1246, 1251, 1261, 1266, 1275, 1281,
       1302, 1320, 1322, 1323, 1335, 1336, 1342, 1343, 1348, 1357, 1376,
       1378, 1380, 1381, 1384, 1413, 1421, 1425, 1430, 1441, 1451])

data_output_drop_out=np.delete(data_output,[   3,    4,   13,   17,   22,   24,   30,   35,   38,   48,   59,
         66,   70,   76,   88,   97,  107,  142,  154,  157,  169,  175,
        181,  185,  193,  198,  216,  217,  218,  223,  225,  238,  242,
        250,  251,  261,  268,  270,  275,  277,  286,  291,  308,  318,
        328,  329,  330,  335,  347,  348,  358,  365,  371,  377,  378,
        393,  397,  401,  410,  418,  431,  439,  441,  445,  451,  457,
        462,  473,  479,  488,  495,  503,  507,  512,  528,  534,  543,
        544,  545,  557,  558,  559,  580,  587,  588,  606,  607,  625,
        627,  629,  631,  638,  651,  657,  661,  665,  668,  679,  680,
        687,  704,  706,  709,  710,  713,  714,  715,  716,  725,  727,
        737,  739,  743,  746,  770,  771,  773,  788,  795,  796,  802,
        807,  808,  811,  854,  863,  873,  884,  895,  897,  901,  914,
        915,  922,  934,  939,  941,  944,  962,  967,  969,  971,  972,
        989,  999, 1021, 1025, 1029, 1045, 1048, 1055, 1061, 1064, 1067,
       1074, 1079, 1091, 1121, 1130, 1138, 1142, 1144, 1149, 1162, 1167,
       1177, 1178, 1180, 1182, 1183, 1184, 1199, 1201, 1204, 1210, 1211,
       1214, 1215, 1218, 1224, 1243, 1246, 1251, 1261, 1266, 1275, 1281,
       1302, 1320, 1322, 1323, 1335, 1336, 1342, 1343, 1348, 1357, 1376,
       1378, 1380, 1381, 1384, 1413, 1421, 1425, 1430, 1441, 1451],0)


# In[ ]:


print(data_output_drop_out.shape)
print(data_input_drop_out.shape)


# In[ ]:


#lasso = make_pipeline(RobustScaler(), Lasso(alpha =0.0005, random_state=1))



alphas2 = [5e-05, 0.0001, 0.0002, 0.0003, 0.0004, 0.0005, 0.0006, 0.0007, 0.0008]
kfolds = KFold(n_splits=10, shuffle=True, random_state=42)
lasso = make_pipeline(RobustScaler(), LassoCV(max_iter=1e7, alphas=alphas2, random_state=42, cv=kfolds))
#lasso = Lasso(alpha =0.0005, random_state=1)
lasso.fit(data_input_drop_out,data_output_drop_out)

lasso_output_train=lasso.predict(data_input_drop_out)
lasso_output=lasso.predict(data_test)
#print(rmsle_cv(lasso).mean())
#print(rmsle_cv(lasso).std())

print(np.sqrt(-cross_val_score(lasso, data_input_drop_out, data_output_drop_out, cv=5, scoring="neg_mean_squared_error")).mean())

print(np.sqrt(-cross_val_score(lasso, data_input, data_output, cv=5, scoring="neg_mean_squared_error")).mean())


# In[ ]:


# xgb_output=model_xgb.predict(lasso_output.reshape(-1,1))
print(df_all.shape)


# In[ ]:


# xgb_output


# In[ ]:


# df_all.shape


# In[ ]:


# data_test.shape


# In[ ]:


model_xgb = xgb.XGBRegressor(colsample_bytree=0.4603, gamma=0.0468, 
                             learning_rate=0.05, max_depth=3, 
                             min_child_weight=1.7817, n_estimators=2200,
                             reg_alpha=0.4640, reg_lambda=0.8571,
                             subsample=0.5213, silent=1,
                             random_state =7, nthread = -1)
model_xgb.fit(data_input_drop_out,data_output_drop_out)
xgb_output_train=model_xgb.predict(data_input)
xgb_output=model_xgb.predict(data_test)
print(np.sqrt(-cross_val_score(model_xgb, data_input_drop_out, data_output_drop_out, cv=5, scoring="neg_mean_squared_error")).mean())

#rmsle_cv(model_xgb).mean()


# In[ ]:




from sklearn.linear_model import Ridge

rr = Ridge(alpha=13)
rr.fit(data_input_drop_out, data_output_drop_out)
np.sqrt(-cross_val_score(rr, data_input_drop_out, data_output_drop_out, cv=5, scoring="neg_mean_squared_error")).mean()


# In[ ]:



# from sklearn.ensemble import RandomForestRegressor

# regr = RandomForestRegressor(max_depth=2, random_state=0,
#                               n_estimators=100)
# regr.fit(data_input_drop_out,data_output_drop_out)
# print(np.sqrt(-cross_val_score(regr, data_input_drop_out, data_output_drop_out, cv=5, scoring="neg_mean_squared_error")).mean())


# In[ ]:


ENet =  ElasticNet(alpha=0.0005, l1_ratio=.9, random_state=1)
ENet.fit(data_input_drop_out,data_output_drop_out)
ENet_output_train=ENet.predict(data_input)
ENet_output=ENet.predict(data_test)

print(np.sqrt(-cross_val_score(ENet, data_input_drop_out, data_output_drop_out, cv=5, scoring="neg_mean_squared_error")).mean())

#rmsle_cv(ENet).mean()


# In[ ]:


#averaged_models = AveragingModels(models = (ENet,   lasso))


# In[ ]:



# stk_inp_train=np.column_stack([lasso_output_train,ENet_output_train,xgb_output_train])
# GBoost = GradientBoostingRegressor(n_estimators=3000, learning_rate=0.05,
#                                    max_depth=4, max_features='sqrt',
#                                    min_samples_leaf=15, min_samples_split=10, 
#                                    loss='huber', random_state =5)
# GBoost.fit(stk_inp_train, data_output)


# rmsle_cv(GBoost).mean()
# #linreg_output=linreg.predict(data_input)
# #stk_inp_test.shape


# In[ ]:


#df_all.Id
stk_inp_train=np.column_stack([lasso_output_train,ENet_output_train])


# In[ ]:


linreg.fit(stk_inp_train,data_output)
# print(linreg.coef_)
#rmsle_cv(linreg).mean()
# #linreg.predict(stk_inp_test)[abs(linreg.predict(stk_inp_test))>15]


# In[ ]:


# linreg.predict(stk_inp_train)


# In[ ]:


# data_test.isnull().sum().sort_values(ascending = False).head(5)


# In[ ]:


testids=df_test['Id']
# print(testids.shape)
# print(data_test.shape) 
# print(data_test.columns)

# i=0

#dat_test[data_test.groupby('MSSubClass').sum() == 0


# In[ ]:


# print(testcolsnoval)


# In[ ]:


lasso_output =lasso.predict(data_test)
# ENet_output=ENet.predict(data_test)
# xgb_output=model_xgb.predict(data_test)
#xgb_output=model_xgb.predict(stk_inp_test))
# stk_inp_test=np.column_stack([lasso_output,ENet_output,xgb_output])
#test_output=GBoost.predict(stk_inp_test) # commenting out to submit lasso
test_output=lasso_output
#test_output=xgb_output


# In[ ]:


#test_output=0.33*lasso_output+0.33*ENet_output+0.34*xgb_output
#test_output=xgb_output
#test_output=lasso_output
#lasso_output.shape

#test_output=GBoost.predict(stk_inp_test)


# In[ ]:


results=pd.concat([testids,pd.Series(np.expm1(test_output))],axis=1,keys=['Id','SalePrice'])


# In[ ]:


results.head(5)


# In[ ]:


results.to_csv('../working/submissionslassooutlierremoval4.csv',index=False)


# In[ ]:



print(os.listdir("../working"))


# In[ ]:


df_train.head()


# In[ ]:


data_test.head()

