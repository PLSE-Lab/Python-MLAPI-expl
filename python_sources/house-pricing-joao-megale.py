#!/usr/bin/env python
# coding: utf-8

# # House Pricing by Joao Megale

# ### basic idea: 
# 1. Brush data (fill missing, add features, log...);
# 2. Run a simple XGBoost regression;
# 3. Use Shap to find and remove irrelevant features from brushed-reduced dataset and
# 4. Tune a XGBoost regression to finish: top 33%
# 

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
from sklearn import preprocessing
import seaborn as sns
from scipy.stats import skew
import warnings
warnings.filterwarnings("ignore")
import os

def checknan(ds):
    dsm = pd.DataFrame(ds.isnull().sum()/ds.shape[0], columns=['nan%'])
    return (dsm[dsm['nan%']>0]['nan%'].sort_values(ascending = False))

def countnan(ds):
    dsm = pd.DataFrame(ds.isnull().sum(), columns=['nan'])
    return (dsm[dsm['nan']>0]['nan'].sort_values(ascending = False))


# In[ ]:


#dftest = pd.read_csv('test.csv')
#dftrain = pd.read_csv('train.csv')
#dftest.shape, dftrain.shape
dftrain = pd.read_csv("../input/train.csv")
dftest = pd.read_csv("../input/test.csv")


# ## remove train dataset outliers

# In[ ]:


corr = dftrain.corr()
corr2 = dftrain[corr.SalePrice.sort_values(ascending=False).head(11).index]
sns.pairplot(corr2)


# In[ ]:


corr2.columns


# In[ ]:


plt.scatter(dftrain['GrLivArea'], dftrain['SalePrice'])


# In[ ]:


dftrain[(dftrain.SalePrice<300000) & (dftrain.GrLivArea>4000)]


# In[ ]:


dftrain.drop([523, 1298], axis=0, inplace=True)


# In[ ]:


plt.scatter(dftrain['GrLivArea'], dftrain['SalePrice'])


# In[ ]:


corr = dftrain.corr()
corr2 = dftrain[corr.SalePrice.sort_values(ascending=False).head(11).index]
sns.pairplot(corr2)


# In[ ]:


corr2.columns


# In[ ]:


def bp(feature):
    fig = plt.figure(figsize=(10,3))
    ax1 = fig.add_subplot(121)
    ax1 = plt.boxplot((dftrain[feature]))
    plt.xlabel(feature)
    ax2 = fig.add_subplot(122)
    ax2 = plt.boxplot(np.log1p(dftrain[feature]))
    plt.xlabel(feature+' log')
    plt.show


# In[ ]:


for featr in corr2.columns:
    bp(featr)


# In[ ]:


'''took a look but decided apply log to all numeric features.. later on this notebook'''


# In[ ]:


'''reset dataframe to start point'''
df = dftrain
df = df.append(dftest, sort=True)
df.shape


# ### check missing data

# In[ ]:


checknan(df)


# ### classify nan features analysis...

# In[ ]:


'''check missing in a df'''
nan_ck = pd.DataFrame(df.isnull().sum(), columns = ['Nan_sum'])
nan_ck = nan_ck.drop('SalePrice')
nan_ck['Nan_cnt'] = pd.DataFrame(df.isnull().count())
nan_ck['Nan%'] = nan_ck['Nan_sum'] / nan_ck['Nan_cnt']
nan_ck = nan_ck[nan_ck['Nan%'] != 0].sort_values(['Nan%'], ascending = False)
nan_ck.head(80)


# ### hi, low, and mid missing data levels...

# In[ ]:


'''low missing data'''
lowmd = nan_ck[(nan_ck['Nan_sum'] <= 4) & (nan_ck['Nan_sum'] > 0)]
'''mid missing data'''
midmd = nan_ck[(nan_ck['Nan_sum'] <= 486) & (nan_ck['Nan_sum'] > 4)]
'''hi missing data'''
himd = nan_ck[(nan_ck['Nan_sum'] > 486)]


# In[ ]:


print('himd:\n',list(himd.index),'\n\nlowmd:\n', list(lowmd.index), '\n\nmidmd:\n', list(midmd.index))


# #### Hi missing data  analysis

# In[ ]:


df[himd.index].info()


# In[ ]:


df['PoolQC'].value_counts(), df['MiscFeature'].value_counts(), df['Alley'].value_counts(), df['Fence'].value_counts(), df['FireplaceQu'].value_counts()


# In[ ]:


df[himd.index] = df[himd.index].fillna('NA')


# #### Low missing data  analysis

# In[ ]:


df[lowmd.index].info()


# In[ ]:


df[lowmd.index].describe()


# In[ ]:


df[lowmd.index]=df[lowmd.index].fillna(df[lowmd.index].median())


# In[ ]:


df[lowmd.index].describe()


# In[ ]:


lowmd1 = df[checknan(df[lowmd.index]).index]
lowmd1.head()


# In[ ]:


print (lowmd1['MSZoning'].value_counts().head(2),'\n', '-' * 25)
print (lowmd1['Functional'].value_counts().head(2),'\n', '-' * 25)
print (lowmd1['Utilities'].value_counts().head(2),'\n', '-' * 25)
print (lowmd1['SaleType'].value_counts().head(2),'\n', '-' * 25)
print (lowmd1['Electrical'].value_counts().head(2),'\n', '-' * 25)
print (lowmd1['KitchenQual'].value_counts().head(2),'\n', '-' * 25)
print (lowmd1['Exterior1st'].value_counts().head(2),'\n', '-' * 25)
print (lowmd1['Exterior2nd'].value_counts().head(2))


# In[ ]:


df['MSZoning'] = df['MSZoning'].fillna('RL')
df['Functional'] = df['Functional'].fillna('Typ')
df['Utilities'] = df['Utilities'].fillna('AllPub')
df['SaleType'] = df['SaleType'].fillna('WD')
df['Electrical'] = df['Electrical'].fillna('SBrkr')
df['KitchenQual'] = df['KitchenQual'].fillna('TA')
df['Exterior1st'] = df['Exterior1st'].fillna('VinylSd')
df['Exterior2nd'] = df['Exterior2nd'].fillna('VinylSd')


# #### Mid missing data  analysis

# In[ ]:


df[midmd.index].info()


# In[ ]:


checknan(df[midmd.index].select_dtypes(include=np.number))


# In[ ]:


print( 'normal >', df['LotFrontage'].skew(), ' -- log >', np.log1p(df['LotFrontage']).skew())
print( 'normal >', df['GarageYrBlt'].skew(), ' -- log >', np.log1p(df['GarageYrBlt']).skew())
print( 'normal >', df['MasVnrArea'].skew(), ' -- log >', np.log1p(df['MasVnrArea']).skew())


# In[ ]:


fig = plt.figure(figsize=(20,15))

ax1 = fig.add_subplot(321)
ax1.hist(df[df['LotFrontage']>=0]['LotFrontage'], bins=18)
plt.xlabel('LotFrontage')
ax2 = fig.add_subplot(322)
ax2.hist(np.log1p(df[df['LotFrontage']>=0]['LotFrontage']), bins=18)
plt.xlabel('LotFrontage Log')

ax3 = fig.add_subplot(323)
ax3.hist(df[df['GarageYrBlt']>=0]['GarageYrBlt'], bins=18)
plt.xlabel('GarageYrBlt')
ax4 = fig.add_subplot(324)
ax4.hist(np.log1p(df[df['GarageYrBlt']>=0]['GarageYrBlt']), bins=18)
plt.xlabel('GarageYrBlt Log')

ax5 = fig.add_subplot(325)
ax5.hist(df[df['MasVnrArea']>=0]['MasVnrArea'], bins=18)
plt.xlabel('MasVnrArea')
ax6 = fig.add_subplot(326)
ax6.hist(np.log1p(df[df['MasVnrArea']>=0]['MasVnrArea']), bins=18)
plt.xlabel('MasVnrArea Log')

plt.show()


# In[ ]:


df['MasVnrArea'].value_counts().head(10) # set to 0


# In[ ]:


df['GarageYrBlt'].value_counts().head(10) # set to 2005


# In[ ]:


'''apply log on to LotFrontage and fillna = median
    fill MasVnrArea NA = 0.0
    fill GarageYrBlt NA = 2005.0'''


# In[ ]:


df['LotFrontage'] = df['LotFrontage'].fillna(df[df['LotFrontage']>0]['LotFrontage'].median())


# In[ ]:


#df['LotFrontage'] = np.log1p(df['LotFrontage'])


# In[ ]:


df['MasVnrArea'] = df['MasVnrArea'].fillna(0.0)


# In[ ]:


df['GarageYrBlt'] = df['GarageYrBlt'].fillna(2005.0)


# In[ ]:


checknan(df[midmd.index].select_dtypes(include=np.number))


# In[ ]:


checknan(df[midmd.index].select_dtypes(include=['object']))


# In[ ]:


lista = list(checknan(df[midmd.index].select_dtypes(include=['object'])).index)


# In[ ]:


print('-' * 60)
for n in lista:
    print('',n,'\n','='*len(n)) 
    print(df[n].value_counts()/df[n].count(),'\n','-'*80)


# In[ ]:


'''
 'GarageFinish' = Unf
 'GarageQual' = TA,
 'GarageCond' = TA,
 'GarageType' = Attchd,
 'BsmtCond' = TA,
 'BsmtExposure' = No,
 'BsmtQual' = TA,
 'BsmtFinType2' = Unf,
 'BsmtFinType1' = Unf,
 'MasVnrType' = None
 '''


# In[ ]:


lista2 = ['Unf','TA','TA','Attchd','TA','No','TA','Unf','Unf','None']
listadict = dict(zip(lista, lista2))
listadict


# In[ ]:


for n in listadict:
    df[n] = df[n].fillna(listadict[n])


# In[ ]:


''' apply log to full df:
    SalePrice,
    GrLivArea, 
    1stFlrSF, 
    GarageArea, 
    TotRmsAbvGrd
'''


# In[ ]:


'''apply log to all numeric features'''
#for featr in ['GrLivArea', 
#              '1stFlrSF', 
#              'GarageArea', 
#              'TotRmsAbvGrd']:
#    df[featr] = np.log1p(df[featr])


# In[ ]:


numerics = list(df.drop(['SalePrice', 'Id'], axis=1).select_dtypes(include=np.number).columns)


# In[ ]:


'''implement squares and cubic features to numerics and and apply log...'''
for featr in numerics:
    df[featr+'2'] =  np.log1p(df[featr] ** 2)
    df[featr+'3'] =  np.log1p(df[featr] ** 3)
    df[featr] = np.log1p(df[featr])


# ### Prepare datasets for modeling

# In[ ]:


all = df
all = pd.get_dummies(all,drop_first=True)


# In[ ]:


X_train = all[:dftrain.shape[0]]
Y_train = X_train[['Id','SalePrice']]
X_test = all[dftrain.shape[0]:]
X_train.drop(['Id','SalePrice'], axis=1, inplace=True)
X_test.drop(['Id','SalePrice'], axis=1, inplace=True)


# In[ ]:


X_train.shape, Y_train.shape, X_test.shape


# In[ ]:





# ### Let the tests begin (xgboost):

# In[ ]:


from sklearn.model_selection import cross_val_score, KFold, learning_curve
import xgboost as xgb
from xgboost import XGBClassifier,XGBRegressor
#from sklearn.grid_search import GridSearchCV 
from sklearn import metrics
seed = 45
n_folds = 5
kfold = KFold(n_folds, shuffle=True, random_state=seed).get_n_splits(X_train)


# In[ ]:


def error(actual, predicted):
    actual = np.log(actual)
    predicted = np.log(predicted)
    return np.sqrt(np.sum(np.square(actual-predicted))/len(actual))

def get_rmse(model):
    rmse = np.sqrt(-cross_val_score(model, 
                                    X_train, 
                                    y=np.log(Y_train.SalePrice), 
                                    scoring="neg_mean_squared_error", 
                                    cv=kfold))
    return rmse.mean()


# In[ ]:


xgbmod = XGBRegressor() 
'''w/o fine tuning'''
xgbmod.fit(X_train, np.log(Y_train.SalePrice))
yhat = np.exp(xgbmod.predict(X_train))
print('rmse:', get_rmse(xgbmod))


# ### review of features to reduce dataset keeping only important ones:

# In[ ]:


import shap
shap_values = shap.TreeExplainer(xgbmod).shap_values(X_train)

global_shap_vals = np.abs(shap_values).mean(0)[:-1]
variables_values = pd.DataFrame(list(zip(X_train.columns,global_shap_vals)))
variables_values.rename(columns={0:'variable',1:'shap_value'},inplace=True)
variables_values.sort_values(by=['shap_value'],ascending=False,inplace=True)
top_n = variables_values.head(12)

pos=range(0,-top_n.shape[0],-1)
plt.barh(pos, top_n['shap_value'], color="#007fff")
plt.yticks(pos, top_n['variable'])
plt.xlabel("mean SHAP value magnitude (do not change in log odds)")
plt.gcf().set_size_inches(8, 4)
plt.gca()
plt.show()


# In[ ]:


shap.summary_plot(shap_values, X_train)


# In[ ]:


'''remove shap_0 features to new prediction'''
top_n = variables_values
remove_featr = list(top_n[top_n.shap_value == 0]['variable'])


# In[ ]:


X_train.drop(remove_featr, axis=1, inplace = True)
X_test.drop(remove_featr, axis=1, inplace = True)


# In[ ]:


xgbmod = XGBRegressor(colsample_bylevel=1, colsample_bytree=1, learning_rate=0.03,max_delta_step=0, 
                      max_depth=6,min_child_weight=6,n_estimators=450,subsample= 0.5)
'''w/ fine tuning'''                      

xgbmod.fit(X_train, np.log(Y_train.SalePrice))
yhat = np.exp(xgbmod.predict(X_train))
print('rmse:', get_rmse(xgbmod))


# ### sumission below gave me top33% position... there is definetely some more data work to be considered..

# ### kaggle submit:

# In[ ]:


#yhat = np.exp(xgbmod.predict(X_test))
#        #yhat = np.exp(yhat)
#Y_test = all[dftrain.shape[0]:]['Id']
#yhat = pd.DataFrame(yhat, columns = ['SalePrice'])
#Y_test = pd.DataFrame (Y_test)
#Y_test['SalePrice'] = yhat.SalePrice
#Y_test.to_csv('subm6.csv', index= False)


# In[ ]:





# ## XGBoost fine tuning space....

# In[ ]:


#'''gridsearch better params'''
#
#xgb_reg = XGBRegressor()
#parameters = {'nthread':[4], #when use hyperthread, xgboost may become slower
#              'colsample_bylevel' : [1],
#              'learning_rate': [0.03,0.04], #so called `eta` value
#              'max_depth': [6],
#              'min_child_weight': [6],
#              'max_delta_step' : [0],
#              'reg_lambda': [1],
#              'subsample': [0.5, 0.6],
#              'colsample_bytree': [ 1],
#              'n_estimators': [450, 550]}
#
#xgb_grid_reg = GridSearchCV(xgb_reg,
#                        parameters,
#                        cv = 2,
#                        n_jobs = 5,
#                        verbose=True)
#
#xgb_grid_reg.fit(X_train,np.log1p(Y_train.SalePrice))
#
#print(xgb_grid_reg.best_score_)
#print(xgb_grid_reg.best_params_)


# In[ ]:


#0.9071702277585866
#{'colsample_bylevel': 1, 'colsample_bytree': 1, 'learning_rate': 0.03, 
# 'max_delta_step': 0, 'max_depth': 6, 'min_child_weight': 7, 'n_estimators': 400, 
# 'nthread': 4, 'reg_lambda': 1, 'subsample': 0.5}
#
#0.9093014037566548
#{'colsample_bylevel': 1, 'colsample_bytree': 1, 'learning_rate': 0.04, 
# 'max_delta_step': 0, 'max_depth': 5, 'min_child_weight': 5, 'n_estimators': 400, 
# 'nthread': 4, 'reg_lambda': 1, 'subsample': 0.6}
#
#0.9075638827570375
#{'colsample_bylevel': 1, 'colsample_bytree': 1, 'learning_rate': 0.03, 
# 'max_delta_step': 0, 'max_depth': 6, 'min_child_weight': 6, 'n_estimators': 500, 
# 'nthread': 4, 'reg_lambda': 1, 'subsample': 0.5}
#
#0.9077751136520033
#{'colsample_bylevel': 1, 'colsample_bytree': 1, 'learning_rate': 0.03, 
# 'max_delta_step': 0, 'max_depth': 6, 'min_child_weight': 6, 'n_estimators': 450, 
# 'nthread': 4, 'reg_lambda': 1, 'subsample': 0.5}

