#!/usr/bin/env python
# coding: utf-8

# # HOUSE PRICES PREDICTION

# # Importing libraries

# In[ ]:


import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


# Load the train data in a dataframe
train = pd.read_csv("../input/train.csv")

# Load the test data in a dataframe
test = pd.read_csv("../input/test.csv")


# In[ ]:


train.info()


# In[ ]:


train.head()


# # MISSING VALUES IMPUTATION

# In[ ]:


nulls = train.isnull().sum().sort_values(ascending=False)
nulls.head(20)


# From the above dataframe -'nulls' we came to know that the attributes PoolQC,MiscFeature,Alley and Fence are having morethan 60% of the values as 'nan'.so, its better to remove them as these columns won't give much info about the SalePrice.

# In[ ]:


train = train.drop(['Id','PoolQC','MiscFeature','Alley','Fence'],axis = 1)


# ## FireplaceQu

# In[ ]:


train[['Fireplaces','FireplaceQu']].head(10)


# In[ ]:


train['FireplaceQu'].isnull().sum()


# In[ ]:


train['Fireplaces'].value_counts()


# The attribute 'FireplaceQu' is having 690 null values.If we compare the columns 'FireplaceQu' and 'Fireplaces' the indexes which are having the zeros in the Fireplaces column are having the 'nan' values in FireplaceQu. It tells that the houses which are not having the Fireplaces are having nan values in FireplaceQu so, i will replace these nulls with "no Fireplace" i,e 'NF'  

# In[ ]:


train['FireplaceQu']=train['FireplaceQu'].fillna('NF')


# ## LotFrontage

# In[ ]:


train['LotFrontage'] =train['LotFrontage'].fillna(value=train['LotFrontage'].mean())


# ## Attributes related to "GARAGE"

# In[ ]:


train['GarageType'].isnull().sum()


# In[ ]:


train['GarageCond'].isnull().sum()


# In[ ]:


train['GarageFinish'].isnull().sum()


# In[ ]:


train['GarageYrBlt'].isnull().sum()


# In[ ]:


train['GarageQual'].isnull().sum()


# In[ ]:


train['GarageArea'].value_counts().head()


# We can observe that all the columns related to Garage are having the sama number of null values. so, there should be a relationship among them and if we look at the 'GarageArea' column it is having the 81 zeros which is equal to no: of 'nans' in these columns.Hence we can conclude that the houses without Garage Area are having 'nan' at all these columns.
# 
# >> I will replace these nans with 'No GarageArea'----> 'NG' 

# In[ ]:


train['GarageType']=train['GarageType'].fillna('NG')
train['GarageCond']=train['GarageCond'].fillna('NG')
train['GarageFinish']=train['GarageFinish'].fillna('NG')
train['GarageYrBlt']=train['GarageYrBlt'].fillna('NG')
train['GarageQual']=train['GarageQual'].fillna('NG')


# ## Bsmt

# In[ ]:


train.BsmtExposure.isnull().sum()


# In[ ]:


train.BsmtFinType2.isnull().sum()


# In[ ]:


train.BsmtFinType1.isnull().sum()


# In[ ]:


train.BsmtCond.isnull().sum() 


# In[ ]:


train.BsmtQual.isnull().sum()


# In[ ]:


train.TotalBsmtSF.value_counts().head()


# In[ ]:


train['BsmtExposure']=train['BsmtExposure'].fillna('NB')
train['BsmtFinType2']=train['BsmtFinType2'].fillna('NB')
train['BsmtFinType1']=train['BsmtFinType1'].fillna('NB')
train['BsmtCond']=train['BsmtCond'].fillna('NB')
train['BsmtQual']=train['BsmtQual'].fillna('NB')


# ## MasVnr

# In[ ]:


train['MasVnrArea'] = train['MasVnrArea'].fillna(train['MasVnrArea'].mean())


# In[ ]:


train['MasVnrType'] = train['MasVnrType'].fillna('none')


# ## Electrical

# In[ ]:


train.Electrical = train.Electrical.fillna('SBrkr')


# In[ ]:


train.isnull().sum().sum()


# # OUTLIERS

# In[ ]:


num_train = train._get_numeric_data()


# In[ ]:


num_train.columns


# In[ ]:


def var_summary(x):
    return pd.Series([x.count(), x.isnull().sum(), x.sum(), x.mean(), x.median(),  x.std(), x.var(), x.min(), x.quantile(0.01), x.quantile(0.05),x.quantile(0.10),x.quantile(0.25),x.quantile(0.50),x.quantile(0.75), x.quantile(0.90),x.quantile(0.95), x.quantile(0.99),x.max()], 
                  index=['N', 'NMISS', 'SUM', 'MEAN','MEDIAN', 'STD', 'VAR', 'MIN', 'P1' , 'P5' ,'P10' ,'P25' ,'P50' ,'P75' ,'P90' ,'P95' ,'P99' ,'MAX'])

num_train.apply(lambda x: var_summary(x)).T


# In[ ]:


sns.boxplot([num_train.LotFrontage])


# In[ ]:


train['LotFrontage']= train['LotFrontage'].clip_upper(train['LotFrontage'].quantile(0.99)) 


# In[ ]:


sns.boxplot(num_train.LotArea)


# In[ ]:


train['LotArea']= train['LotArea'].clip_upper(train['LotArea'].quantile(0.99)) 


# In[ ]:


sns.boxplot(train['MasVnrArea'])


# In[ ]:


train['MasVnrArea']= train['MasVnrArea'].clip_upper(train['MasVnrArea'].quantile(0.99))


# In[ ]:


sns.boxplot(train['BsmtFinSF1']) 


# In[ ]:


sns.boxplot(train['BsmtFinSF2']) 


# In[ ]:


train['BsmtFinSF1']= train['BsmtFinSF1'].clip_upper(train['BsmtFinSF1'].quantile(0.99)) 
train['BsmtFinSF2']= train['BsmtFinSF2'].clip_upper(train['BsmtFinSF2'].quantile(0.99)) 


# In[ ]:


sns.boxplot(train['TotalBsmtSF'])


# In[ ]:


train['TotalBsmtSF']= train['TotalBsmtSF'].clip_upper(train['TotalBsmtSF'].quantile(0.99))


# In[ ]:


sns.boxplot(train['1stFlrSF'])


# In[ ]:


train['1stFlrSF']= train['1stFlrSF'].clip_upper(train['1stFlrSF'].quantile(0.99))


# In[ ]:


sns.boxplot(train['2ndFlrSF'])


# In[ ]:


train['2ndFlrSF']= train['2ndFlrSF'].clip_upper(train['2ndFlrSF'].quantile(0.99))


# In[ ]:


sns.boxplot(train['GrLivArea'])


# In[ ]:


train['GrLivArea']= train['GrLivArea'].clip_upper(train['GrLivArea'].quantile(0.99))


# In[ ]:


sns.boxplot(train['BedroomAbvGr'])


# In[ ]:


train['BedroomAbvGr']= train['BedroomAbvGr'].clip_upper(train['BedroomAbvGr'].quantile(0.99))
train['BedroomAbvGr']= train['BedroomAbvGr'].clip_lower(train['BedroomAbvGr'].quantile(0.01))


# In[ ]:


sns.boxplot(train['GarageCars'])


# In[ ]:


train['GarageCars']= train['GarageCars'].clip_upper(train['GarageCars'].quantile(0.99))


# In[ ]:


sns.boxplot(train['GarageArea'])


# In[ ]:


train['GarageArea']= train['GarageArea'].clip_upper(train['GarageArea'].quantile(0.99))


# In[ ]:


sns.boxplot(train['WoodDeckSF'])


# In[ ]:


train['WoodDeckSF']= train['WoodDeckSF'].clip_upper(train['WoodDeckSF'].quantile(0.99))


# In[ ]:


sns.boxplot(train['OpenPorchSF'])


# In[ ]:


train['OpenPorchSF']= train['OpenPorchSF'].clip_upper(train['OpenPorchSF'].quantile(0.99))


# In[ ]:


sns.boxplot(train['EnclosedPorch'])


# In[ ]:


train['EnclosedPorch']= train['EnclosedPorch'].clip_upper(train['EnclosedPorch'].quantile(0.99))


# In[ ]:


sns.boxplot(train['3SsnPorch'])


# In[ ]:


train['3SsnPorch']= train['3SsnPorch'].clip_upper(train['3SsnPorch'].quantile(0.99))


# In[ ]:


sns.boxplot(train['ScreenPorch'])


# In[ ]:


train['ScreenPorch']= train['ScreenPorch'].clip_upper(train['ScreenPorch'].quantile(0.99))


# In[ ]:


sns.boxplot(train['PoolArea'])


# In[ ]:


train['PoolArea']= train['PoolArea'].clip_upper(train['PoolArea'].quantile(0.99))


# In[ ]:


sns.boxplot(train['MiscVal'])


# In[ ]:


sns.boxplot(train.SalePrice)


# In[ ]:


train['SalePrice']= train['SalePrice'].clip_upper(train['SalePrice'].quantile(0.99))
train['SalePrice']= train['SalePrice'].clip_lower(train['SalePrice'].quantile(0.01))


# In[ ]:


train['MiscVal']= train['MiscVal'].clip_upper(train['MiscVal'].quantile(0.99))


# In[ ]:


num_corr=num_train .corr()
plt.subplots(figsize=(13,10))
sns.heatmap(num_corr,vmax =.8 ,square = True)


# In[ ]:


k = 14
cols = num_corr.nlargest(k, 'SalePrice')['SalePrice'].index
cm = np.corrcoef(num_train[cols].values.T)
sns.set(font_scale=1.35)
f, ax = plt.subplots(figsize=(10,10))
hm=sns.heatmap(cm, annot = True,vmax =.8, yticklabels=cols.values, xticklabels = cols.values)


# # FEATURE SELECTION

# ## Extracting new Features using PCA

# In[ ]:


from sklearn.preprocessing import StandardScaler


# In[ ]:


train_d = pd.get_dummies(train)
train_d1 = train_d.drop(['SalePrice'],axis = 1)
y = train_d.SalePrice


#  Before applying PCA we have to convert all the data into a single scale.I used Standard Scalar method to scale the data.

# In[ ]:


scaler = StandardScaler()
scaler.fit(train_d1)                
t_train = scaler.transform(train_d1)


# In[ ]:


from sklearn.decomposition import PCA


# In[ ]:


pca_hp = PCA(30)
x_fit = pca_hp.fit_transform(t_train)


# In[ ]:


np.exp(pca_hp.explained_variance_ratio_)


# ## LINEAR REGRESSION

# In[ ]:


from sklearn.model_selection import train_test_split


# In[ ]:


from sklearn.linear_model import LinearRegression
linear = LinearRegression()
linear.fit(x_fit,y)


# In[ ]:


X_train , X_test, Y_train, Y_test = train_test_split(
        x_fit,
        y,
        test_size=0.20,
        random_state=123)


# In[ ]:


y_pred = linear.predict(X_test)


# In[ ]:


from sklearn import metrics


# In[ ]:


metrics.r2_score(Y_test, y_pred)


# In[ ]:


rmse = np.sqrt(metrics.mean_squared_error(Y_test, y_pred))
rmse


# ## DECISION TREES

# In[ ]:


from sklearn import metrics
from sklearn.tree import DecisionTreeRegressor, export_graphviz, export 
from sklearn.grid_search import GridSearchCV


# >>I have used GridSearch cv tofind the hyper parameters, such as n_estimators and depth .

# In[ ]:


depth_list = list(range(1,20))
for depth in depth_list:
    dt_obj = DecisionTreeRegressor(max_depth=depth)
    dt_obj.fit(X_train, Y_train)
    print ('depth:', depth, 'R_squared:', metrics.r2_score(Y_test, dt_obj.predict(X_test)))


# In[ ]:


param_grid = {'max_depth': np.arange(3,20)}
tree = GridSearchCV(DecisionTreeRegressor(), param_grid, cv=10)
tree.fit(X_train, Y_train)


# In[ ]:


tree.best_params_


# In[ ]:


tree.best_score_


# In[ ]:


tree_final = DecisionTreeRegressor(max_depth=5)
tree_final.fit(X_train, Y_train)


# In[ ]:


tree_test_pred = pd.DataFrame({'actual': Y_test, 'predicted': tree_final.predict(X_test)})


# In[ ]:


metrics.r2_score(Y_test, tree_test_pred.predicted)


# In[ ]:


rmse = np.sqrt(metrics.mean_squared_error(Y_test, tree_test_pred.predicted))
rmse


# ## RANDOM FORESTS

# In[ ]:


from sklearn.ensemble import RandomForestRegressor


# In[ ]:


depth_list = list(range(1,20))
for depth in depth_list:
    dt_obj = RandomForestRegressor(max_depth=depth)
    dt_obj.fit(X_train, Y_train)
    print ('depth:', depth, 'R_Squared:', metrics.r2_score(Y_test, dt_obj.predict(X_test)))


# In[ ]:


radm_clf = RandomForestRegressor(oob_score=True,n_estimators=100,max_depth = 11)
radm_clf.fit( X_train, Y_train )


# In[ ]:


radm_test_pred = pd.DataFrame( { 'actual':  Y_test,
                            'predicted': radm_clf.predict( X_test ) } )


# In[ ]:


metrics.r2_score( radm_test_pred.actual, radm_test_pred.predicted )


# In[ ]:


rmse = np.sqrt(metrics.mean_squared_error(Y_test, tree_test_pred.predicted))
rmse


# ## BAGGED DECISION TREES

# In[ ]:


from sklearn.ensemble import BaggingRegressor


# In[ ]:


param_bag = {'n_estimators': list(range(100, 801, 100)),
             }


# In[ ]:


from sklearn.grid_search import GridSearchCV
bag_cl = GridSearchCV(estimator=BaggingRegressor(),
                  param_grid=param_bag,
                  cv=5,
                  verbose=True, n_jobs=-1)


# In[ ]:


bag_cl.get_params()


# In[ ]:


bag_cl.fit(X_train, Y_train)


# In[ ]:


bag_cl.best_params_


# In[ ]:


bagclm = BaggingRegressor(oob_score=True, n_estimators=700)
bagclm.fit(X_train, Y_train)


# In[ ]:


y_pred = pd.DataFrame( { 'actual':  Y_test,
                            'predicted': bagclm.predict( X_test) } )


# In[ ]:


metrics.r2_score(y_pred.actual, y_pred.predicted)


# In[ ]:


rmse = np.sqrt(metrics.mean_squared_error(Y_test, y_pred.predicted))
rmse


# # BOOSTING TECHNIQUES

# ## ADABOOST

# In[ ]:


from sklearn.ensemble import AdaBoostRegressor, GradientBoostingRegressor


# In[ ]:


[10**x for x in range(-3, 3)]


# In[ ]:


paragrid_ada = {'n_estimators': [100, 200, 400, 600, 800],
               'learning_rate': [10**x for x in range(-3, 3)]}


# In[ ]:


from sklearn.grid_search import GridSearchCV
ada = GridSearchCV(estimator=AdaBoostRegressor(),
                  param_grid=paragrid_ada,
                  cv=5,
                  verbose=True, n_jobs=-1)


# In[ ]:


ada.fit(X_train, Y_train)


# In[ ]:


ada.best_params_


# In[ ]:


ada_clf = AdaBoostRegressor(learning_rate=0.1, n_estimators=600)


# In[ ]:


ada_clf.fit(X_train, Y_train)


# In[ ]:


ada_test_pred = pd.DataFrame({'actual': Y_test,
                            'predicted': ada_clf.predict(X_test)})


# In[ ]:


metrics.r2_score(ada_test_pred.actual, ada_test_pred.predicted)


# In[ ]:


rmse = np.sqrt(metrics.mean_squared_error(Y_test, y_pred.predicted))
rmse


# ## GRADIENT BOOSTING

# In[ ]:


param_test1 = {'n_estimators': [100, 200, 400, 600, 800],
              'max_depth': list(range(1,10))}
gsearch1 = GridSearchCV(estimator = GradientBoostingRegressor(learning_rate=0.1, min_samples_split=500, min_samples_leaf=50,
                                                               max_features='sqrt',subsample=0.8,verbose = 0), 
                        param_grid = param_test1, scoring='r2',n_jobs=4,iid=False, cv=5)


# In[ ]:


gsearch1.fit(X_train, Y_train)


# In[ ]:


gsearch1.best_params_


# In[ ]:


gbm = GradientBoostingRegressor(learning_rate=0.1, min_samples_split=500, min_samples_leaf=50,max_depth=5, n_estimators=400,
                                                               max_features='sqrt',subsample=0.8, random_state=10)


# In[ ]:


gbm.fit(X_train, Y_train)


# In[ ]:


gbm_test_pred = pd.DataFrame({'actual': Y_test,
                            'predicted': gbm.predict(X_test)})


# In[ ]:


metrics.r2_score(gbm_test_pred.actual, gbm_test_pred.predicted)


# In[ ]:


rmse = np.sqrt(metrics.mean_squared_error(gbm_test_pred.actual, gbm_test_pred.predicted))
rmse

