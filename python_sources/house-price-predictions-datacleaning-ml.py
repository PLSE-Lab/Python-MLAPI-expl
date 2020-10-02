#!/usr/bin/env python
# coding: utf-8

# ### So in this what we are going to do is trying out every combination of feature engineering and 
# ### will go for every ML algo which is most suitable and will pick the best.
# ### The dataset is all about everything for a house related things 
# ### Target Variable : SalePrice

# #### Importing all libraries

# In[ ]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from scipy import stats
import statsmodels.api as sm


from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score,mean_squared_error
from sklearn.linear_model import Lasso,LassoCV,Ridge,RidgeCV,ElasticNet,ElasticNetCV
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')


# #### Glimpse of the data - Train and test

# In[ ]:


df = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/train.csv')
dft = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/test.csv')
# pd.set_option('display.max_rows',None)
# pd.set_option('display.max_columns',None)


# In[ ]:


df.head()


# In[ ]:


dft.head()


# In[ ]:


# checking for null Values
df.isnull().sum()


# In[ ]:


full = df.append(dft)


# In[ ]:


full.info()


# In[ ]:


df.shape


# #### As we are seeing below that slightly data is right skewed so we will convert the target varaible into log form.

# In[ ]:


sns.distplot(df['SalePrice'],fit = stats.norm)


# In[ ]:


mu,sigma = stats.norm.fit(df['SalePrice'])


# In[ ]:


# mu, sigma


# In[ ]:


plt.legend(['Normal dist. ($\mu=$ {:.2f} and $\sigma=$ {:.2f} )'.format(mu, sigma)], loc='best')
plt.ylabel('Frequency')

#Probablity plot

fig = plt.figure()
stats.probplot(df['SalePrice'], plot=plt)
plt.show()


# In[ ]:


sm.qqplot(df['SalePrice'],line = 'r')


# In[ ]:


df['SalePrice'] = np.log(df['SalePrice']+1)

#Check again for more normal distribution

plt.subplots(figsize=(12,9))
sns.distplot(df['SalePrice'], fit=stats.norm)

# Get the fitted parameters used by the function
(mu, sigma) = stats.norm.fit(df['SalePrice'])

# plot with the distribution

plt.legend(['Normal dist. ($\mu=$ {:.2f} and $\sigma=$ {:.2f} )'.format(mu, sigma)], loc='best')
plt.ylabel('Frequency')

#Probablity plot

fig = plt.figure()
stats.probplot(df['SalePrice'], plot=plt)
plt.show()


# In[ ]:


df.isnull().sum()


# In[ ]:


# % of missing values
Isnull  = df.isnull().sum()/len(df) * 100
Isnull = Isnull[Isnull>0]
Isnull.sort_values(inplace = True,ascending = False)
Isnull


# In[ ]:


Isnull = Isnull.to_frame()


# In[ ]:


Isnull.columns = ['count']
Isnull.index.names = ['Name']
Isnull['Name'] = Isnull.index


# In[ ]:


Isnull


# In[ ]:


#plot Missing values
plt.figure(figsize=(13, 5))
sns.set(style='whitegrid')
sns.barplot(x='Name', y='count', data=Isnull)
plt.xticks(rotation = 60)
plt.show()


# In[ ]:


df_corr = df.select_dtypes(include=[np.number])


# In[ ]:


df_corr.shape


# In[ ]:


df_corr


# In[ ]:


#Delete Id because that is not need for corralation plot
del df_corr['Id']


# In[ ]:


corr = df_corr.corr()
plt.subplots(figsize=(20,9))
sns.heatmap(corr, annot=True)


# In[ ]:


top_feature = corr.index[abs(corr['SalePrice']>0.5)]
plt.subplots(figsize=(12, 8))
top_corr = df[top_feature].corr()
sns.heatmap(top_corr, annot=True)
plt.show()


# ### Data Imputation : Filling Na

# In[ ]:


num_cols = list(df.select_dtypes(include = np.number))


# In[ ]:


num_colst = list(dft.select_dtypes(include = np.number))


# In[ ]:


dfn = df[num_cols]
dftn = dft[num_colst]


# In[ ]:


# dfn.isnull().sum()


# In[ ]:


# dftn.isnull().sum()


# In[ ]:


dfn['LotFrontage'].fillna(dfn['LotFrontage'].mode()[0],inplace = True)
dftn['LotFrontage'].fillna(dftn['LotFrontage'].mode()[0],inplace = True)


# In[ ]:


dfn['MasVnrArea'].fillna(dfn['MasVnrArea'].mode()[0],inplace = True)
dftn['MasVnrArea'].fillna(dftn['MasVnrArea'].mode()[0],inplace = True)


# In[ ]:


dfn['GarageYrBlt'].unique()


# In[ ]:


dfn['GarageYrBlt'].fillna(dfn['GarageYrBlt'].median(),inplace = True)
dftn['GarageYrBlt'].fillna(dftn['GarageYrBlt'].median(),inplace = True)

dfn.info()


# In[ ]:


dftn.isnull().sum()


# In[ ]:


dftn['BsmtFinSF1'].fillna(method='ffill',inplace = True)
dftn['BsmtFinSF2'].fillna(method='ffill',inplace = True)
dftn['BsmtUnfSF'].fillna(method='ffill',inplace = True)
dftn['TotalBsmtSF'].fillna(method='ffill',inplace = True)
dftn['BsmtFullBath'].fillna(method='ffill',inplace = True)
dftn['BsmtHalfBath'].fillna(method='ffill',inplace = True)
dftn['GarageCars'].fillna(method='ffill',inplace = True)
dftn['GarageArea'].fillna(method='ffill',inplace = True)


# In[ ]:


dftn.isnull().sum()


# In[ ]:


dfn['MasVnrArea'].unique()


# In[ ]:


dfn['MasVnrArea'] = dfn['MasVnrArea'].astype(float)
dfn.isnull().sum()


# In[ ]:


dfn.drop('Id',axis = 1,inplace = True)
dftn.drop('Id',axis = 1,inplace = True)


# In[ ]:


for col in dfn.columns:
    print(' ')
    print(col)
    print(dfn[col].value_counts())


# In[ ]:


co = list(dfn['OverallQual'].value_counts().head(5).index)
dfn['OverallQual'] = np.where(dfn['OverallQual'].isin(co),dfn['OverallQual'],'Other')
co = list(dftn['OverallQual'].value_counts().head(5).index)
dftn['OverallQual'] = np.where(dftn['OverallQual'].isin(co),dftn['OverallQual'],'Other')


# In[ ]:


co = list(dfn['OverallCond'].value_counts().head(3).index)
dfn['OverallCond'] = np.where(dfn['OverallCond'].isin(co),dfn['OverallCond'],'Other')

co = list(dftn['OverallCond'].value_counts().head(3).index)
dftn['OverallCond'] = np.where(dftn['OverallCond'].isin(co),dftn['OverallCond'],'Other')


# In[ ]:


cd = list(dfn['YearBuilt'].value_counts().head(10).index)
dfn['YearBuilt'] = np.where(dfn['YearBuilt'].isin(cd),dfn['YearBuilt'],'Other')

cd = list(dftn['YearBuilt'].value_counts().head(10).index)
dftn['YearBuilt'] = np.where(dftn['YearBuilt'].isin(cd),dftn['YearBuilt'],'Other')


# In[ ]:


cf = list(dfn['YearRemodAdd'].value_counts().head(10).index)
dfn['YearRemodAdd'] = np.where(dfn['YearRemodAdd'].isin(cf),dfn['YearRemodAdd'],'Other')
cf = list(dftn['YearRemodAdd'].value_counts().head(10).index)
dftn['YearRemodAdd'] = np.where(dftn['YearRemodAdd'].isin(cf),dftn['YearRemodAdd'],'Other')


# In[ ]:


ce = list(dfn['GarageYrBlt'].value_counts().head(10).index)
dfn['GarageYrBlt'] = np.where(dfn['GarageYrBlt'].isin(ce),dfn['GarageYrBlt'],'Other')

ce = list(dftn['GarageYrBlt'].value_counts().head(10).index)
dftn['GarageYrBlt'] = np.where(dftn['GarageYrBlt'].isin(ce),dftn['GarageYrBlt'],'Other')


# In[ ]:


cg = list(dfn['GarageCars'].value_counts().head(3).index)
dfn['GarageCars'] = np.where(dfn['GarageCars'].isin(cg),dfn['GarageCars'],'Other')

cg = list(dftn['GarageCars'].value_counts().head(3).index)
dftn['GarageCars'] = np.where(dftn['GarageCars'].isin(cg),dftn['GarageCars'],'Other')


# In[ ]:


cat_cols = ['OverallQual','OverallCond','YearBuilt','GarageYrBlt','GarageCars','YearRemodAdd']


# In[ ]:


dfn=pd.get_dummies(dfn,columns=cat_cols,drop_first=True)
dftn=pd.get_dummies(dftn,columns=cat_cols,drop_first=True)


# In[ ]:


print(dfn.shape)
print(dftn.shape)


# ### Modelling

# In[ ]:


X = dfn.drop('SalePrice',axis = 1)
y = dfn['SalePrice']


# ### Random Forest

# In[ ]:


from sklearn.ensemble import RandomForestRegressor


# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3,random_state = 42)


# In[ ]:


rf = RandomForestRegressor()
rf = rf.fit(X_train, y_train)
y_pred =  rf.predict(X_test)
rf.score(X_test,y_test)


# In[ ]:


r2_score(y_test,y_pred)


# In[ ]:


print('RMSe : ',np.sqrt(mean_squared_error(y_test,y_pred)))


# In[ ]:


dfn.head()


# In[ ]:


dftn.head()


# In[ ]:





# In[ ]:


def rmsle(real, predicted):
    sum=0.0
    for x in range(len(predicted)):
        if predicted[x]<0 or real[x]<0: #check for negative values
            continue
        p = np.log(predicted[x]+1)
        r = np.log(real[x]+1)
        sum = sum + (p - r)**2
    return (sum/len(predicted))**0.5


#  ### XGBOOSt 

# In[ ]:


X = dfn.drop('SalePrice',axis = 1)
y = dfn['SalePrice']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3,random_state = 42)


# In[ ]:


from xgboost import XGBRegressor


# In[ ]:


xgb = XGBRegressor()
xgb = xgb.fit(X_train,y_train)
y_pred = xgb.predict(X_test)


# In[ ]:


xgb.score(X_test,y_test)


# In[ ]:


r2_score(y_test,y_pred)


# In[ ]:


print('RMSe : ',np.sqrt(mean_squared_error(y_test,y_pred)))


# In[ ]:


# rmsle(y_test,y_pred)


# In[ ]:





# In[ ]:


X = dfn.drop('SalePrice',axis = 1)
y = dfn['SalePrice']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3,random_state = 42)


# In[ ]:


from sklearn.metrics import mean_squared_error
def adjusted_r2_score(y_true, y_pred, X_test):
    r2 = r2_score(y_true=y_true, y_pred=y_pred)
    adjusted_r2 = 1 - (1-r2)*(len(y_true)-1)/(len(y_true) - X_test.shape[1]-1)
    return adjusted_r2


# In[ ]:


xgr = XGBRegressor(objective='reg:linear', n_estimators=50, max_depth=5, n_jobs=-1, random_state=42)
xgr.fit(X_train, y_train)

y_pred = xgr.predict(X_test)

rsq_baseline2_xgb = r2_score(y_true=y_test, y_pred=y_pred)
adj_rsq_baseline2_xgb = adjusted_r2_score(y_true=y_test, y_pred=y_pred, X_test=X_test)
rmse_baseline2_xgb = mean_squared_error(y_true=y_test, y_pred=y_pred) ** 0.5
print('R-sq:', rsq_baseline2_xgb)
print('Adj. R-sq:', adj_rsq_baseline2_xgb)
print('RMSE:', rmse_baseline2_xgb)


# In[ ]:





# In[ ]:





# In[ ]:





# ### Decision Tree

# In[ ]:


from sklearn.tree import DecisionTreeRegressor


# In[ ]:


X = dfn.drop('SalePrice',axis = 1)
y = dfn['SalePrice']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3,random_state = 42)


# In[ ]:


dt = DecisionTreeRegressor()
dt = dt.fit(X_train,y_train)
y_pred = dt.predict(X_test)


# In[ ]:



print('r2_score: ',r2_score(y_test,y_pred))
print('Accuracy: ',dt.score(X_test,y_test))
print('RMSE: ',np.sqrt(mean_squared_error(y_test,y_pred)))
print('adj r sq.: ',1 - (1-r2_score(y_test,y_pred))*(len(y_test)-1)/(len(y_test) - X_test.shape[1]-1) )


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# ### Grid Search CV

# In[ ]:


from sklearn.model_selection import GridSearchCV


# In[ ]:


# # rf = RandomForestRegressor()

# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

# rfc = RandomForestRegressor(n_jobs=-1 , oob_score = True,random_state = 42) 

# param_grid = {
#     'n_estimators': [50,100,200, 700],
#     'max_features': ['auto', 'sqrt', 'log2'],
#     }

# CV_rfc = GridSearchCV(estimator=rfc, param_grid=param_grid, cv= 5)
# CV_rfc.fit(X_train,y_train)
# print('\n',CV_rfc.best_estimator_)


# In[ ]:


rf = RandomForestRegressor(bootstrap=True, criterion='mse', max_depth=None,
                      max_features='sqrt', max_leaf_nodes=None,
                      min_impurity_decrease=0.0, min_impurity_split=None,
                      min_samples_leaf=1, min_samples_split=2,
                      min_weight_fraction_leaf=0.0, n_estimators=700, n_jobs=-1,
                      oob_score=True, random_state=42, verbose=0,
                      warm_start=False)


X = dfn.drop('SalePrice',axis = 1)
y = dfn['SalePrice']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3,random_state = 42)

rf.fit(X_train,y_train)
y_pred = rf.predict(X_test)
# y_pred = np.exp(y_pred-1)
print('r2_score: ',r2_score(y_test,y_pred))
print('Accuracy: ',dt.score(X_test,y_test))
print('RMSE: ',np.sqrt(mean_squared_error(y_test,y_pred)))
print('adj r sq.: ',1 - (1-r2_score(y_test,y_pred))*(len(y_test)-1)/(len(y_test) - X_test.shape[1]-1) )


# In[ ]:


X.head()


# In[ ]:


dftn.head()


# In[ ]:


pred = rf.predict(dftn)


# In[ ]:


predantilog = np.exp(pred)-1


# In[ ]:


predantilog


# In[ ]:


import xgboost as xgb


# In[ ]:


data_dmatrix = xgb.DMatrix(data=X,label=y)


# In[ ]:


# X_train.head()


# In[ ]:


params = {"objective":"reg:linear",'colsample_bytree': 0.3,'learning_rate': 0.1,
                'max_depth': 20, 'alpha': 10}

cv_results = xgb.cv(dtrain=data_dmatrix, params=params, nfold=3,
                    num_boost_round=50,early_stopping_rounds=10,metrics="rmse", as_pandas=True, seed=123)


# In[ ]:


cv_results.head()


# In[ ]:


print((cv_results["test-rmse-mean"]).tail(1))


# In[ ]:





# In[ ]:


# xg = XGBRegressor(n_jobs = -1)
# params = {
#         'max_depth' : [10,20],
#         'learning_rate' : [0.1,0.2],
#         'n_estimators' : [100,200],
#         "subsample" : [0.5, 0.8]
        
#         }

# grid = GridSearchCV(estimator = xg,param_grid=params,cv = 5,n_jobs = -1)
# grid.fit(X_train,y_train)
# grid.best_params_


# In[ ]:


xg  = XGBRegressor(max_depth = 20,subsample=0.8).fit(X_train,y_train)
predic = xg.predict(X_test)
print('r2_score: ',r2_score(y_test,predic))
print('Accuracy: ',dt.score(X_test,predic))
print('RMSE: ',np.sqrt(mean_squared_error(y_test,predic)))
print('adj r sq.: ',1 - (1-r2_score(y_test,predic))*(len(y_test)-1)/(len(y_test) - X_test.shape[1]-1) )


# In[ ]:


xg.feature_importances_


# In[ ]:


feat_importances = pd.Series(xg.feature_importances_, index=X.columns)
feat_importances.nlargest(10).plot(kind='barh')


# In[ ]:





# In[ ]:





# In[ ]:


X.head()


# In[ ]:


X.shape


# In[ ]:


from sklearn.preprocessing import StandardScaler


# In[ ]:


X_std = pd.DataFrame(StandardScaler().fit_transform(X),columns = X.columns)


# In[ ]:


cov_matrix = np.cov(X_std.T)
print(']n Covariance Matrix \n%s',cov_matrix)


# In[ ]:





# In[ ]:


eig_vals, eig_vecs = np.linalg.eig(cov_matrix)
print('Eigen Values \n%s',eig_vals)
print('Eigen Vectors \n%s',eig_vecs)


# In[ ]:


eigen_pairs = [(np.abs(eig_vals[i]),eig_vecs[:,i]) for i in range(len(eig_vals))]
tot  = sum(eig_vals)
var_exp = [(i/tot)*100 for i in sorted(eig_vals,reverse = True)]
cum_var_exp = np.cumsum(var_exp)
print('Cumulative VAriance Explained',cum_var_exp)


# In[ ]:


from sklearn.preprocessing import StandardScaler
scX = StandardScaler() 
X_train = scX.fit_transform(X_train) 
X_test = scX.fit_transform(X_test)


# In[ ]:


from sklearn.decomposition import PCA
pca = PCA(n_components = None) 
X_train = pca.fit_transform(X_train)
X_test = pca.transform(X_test)
explainedvariance = pca.explained_variance_ratio_


# In[ ]:


# rf = RandomForestRegressor(bootstrap=True, criterion='mse', max_depth=None,
#                       max_features='sqrt', max_leaf_nodes=None,
#                       min_impurity_decrease=0.0, min_impurity_split=None,
#                       min_samples_leaf=1, min_samples_split=2,
#                       min_weight_fraction_leaf=0.0, n_estimators=700, n_jobs=-1,
#                       oob_score=True, random_state=42, verbose=0,
#                       warm_start=False)
from sklearn.ensemble import AdaBoostRegressor
import xgboost as xgb
rf = xgb.XGBRegressor(max_depth=3)
predicts = rf.fit(X_train,y_train).predict(X_test)
np.sqrt(mean_squared_error(y_test,predicts))


# In[ ]:


# dftn = scX.fit_transform(dftn) 
# dftn = pca.fit_transform(dftn)


# In[ ]:


# predictrf = rf.predict(dftn)


# In[ ]:


# predictrfanti = np.exp(predictrf)-1


# In[ ]:


xg  = XGBRegressor(max_depth = 20,subsample=0.8).fit(X_train,y_train)
predic = xg.predict(X_test)
print('r2_score: ',r2_score(y_test,predic))
print('Accuracy: ',dt.score(X_test,predic))
print('RMSE: ',np.sqrt(mean_squared_error(y_test,predic)))
print('adj r sq.: ',1 - (1-r2_score(y_test,predic))*(len(y_test)-1)/(len(y_test) - X_test.shape[1]-1) )


# In[ ]:





# ### The best results were given by Random Forest with hyperparameters tuning and rmse of 0.17

# In[ ]:


t = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/test.csv')
submission_predicted = pd.DataFrame({'Id' : t['Id'],'SalePrice':predantilog })
submission_predicted.head()


# In[ ]:


submission_predicted.to_csv('submission.csv',index = False)


# In[ ]:




