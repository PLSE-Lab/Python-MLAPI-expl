#!/usr/bin/env python
# coding: utf-8

# # Predict the Housing Price
# ##### By Chintan Chitroda

# In[ ]:


### This Notebook contains many models and predictions.
### The best performing submission was Gradient Boosting Algorithm which was my highest in leaders board.
### Best solution not submitted as i ran out of submission.
### This notebook creates my best sol using Gradient Boosting
### For csv of other models predictions remove # from makecsv method below model block.


# In[ ]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import warnings


# In[ ]:


df_tr = pd.read_csv('/kaggle/input/predict-the-housing-price/train.csv')
df_ts = pd.read_csv('/kaggle/input/predict-the-housing-price/Test.csv')
print("Train:",df_tr.shape)
print("Test:",df_ts.shape)


# #### Data Analysis

# In[ ]:


df_tr.head(5)


# In[ ]:


print('Train Dataset Infomarion')
print ("Rows     : " ,df_tr.shape[0])
print ("Columns  : " ,df_tr.shape[1])
print ("\nFeatures : \n" ,df_tr.columns.tolist())
print ("\nMissing values :  ",df_tr.isnull().sum().values.sum())
print ("\nUnique values :  \n",df_tr.nunique())


# In[ ]:


df_ts.head(5)


# In[ ]:


print('Test Dataset Infomarion')
print ("Rows     : " ,df_ts.shape[0])
print ("Columns  : " ,df_ts.shape[1])
print ("\nFeatures : \n" ,df_ts.columns.tolist())
print ("\nMissing values :  ",df_ts.isnull().sum().values.sum())
print ("\nUnique values :  \n",df_ts.nunique())


# In[ ]:


### Train Null values
sns.heatmap(df_tr.isnull())


# In[ ]:


# Train null list
nullist = []
nullist = df_tr.isnull().sum()
#nullist.loc[nullist != 0]
nul = pd.DataFrame(nullist.loc[nullist != 0])
nul


# In[ ]:


# Numeric Nulls in Train
cols_tr = df_tr.columns
num_cols_tr= df_tr._get_numeric_data().columns
cat_cols_tr = list(set(cols_tr) - set(num_cols_tr))

sns.heatmap(df_tr[num_cols_tr].isnull())


# In[ ]:


## Categorical nulls in Train
sns.heatmap(df_tr[cat_cols_tr].isnull())


# In[ ]:


# Test null list
nullist1 = []
nullist1 = df_ts.isnull().sum()
#nullist.loc[nullist != 0]
nul1 = pd.DataFrame(nullist1.loc[nullist1 != 0])
nul1


# In[ ]:


### Test Null values
sns.heatmap(df_ts.isnull())


# In[ ]:


# Numeric Nulls in Test
cols = df_ts.columns
num_cols = df_ts._get_numeric_data().columns
cat_cols = list(set(cols) - set(num_cols))

sns.heatmap(df_ts[num_cols].isnull())


# In[ ]:


### Categorical null cols in test
sns.heatmap(df_ts[cat_cols].isnull())


# ##### Data Cleaning

# In[ ]:


### Droping cols with too many nulls
drop_columns = ['FireplaceQu','PoolQC','Fence','MiscFeature','BsmtUnfSF']
df_tr.drop(drop_columns, axis = 1, inplace = True)
df_ts.drop(drop_columns, axis = 1, inplace = True)


# In[ ]:


cols = df_tr.columns
num_cols = df_tr._get_numeric_data().columns
cat_cols = list(set(cols) - set(num_cols))


# In[ ]:


fill_col = ['Alley','MasVnrType','BsmtQual','BsmtCond','BsmtExposure','BsmtFinType1','BsmtFinType2',
            'GarageType','GarageFinish','GarageCond']
for i in fill_col:
    print(i,"values :\n",df_tr[i].value_counts())
    print("_____________________")


# In[ ]:


### Categorical data
for i in cat_cols:
    print(i,"values :\n",df_tr[i].value_counts())
    print("_____________________")


# In[ ]:


## Filling No where Nan in Categorical data
for col in df_tr[fill_col]:
    df_tr[col] = df_tr[col].fillna('None')
for col in df_ts[fill_col]:
    df_ts[col] = df_ts[col].fillna('None')


# In[ ]:


colfil = ['BsmtFinSF1','BsmtFinSF2','TotalBsmtSF','BsmtFullBath','BsmtHalfBath','GarageCars', 
            'GarageArea']
for coll in colfil:
    df_ts[coll].fillna(df_ts[coll].median(), inplace = True)


# In[ ]:


num_cols = df_tr._get_numeric_data().columns
cat_cols = list(set(cols) - set(num_cols))


# In[ ]:


df_tr['LotFrontage'].describe()


# In[ ]:


(df_tr['LotFrontage'].plot.box()) 


# In[ ]:


sns.violinplot(df_tr['LotFrontage'])


# In[ ]:


### replace null with median as there are many outliers
df_tr['LotFrontage'].fillna(value=df_tr['LotFrontage'].median(),inplace=True)
df_ts['LotFrontage'].fillna(value=df_ts['LotFrontage'].median(),inplace=True)


# In[ ]:


df_tr.GarageYrBlt.describe()


# In[ ]:


(df_tr['GarageYrBlt'].plot.box()) 


# In[ ]:


sns.violinplot(df_tr['GarageYrBlt'])


# In[ ]:


### replace null with mean as there are many outliers
df_tr['GarageYrBlt'].fillna(value=df_tr['GarageYrBlt'].mean(),inplace=True)
df_ts['GarageYrBlt'].fillna(value=df_ts['GarageYrBlt'].mean(),inplace=True)


# In[ ]:


df_tr['MasVnrArea'].describe()


# In[ ]:


(df_tr['MasVnrArea'].plot.box()) 


# In[ ]:


### replace null with median as there are many outliers
df_tr['MasVnrArea'].fillna(value=df_tr['MasVnrArea'].median(),inplace=True)
df_ts['MasVnrArea'].fillna(value=df_ts['MasVnrArea'].median(),inplace=True)


# In[ ]:


#sns.heatmap(df_tr.isnull())
df_tr.isnull().sum()


# In[ ]:


df_tr.columns


# In[ ]:


### Creating some Featrues 
both_col = [df_tr, df_ts]
for col in both_col:
    col['YrBltAndRemod'] = col['YearBuilt'] + col['YearRemodAdd']
    col['TotalSF'] = col['TotalBsmtSF'] + col['1stFlrSF'] + col['2ndFlrSF']
    col['Total_sqr_footage'] = (col['BsmtFinSF1'] + col['BsmtFinSF2'] +
                                 col['1stFlrSF'] + col['2ndFlrSF'])

    col['Total_Bathrooms'] = (col['FullBath'] + (0.5 * col['HalfBath']) +
                               col['BsmtFullBath'] + (0.5 *col['BsmtHalfBath']))

    col['Total_porch_sf'] = (col['OpenPorchSF'] + col['3SsnPorch'] +
                              col['EnclosedPorch'] + col['ScreenPorch'] +
                              col['WoodDeckSF'])


# In[ ]:


## Binary some feature
both_col = [df_tr, df_ts]
for col in both_col:
    col['haspool'] = col['PoolArea'].apply(lambda x: 1 if x > 0 else 0)
    col['has2ndfloor'] = col['2ndFlrSF'].apply(lambda x: 1 if x > 0 else 0)
    col['hasgarage'] = col['GarageArea'].apply(lambda x: 1 if x > 0 else 0)
    col['hasbsmt'] = col['TotalBsmtSF'].apply(lambda x: 1 if x > 0 else 0)
    col['hasfireplace'] = col['Fireplaces'].apply(lambda x: 1 if x > 0 else 0)


# In[ ]:


plt.subplots(figsize=(30,30))
sns.heatmap(df_tr.corr(),cmap="GnBu",vmax=0.9, square=True)


# In[ ]:


### droping some columns
drop_col = ['Exterior2nd','GarageYrBlt','Condition2','RoofMatl','Electrical','HouseStyle','Exterior1st',
            'Heating','GarageQual','Utilities','MSZoning','Functional','KitchenQual']
df_tr.drop(drop_col, axis = 1,inplace = True)
df_ts.drop(drop_col, axis = 1,inplace = True)


# In[ ]:


df_tr


# In[ ]:


df_ts


# In[ ]:


cols = df_tr.columns
num_cols = df_tr._get_numeric_data().columns
cat_cols = list(set(cols) - set(num_cols))


# In[ ]:


sns.heatmap(df_tr.isnull())


# In[ ]:


sns.heatmap(df_ts.isnull())


# ##### NO More NUll values

# In[ ]:


df_ts[cat_cols]


# In[ ]:


df_tr[cat_cols]


# In[ ]:


### value counts in categorical data in train
for i in df_tr[cat_cols]:
    print(i,":",len(df_tr[i].unique()))


# In[ ]:


### value counts in categorical data in test
for i in df_ts[cat_cols]:
    print(i,":",len(df_ts[i].unique()))


# In[ ]:


### LabelEncoding of categorical data


# In[ ]:


from sklearn.preprocessing import LabelEncoder


# In[ ]:


dftr = df_tr[cat_cols].apply(LabelEncoder().fit_transform)


# In[ ]:


dfts = df_ts[cat_cols].apply(LabelEncoder().fit_transform)


# In[ ]:


df_tr_final = df_tr[num_cols].join(dftr)


# In[ ]:


num_cols = df_ts._get_numeric_data().columns
df_ts_final = df_ts[num_cols].join(dfts)


# In[ ]:


df_tr_final


# In[ ]:


df_ts_final


# In[ ]:





# In[ ]:


ids = df_ts['Id']
df_tr_final.drop('Id',axis=1,inplace=True)
df_ts_final.drop('Id',axis=1,inplace=True)


# ## Model build

# #### Single Linear Regression On all feat

# In[ ]:


from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
import numpy as np
import statsmodels.api as sm


# In[ ]:


### SLR on all columns
for i in df_tr_final.columns:
    X = df_tr_final[[i]]#.values.reshape(1,-1)
    y = df_tr_final[['SalePrice']]#.values.reshape(1,-1)

    X_train,X_test,y_train,y_test = train_test_split(X,y,train_size=.7,random_state=101)
    LR = LinearRegression()
    LR.fit(X_train,y_train)
    y_pred = LR.predict(X_test)
    print(i,"gives R2 score",r2_score(y_pred,y_test))
    print(i,'gives MSE is:',mean_squared_error(y_test, y_pred))
    rms = np.sqrt(mean_squared_error(y_test, y_pred))
    print(i,'gives RMSE is:',rms)
    print("------------------------------------------")
    #print('Coefficient is',LR.coef_[0][0])
    #print('intercept is',LR.intercept_[0])


# ### Multiple LInear Regression Using RFE

# In[ ]:


X = df_tr_final.drop('SalePrice',axis=1)#.values.reshape(1,-1)
y = df_tr_final['SalePrice']#.values.reshape(1,-1)

X_train,X_test,y_train,y_test = train_test_split(X,y,train_size=.7,random_state=101)


# #### Input features 'n' u want to train model with

# In[ ]:


### Using Rfe
from sklearn.feature_selection import RFE
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
#X_train1 = scaler.fit_transform(X_train)
#y_train1 = scaler.fit_transform(y_train)
rfe = RFE(LR, 10)
rfe.fit(X_train,y_train)


# In[ ]:


#rfe.support_


# In[ ]:


X_train.columns[rfe.support_]


# In[ ]:


cols = X_train.columns[rfe.support_]


# In[ ]:


LR.fit(X_train[cols],y_train)


# In[ ]:


y_pred = LR.predict(X_test[cols])
print("gives R2 score",r2_score(y_pred,y_test))
print('gives MSE is:',mean_squared_error(y_test, y_pred))
rms = np.sqrt(mean_squared_error(y_test, y_pred))
print('gives RMSE is:',rms)
print("-----------------------------")


# In[ ]:


y_pred = LR.predict(df_ts_final[cols])


# In[ ]:


#For creating Output CSV file
def makecsv(y_pred,subno): ### input file name in ""
    subdf = pd.DataFrame()
    subdf['Id'] = df_ts['Id']
    subdf['SalePrice'] = y_pred
    subdf.to_csv(subno, index=False)


# ##### Make Csv for result

# In[ ]:


# makecsv(y_pred,"rfesol.csv")


# In[ ]:


import scipy.stats as stats


# In[ ]:


stats.ttest_1samp(a=df_tr['OverallQual'],popmean=df_tr['SalePrice'].mean())


# In[ ]:


model = sm.OLS(y, X)
results = model.fit()
print(results.summary())


# ### Models Using all Features

# In[ ]:


X = df_tr_final.drop('SalePrice',axis=1)#.values.reshape(1,-1)
y = df_tr_final['SalePrice']#.values.reshape(1,-1)
X_train,X_test,y_train,y_test = train_test_split(X,y,train_size=.7,random_state=101)


# In[ ]:


### For using rfe selected features
#X_train = X_train[cols]
#X_test = X_test[cols]


# ## Multiple LInear Regression Algorithm

# In[ ]:


LR.fit(X_train,y_train)


# In[ ]:


### Multiple Linear regression fo all
y_pred = LR.predict(X_test)
print("Multiple Linear regression gives R2 score",r2_score(y_pred,y_test))
print('Multiple Linear regression gives MSE is:',mean_squared_error(y_test, y_pred))
rms = np.sqrt(mean_squared_error(y_test, y_pred))
print('Multiple Linear regression gives RMSE is:',rms)
print("-------------------------------------------")


# In[ ]:


## Testing on Test Dataset
y_pred = LR.predict(df_ts_final)


# #### Make Csv for  result

# In[ ]:


#makecsv(y_pred,"MLsol.csv")


# ### RandomForest  Algorithm

# In[ ]:


from sklearn.ensemble import RandomForestRegressor


# In[ ]:


rf = RandomForestRegressor(n_estimators = 300, random_state = 0)
rf.fit(X_train,y_train)


# In[ ]:


y_pred = rf.predict(X_test)


# In[ ]:


print('all gives R2 score',r2_score(y_pred,y_test))
print('all gives MSE is:',mean_squared_error(y_test, y_pred))
rms = np.sqrt(mean_squared_error(y_test, y_pred))
print('all gives RMSE is:',rms)
print("-----------------------------------------")


# In[ ]:


## Testing on Test Dataset
y_pred = rf.predict(df_ts_final)


# #### Make Csv for reult

# In[ ]:


#makecsv(y_pred,"Rfsol.csv")


# ### XGB Regressor Algorithm

# In[ ]:


import xgboost as xgb


# In[ ]:


model_xgb = xgb.XGBRegressor(colsample_bytree=0.4603, gamma=0.0468, 
                             learning_rate=0.05, max_depth=3, 
                             min_child_weight=1.7817, n_estimators=2200,
                             reg_alpha=0.4640, reg_lambda=0.8571,
                             subsample=0.5213, silent=1,
                             random_state =42, nthread = -1)


# In[ ]:


model_xgb.fit(X_train,y_train)


# In[ ]:


y_pred = model_xgb.predict(X_test)
print('XGB score:',model_xgb.score(X_train,y_train))
print('all gives R2 score',r2_score(y_pred,y_test))
print('all gives MSE is:',mean_squared_error(y_test, y_pred))
rms = np.sqrt(mean_squared_error(y_test, y_pred))
print('all gives RMSE is:',rms)
print("-----------------------------------------")


# In[ ]:


## Testing on Test Dataset
y_pred = model_xgb.predict(df_ts_final)


# ##### Make Csv for result

# In[ ]:


#makecsv(y_pred,"xgbsol.csv")


# ### Gradient Boosting Algorithm

# In[ ]:


from sklearn import ensemble


# In[ ]:


GBoost = ensemble.GradientBoostingRegressor(n_estimators = 3000, max_depth = 5,max_features='sqrt',
                                            min_samples_split = 10,learning_rate = 0.005,loss = 'huber',
                                            min_samples_leaf=15,random_state =10)
GBoost.fit(X_train, y_train)


# In[ ]:


y_pred = GBoost.predict(X_test)
print('GBosst score:',GBoost.score(X_train,y_train))
print('all gives R2 score',r2_score(y_pred,y_test))
print('all gives MSE is:',mean_squared_error(y_test, y_pred))
rms = np.sqrt(mean_squared_error(y_test, y_pred))
print('all gives RMSE is:',rms)
print("-----------------------------------------")


# In[ ]:


#Testing on Test Dataset
y_pred = GBoost.predict(df_ts_final)


# ##### Make Csv for result

# In[ ]:


makecsv(y_pred,"gbsol.csv")


# #### The End
