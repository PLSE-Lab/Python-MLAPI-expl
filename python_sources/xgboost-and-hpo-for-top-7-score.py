#!/usr/bin/env python
# coding: utf-8

# **Step 1 is combining test and train data so that EDS and feature engineering can be done together **

# In[ ]:


import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
data= pd.read_csv("../input/trainandtest/trainandtest.csv")
pd.pandas.set_option('display.max_columns',None)
data.shape


# In[ ]:


# calculate the correlation matrix
corr = data.corr()

# display the correlation matrix
display(corr)

# plot the correlation heatmap
sns.heatmap(corr, xticklabels=corr.columns, yticklabels=corr.columns, cmap='RdBu')


# In[ ]:


## First lets handle Categorical features which are missing
features_nan=[feature for feature in data.columns if data[feature].isnull().sum()>1 and data[feature].dtypes=='O']

for feature in features_nan:
    print("{}: {}% missing values".format(feature,np.round(data[feature].isnull().mean(),4)))


# In[ ]:


## taking care of missing values
data['Alley']=data['FireplaceQu'].fillna('MissingAlley')
data['FireplaceQu']=data['FireplaceQu'].fillna('MissingFireplace')
data['Fence']=data['Fence'].fillna('MissingFENCE')
data['PoolQC']=data['PoolQC'].fillna('MissingPOOL')
data['MiscFeature']=data['MiscFeature'].fillna('MissingMISC')


# In[ ]:


## Replace missing value with a new label for train
def replace_cat_feature(data,features_nan):
    dataset=data.copy()
    dataset[features_nan]=dataset[features_nan].fillna('Missing')
    return dataset

data=replace_cat_feature(data,features_nan)

data[features_nan].isnull().sum()


# In[ ]:


## Checking numerical missing values
numerical_with_nan=[feature for feature in data.columns if data[feature].isnull().sum()>1 and data[feature].dtypes!='O']

## We will print the numerical nan variables and percentage of missing values

for feature in numerical_with_nan:
    print("{}: {}% missing value".format(feature,np.around(data[feature].isnull().mean(),4)))


# In[ ]:


## Replacing the numerical Missing Values 

for feature in numerical_with_nan:
    ## We will replace by using median since there are outliers
    median_value=data[feature].median()
    
    ## create a new feature to capture nan values
    #data[feature+'nan']=np.where(data[feature].isnull(),1,0)
    data[feature].fillna(median_value,inplace=True)
    
data[numerical_with_nan].isnull().sum()


# In[ ]:


## (Date Time Variables)

for feature in ['YearBuilt','YearRemodAdd','GarageYrBlt']:
       
    data[feature]=data['YrSold']-data[feature]
    


# In[ ]:


### log transorm for outliers
import numpy as np
num_features=['LotFrontage', 'LotArea', '1stFlrSF', 'GrLivArea']

for feature in num_features:
    data[feature]=np.log(data[feature])


# In[ ]:


## making a new categorical variable for variables occuring rarely
categorical_features=[feature for feature in data.columns if data[feature].dtype=='O']

for feature in categorical_features:
    temp=data.groupby(feature)['SalePrice'].count()/len(data)
    temp_df=temp[temp>0.01].index
    data[feature]=np.where(data[feature].isin(temp_df),data[feature],'Rare_var')


# In[ ]:


#one hot enceoding
for feature in categorical_features:
    dummy = pd.get_dummies(data[feature])
    data=pd.concat([data,dummy],axis=1)
    data=data.drop(feature,axis=1)


# In[ ]:


## feature scaling 
feature_scale=[feature for feature in data if feature not in ['Id','SalePrice']]
from sklearn.preprocessing import MinMaxScaler
scaler=MinMaxScaler()
scaler.fit(data[feature_scale])


# In[ ]:


## removing duplicate columns if any
def remove_dup_columns(data):
     keep_names = set()
     keep_icols = list()
     for icol, name in enumerate(data.columns):
          if name not in keep_names:
               keep_names.add(name)
               keep_icols.append(icol)
     return data.iloc[:, keep_icols]


# In[ ]:


data = remove_dup_columns(data)


# In[ ]:


## set up a customer filter to differentiate  test and train (for test columns, sales is 12345678)
## wanted to do eda and feature engineering on train and test together
filtertest = data.SalePrice== 12345678
filtertrain = data.SalePrice!= 12345678


# In[ ]:


## setting up test and train
test=data[filtertest]
train=data[filtertrain]
x_train = train.drop(['Id','SalePrice'],axis=1)
y_train = train['SalePrice']
x_test=test.drop(['Id','SalePrice'],axis=1)


# In[ ]:


## FEATURE SELECTION


# In[ ]:


from sklearn.linear_model import Lasso
from sklearn.feature_selection import SelectFromModel


# In[ ]:


## finding alpha for lasso
from sklearn.linear_model import LassoCV
from yellowbrick.regressor import AlphaSelection
from yellowbrick.datasets import load_concrete
# Create a list of alphas to cross-validate against
#alphas = np.logspace(-10,1,50,100)
alphas=(1,2,2,3,3,4,4,5,5,6,6,7,7,8,8,9,9,10,10,11,11,12,12,13,13,14,14,15,15,16,16,17,17,18,18,19,19,20,20,21,21,22,22,23,23,24,24,25,25,26,26,27,27,28,28,29,29,30,30,31,31,32,32,33,33,34,34,35,35,36,36,37,37,38,38,39,39,40,40,41,41,42,42,43,43,44,44,45,45,46,46,47,47,48,48,49,49,50,50,51,51,52,52,53,53,54,54,55,55,56,56,57,57,58,58,59,59,60,60,61,61,62,62,63,63,64,64,65,65,66,66,67,67,68,68,69,69,70,70,71,71,72,72,73,73,74,74,75,75,76,76,77,77,78,78,79,79,80,80,81,81,82,82,83,83,84,84,85,85,86,86,87,87,88,88,89,89,90,90,91,91,92,92,93,93,94,94,95,95,96,96,97,97,98,98,99,99,100,100,101,101,102,102,103,103,104,104,105,105,106,106,107,107,108,108,109,109,110,110,111,111,112,112,113,113,114,114,115,115,116,116,117,117,118,118,119)
# Instantiate the linear model and visualizer
model = LassoCV(alphas=alphas,cv=6,max_iter=1500)
visualizer = AlphaSelection(model)
visualizer.fit(x_train,y_train)
visualizer.show()


# In[ ]:


sel_= SelectFromModel(Lasso(alpha=200)) #
sel_.fit(x_train, y_train)


# In[ ]:


# let's print the number of total and selected features

# this is how we can make a list of the selected features
selected_feat = x_train.columns[(sel_.get_support())]

# let's print some stats
print('total features: {}'.format((x_train.shape[1])))
print('selected features: {}'.format(len(selected_feat)))

print('features with coefficients shrank to zero: {}'.format(np.sum(sel_.estimator_.coef_ == 0)))


# In[ ]:


x_train = x_train[selected_feat]
x_test= x_test[selected_feat]


# In[ ]:


###hyper parameter tuning(naivebayes)


# In[ ]:


from sklearn.model_selection import GridSearchCV, RandomizedSearchCV


# In[ ]:


import  xgboost as xgb
import  xgboost as xgb
from hyperopt import hp, tpe, fmin
from sklearn.model_selection import cross_val_score
estimator = xgb.XGBRegressor()


# In[ ]:


parameters = {
    'max_depth': range (2, 10, 1),
    'learning_rate': [0.07,0.1,0.5,1]
}


# In[ ]:


space = {'n_estimators':hp.quniform('n_estimators', 1000, 4000, 100),
         'gamma':hp.uniform('gamma', 0.01, 0.05),
         'learning_rate':hp.uniform('learning_rate', 0.00001, 0.025),
         'max_depth':hp.quniform('max_depth', 3,7,1),
         'subsample':hp.uniform('subsample', 0.60, 0.95),
         'colsample_bytree':hp.uniform('colsample_bytree', 0.60, 0.98),
         'colsample_bylevel':hp.uniform('colsample_bylevel', 0.60, 0.98),
         'reg_lambda': hp.uniform('reg_lambda', 1, 20)
        }


# In[ ]:


def objective(params):
    params= {'n_estimators': int(params['n_estimators']),
             'gamma': params['gamma'],
             'learning_rate': params['learning_rate'],
             'max_depth': int(params['max_depth']),
             'subsample': params['subsample'],
             'colsample_bytree': params['colsample_bytree'],
             'colsample_bylevel': params['colsample_bylevel'],
             'reg_lambda': params['reg_lambda']}
    xb_a= xgb.XGBRegressor(**params)
    score = cross_val_score(xb_a,x_train,y_train,scoring='neg_mean_squared_error', cv=7, n_jobs=-1).mean()
    return -score


# In[ ]:


best = fmin(fn= objective, space= space, max_evals=20, rstate=np.random.RandomState(1), algo=tpe.suggest)


# In[ ]:


print(best)


# In[ ]:


## using parameters selected
xb_b = xgb.XGBRegressor(random_state=0,
                        n_estimators=int(best['n_estimators']), 
                        colsample_bytree= best['colsample_bytree'],
                        gamma= best['gamma'],
                        learning_rate= best['learning_rate'],
                        max_depth= int(best['max_depth']),
                        subsample= best['subsample'],
                        colsample_bylevel= best['colsample_bylevel'],
                        reg_lambda= best['reg_lambda']
                       )

xb_b.fit(x_train, y_train)


# In[ ]:


y_pred= xb_b.predict(x_test)
y_pred= pd.DataFrame(y_pred)
#y_pred_a=y_pred.to_csv('C:\\Users\\Desktop\\Kaggle\\House Pred\\finalsubmission.csv',header= True)


# In[ ]:




