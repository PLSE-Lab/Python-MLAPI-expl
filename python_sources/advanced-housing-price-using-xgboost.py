#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#Importing the libraries for Data Manipulation and Visualization

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
matplotlib.style.use('fivethirtyeight')


# In[ ]:


#Importing the libraries for data modelling and error metrics
from sklearn import preprocessing
lbl = preprocessing.LabelEncoder()
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import RandomizedSearchCV


# In[ ]:


#Importing the train and test dataset
train=pd.read_csv("../input/house-prices-advanced-regression-techniques/train.csv")
test=pd.read_csv("../input/house-prices-advanced-regression-techniques/test.csv")


# In[ ]:





# In[ ]:


traindata=train.copy()
testdata=test.copy()


# #### Checking the train data

# In[ ]:


train.head()


# In[ ]:


train.shape


# In[ ]:


train.info()


# In[ ]:


features_withna=[features for features in train.columns if train[features].isna().sum()>1]
for features in features_withna:
    print(features,' has ' , train[features].isna().sum(), "  Missing Values out of ", train.shape[0], "\n")


# In[ ]:


#Removing the features with large number of missing values
train.drop(['MiscFeature','Fence','PoolQC','Alley'], axis=1, inplace=True)


# In[ ]:


features_withna.remove('MiscFeature')
features_withna.remove('Fence')
features_withna.remove('PoolQC')
features_withna.remove('Alley')


# In[ ]:


train.columns.str.contains('Yr') |train.columns.str.contains('Year')


# ### Handling Missing values

# In[ ]:


temporal_features=[features for features in train.columns if 'Yr' in features or 'Year' in features]
numeric_features=[features for features in train.columns if train[features].dtypes in ['int64', 'float64'] and features not in temporal_features ]
object_features=[features for features in train.columns if train[features].dtypes=='object']


# In[ ]:


numeric_features


# In[ ]:


object_features


# In[ ]:


temporal_features


# ### Get number of unique values for categorical features

# In[ ]:





# In[ ]:


for features in object_features:
    print(features ,' has ', train[features].nunique(), ' unique values\n')


# ### Replacing the Numeric missing values with Median values of the Column and Object Column missing values with max values 

# In[ ]:


for features in features_withna:
    if features in object_features:
        train[features].fillna(train[features].mode()[0], inplace=True)
    else:
        train[features].fillna(train[features].median(), inplace=True)        


# In[ ]:


for features in features_withna:
    print(features, train[features].isna().sum(), "  Missing Values out of ", train.shape[0], "\n")


# Now all the missing values are handled in train data, let's check out test data

# ## Test Data

# In[ ]:


test.head()


# In[ ]:


test.shape


# Here we have 1 feature less which is the Target variable 'Sale Price'

# In[ ]:


train.info()


# In[ ]:


features_withnatest=[features for features in test.columns if test[features].isna().sum()>1]
for features in features_withnatest:
    print(features, ' has ' , test[features].isna().sum(), "  Missing Values out of ", test.shape[0], "\n")


# In[ ]:


#Removing the features with large number of missing values
test.drop(['MiscFeature','Fence','PoolQC','Alley'], axis=1, inplace=True)


# In[ ]:


features_withnatest


# In[ ]:


features_withnatest.remove('MiscFeature')
features_withnatest.remove('Fence')
features_withnatest.remove('PoolQC')
features_withnatest.remove('Alley')


# ### Handling Missing values in Test Data

# In[ ]:


temporal_features_test=[features for features in test.columns if 'Yr' in features or 'Year' in features]
numeric_features_test=[features for features in test.columns if test[features].dtypes in ['int64', 'float64'] and features not in temporal_features ]
object_features_test=[features for features in test.columns if test[features].dtypes=='object']


# ### Get number of Unique features for Categorical Values

# In[ ]:


for features in object_features_test:
    print(features ,' has ', test[features].nunique(), 'unique values\n')


# ### Replacing the Numeric missing values with Median values of the Column and Object Column missing values with max values 

# In[ ]:


for features in features_withnatest:
    if features in object_features_test:
        test[features].fillna(test[features].mode()[0], inplace=True)
    else:
        test[features].fillna(test[features].median(), inplace=True)    


# In[ ]:


for features in features_withnatest:
    print(features, ' has ' , test[features].isna().sum(), "  Missing Values out of ", test.shape[0], "\n")


# In[ ]:


test['BsmtHalfBath'].fillna(test['BsmtHalfBath'].mode()[0], inplace=True)


# In[ ]:


test['BsmtFullBath'].fillna(test['BsmtFullBath'].mode()[0], inplace=True)


# In[ ]:


test['Functional'].fillna(test['Functional'].mode()[0], inplace=True)


# In[ ]:


test['MSZoning'].fillna(test['MSZoning'].mode()[0], inplace=True)


# In[ ]:


test['Utilities'].fillna(test['Utilities'].mode()[0], inplace=True)


# ## EDA for Training Data

# In[ ]:


train.head()


# ### Checking the various categorical variables

# In[ ]:


## Lets Find the relationship between Categorical Features and Sale PRice

for feature in object_features:
    sns.barplot(x=train[feature],y=train['SalePrice'])
    plt.xlabel(feature)
    plt.ylabel('SalePrice')
    plt.title(feature)
    plt.show()


# In[ ]:


## Lets Find the realtionship between Categorical Features and Sale Price

for feature in numeric_features:
    plt.scatter(x=train[feature],y=train['SalePrice'])
    plt.xlabel(feature)
    plt.ylabel('SalePrice')
    plt.title(feature)
    plt.show()


# In[ ]:


## Lets Find the realtionship between Temporal Features and Sale PRice

for feature in temporal_features:
    sns.lineplot(x=train[feature],y=train['SalePrice'])
    plt.xlabel(feature)
    plt.ylabel('SalePrice')
    plt.title(feature)
    plt.show()


# ### Boxplot for outliers

# In[ ]:


## Lets Find the realtionship between Categorical Features and Sale Price

for feature in numeric_features:
    sns.boxplot(y=train[feature])
    plt.ylabel(feature)
    plt.title(feature)
    plt.show()


# In[ ]:


for feature in numeric_features:
    if 0 in train[feature].unique():
        pass
    else:
        train[feature]=np.log(train[feature])
        train.boxplot(column=feature)
        plt.ylabel(feature)
        plt.title(feature)
        plt.show()


# ### Handling the Temporal Variables

# In[ ]:


temporal_features


# In[ ]:


train['YearBuilt']=train['YrSold']-train['YearBuilt']


# In[ ]:


train['YearRemodAdd']=train['YrSold']-train['YearRemodAdd']


# In[ ]:


train['GarageYrBlt']=train['YrSold']-train['GarageYrBlt']


# In[ ]:


## Here we will compare the difference between All years feature with SalePrice

for feature in temporal_features:
    if feature!='YrSold':
        sns.lineplot(train[feature],train['SalePrice'])
        plt.xlabel(feature)
        plt.ylabel('SalePrice')
        plt.show()


# ## Handling Categorical Variables

# In[ ]:


object_features


# In[ ]:


train.head()


# ## Performing the same Feature Engineering on Test Data

# In[ ]:


for feature in numeric_features_test:
    if 0 in train[feature].unique():
        pass
    else:
        test[feature]=np.log(test[feature])
        test.boxplot(column=feature)
        plt.ylabel(feature)
        plt.title(feature)
        plt.show()


# ### Handling Temporal Variables

# In[ ]:


temporal_features_test


# In[ ]:


test['YearBuilt']=test['YrSold']-test['YearBuilt']


# In[ ]:


test['YearRemodAdd']=test['YrSold']-test['YearRemodAdd']


# In[ ]:


test['GarageYrBlt']=test['YrSold']-test['GarageYrBlt']


# In[ ]:


final_df=pd.concat([train,test],axis=0)


# In[ ]:


final_df.info()


# In[ ]:


final_df.head()


# In[ ]:


object_features=[features for features in final_df.columns if final_df[features].dtypes=='object']


# In[ ]:





# In[ ]:


def one_hot_encoding(obj_features):
    final_dfcopy=final_df.copy()
    for features in obj_features:
        print(features)
        df=pd.get_dummies(final_dfcopy[features],drop_first=True)
        df.head()
        final_dfcopy=pd.concat([df,final_dfcopy],axis=1)
    final_dfcopy.drop(obj_features,axis=1,inplace=True)   
    return final_dfcopy


# In[ ]:


final_df=one_hot_encoding(object_features)


# In[ ]:


final_df =final_df.loc[:,~final_df.columns.duplicated()]


# In[ ]:


final_df.shape


# In[ ]:


final_df.head()


# In[ ]:


final_df['SalePrice']


# In[ ]:


final_df.info()


# ## Model Building and Prediction

# In[ ]:


df_Train=final_df.iloc[:1422,:]
df_Test=final_df.iloc[1422:,:]


# In[ ]:


df_Train.head()


# In[ ]:


df_Train['SalePrice']


# In[ ]:


df_Test.head()


# In[ ]:


X_train=df_Train.drop(['SalePrice'],axis=1)
y_train=df_Train['SalePrice']


# In[ ]:


regressor=XGBRegressor()


# In[ ]:


booster=['gbtree','gblinear']
base_score=[0.25,0.5,0.75,1]


# In[ ]:


## Hyper Parameter Optimization


n_estimators = [100, 500, 900, 1100, 1500]
max_depth = [2, 3, 5, 10, 15]
booster=['gbtree','gblinear']
learning_rate=[0.05,0.1,0.15,0.20]
min_child_weight=[1,2,3,4]

# Define the grid of hyperparameters to search
hyperparameter_grid = {
    'n_estimators': n_estimators,
    'max_depth':max_depth,
    'learning_rate':learning_rate,
    'min_child_weight':min_child_weight,
    'booster':booster,
    'base_score':base_score
    }


# In[ ]:


# Set up the random search with 4-fold cross validation
random_cv = RandomizedSearchCV(estimator=regressor,
            param_distributions=hyperparameter_grid,
            cv=5, n_iter=50,
            scoring = 'neg_mean_absolute_error',n_jobs = 4,
            verbose = 5, 
            return_train_score = True,
            random_state=42)


# In[ ]:


random_cv.fit(X_train,y_train)


# In[ ]:


random_cv.best_estimator_


# In[ ]:


model = XGBRegressor(base_score=0.25, booster='gbtree', colsample_bylevel=1,
             colsample_bynode=1, colsample_bytree=1, gamma=0,
             importance_type='gain', learning_rate=0.05, max_delta_step=0,
             max_depth=2, min_child_weight=4, missing=None, n_estimators=900,
             n_jobs=1, nthread=None, objective='reg:linear', random_state=0,
             reg_alpha=0, reg_lambda=1, scale_pos_weight=1, seed=None,
             silent=None, subsample=1, verbosity=1)


# In[ ]:


regressor.fit(X_train,y_train)


# In[ ]:


df_Test.drop(['SalePrice'],axis=1,inplace=True)


# In[ ]:


df_Test.shape


# In[ ]:


df_Test.head()


# In[ ]:


df_Test.head()


# In[ ]:


y_pred=regressor.predict(df_Test)


# In[ ]:


y_pred


# In[ ]:


#Create Sample Submission file and Submit using ANN
pred=pd.DataFrame(y_pred)
sub_df=pd.read_csv('../input/house-prices-advanced-regression-techniques/sample_submission.csv')
datasets=pd.concat([sub_df['Id'],pred],axis=1)
datasets.columns=['Id','SalePrice']
datasets.to_csv('AdvancedHousingusingXGBoost.csv',index=False)


# In[ ]:


pred


# In[ ]:




