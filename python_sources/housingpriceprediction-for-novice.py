#!/usr/bin/env python
# coding: utf-8

# # If you like the notebook, please give it a Upvote!

# ## Importing Libraries

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

plt.style.use('ggplot')


# ## Import Dataset

# In[ ]:


train = pd.read_csv('../input/house-prices-advanced-regression-techniques/train.csv')
test = pd.read_csv('../input/house-prices-advanced-regression-techniques/test.csv')


# In[ ]:


train.shape


# In[ ]:


test.shape


# ## Handling Missing Data
# 
# We'll take a look at the dataset and determine how to handle the missing data.

# In[ ]:


total = train.isnull().sum().sort_values(ascending=False)
percent = (train.isnull().sum()/train.isnull().count()).sort_values(ascending=False)
missing_data = pd.concat([total,percent], axis=1, keys=['Total', 'Percent'])
missing_data.head(20)


# In[ ]:


test_total = test.isnull().sum().sort_values(ascending=False)
test_percent = (test.isnull().sum()/test.isnull().count()).sort_values(ascending=False)
test_missing_data = pd.concat([test_total, test_percent], axis=1, keys=['Total', 'Percent'])
test_missing_data.head(35)


# When there is more than 15% missing data, we should discard it instead of trying any trick to fill the data. 
# 
# As we can see, in both test and train, the features 
# * 'POOLQc'
# * 'MiscFeature'
# * 'Alley'
# 
# are missing more than 95% of the values. Instead of trying fixing them, we should discard those variables.
# 
# Other variable like 'GarageX' are also very co-related and their most important information is depicted by 'GarageCars', similar goes for 'BasmentX' variables.
# 
# Regarding the 'MasVnrArea' and 'MasVnrType', these two have high co-relation coeff with 'YearBuilt'.
#  
# 

# In[ ]:


col = missing_data[missing_data['Total']>1].index
train = train.drop(col, 1)
test = test.drop(col, 1)


# In[ ]:


train.columns


# In[ ]:


test.columns


# Now, we'll work on filling up the data for the Test dataset

# In[ ]:


columns = ['Utilities', 'BsmtFullBath', 'BsmtHalfBath','Functional',
          'SaleType', 'Exterior2nd', 'Exterior1st','KitchenQual']

columns1 = ['BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', 'GarageArea', 'GarageCars']


# In[ ]:


test['MSZoning'] = test['MSZoning'].fillna(test['MSZoning'].mode()[0])


# In[ ]:


for item in columns:
    test[item] = test[item].fillna(test[item].mode()[0])
    
for item in columns1:
    test[item] = test[item].fillna(test[item].median())


# In[ ]:


test.drop(columns=['Id'], inplace=True)
train.drop(columns=['Id'], inplace=True)


# In[ ]:


train = train.drop(train.loc[train['Electrical'].isnull()].index)


# In[ ]:


train.isnull().any().any()


# In[ ]:


train.shape


# In[ ]:


test.isnull().any().any()


# In[ ]:


test.shape


# ## Feature Engineering by OneHotEncoding

# In[ ]:


columns = ['MSZoning', 'Street', 'LotShape', 'LandContour',
           'Utilities', 'LotConfig', 'LandSlope', 'Neighborhood',
           'Condition1', 'Condition2', 'BldgType', 'HouseStyle',
           'RoofStyle', 'RoofMatl', 'Exterior1st', 'Exterior2nd',
           'ExterQual', 'ExterCond', 'Foundation', 'Heating',
           'HeatingQC', 'CentralAir', 'Electrical','KitchenQual',
           'Functional','PavedDrive', 'SaleType', 'SaleCondition']


# In[ ]:


len(columns)


# In[ ]:


final_df = pd.concat([train, test], axis=0)


# In[ ]:


def OneHotEncoding(columns):
    df_final = final_df
    i=0
    for fields in columns:
        df1 = pd.get_dummies(final_df[fields], drop_first=True)
        final_df.drop([fields], axis=1, inplace=True)
        if i==0:
            df_final=df1.copy()
        else:
            df_final = pd.concat([df_final, df1], axis=1)
            
        i=i+1
        
        
    df_final = pd.concat([final_df, df_final], axis=1)
    return df_final


# In[ ]:


final_df = OneHotEncoding(columns)


# In[ ]:


final_df.shape


# In[ ]:


final_df = final_df.loc[:, ~final_df.columns.duplicated()]


# In[ ]:


final_df.shape


# In[ ]:


df_train = final_df.iloc[:1459,:]
df_test = final_df.iloc[1459:,:]


# In[ ]:


df_test.drop(['SalePrice'], axis=1, inplace=True)


# In[ ]:


X_train = df_train.drop(['SalePrice'], axis=1)
y_train = df_train['SalePrice']


# In[ ]:


from sklearn.ensemble import RandomForestClassifier

regressor = RandomForestClassifier()


# ## Hyperparameter Tunning

# In[ ]:


from sklearn.model_selection import RandomizedSearchCV

n_estimators = [100, 500, 900]
criterion = ['gini', 'entropy']
depth = [3, 5, 10, 15]
min_split = [2, 3, 4]
min_leaf = [2, 3, 4]
bootstrap = ['True', 'False']
verbose = [5]

hyperparameter_grid = {
    'n_estimators': n_estimators,
    'max_depth': depth,
    'criterion': criterion,
    'bootstrap': bootstrap,
    'verbose': verbose,
    'min_samples_split': min_split,
    'min_samples_leaf': min_leaf
}

random_cv = RandomizedSearchCV(estimator=regressor,
                              param_distributions = hyperparameter_grid,
                              cv=5,
                              scoring = 'neg_mean_absolute_error',
                              n_jobs = 4,
                              return_train_score=True,
                              random_state=42)


# In[ ]:


random_cv.fit(X_train, y_train)


# In[ ]:


random_cv.best_estimator_


# In[ ]:


regressor = RandomForestClassifier(bootstrap='False', class_weight=None,
                       criterion='entropy', max_depth=10, max_features='auto',
                       max_leaf_nodes=None, min_impurity_decrease=0.0,
                       min_impurity_split=None, min_samples_leaf=3,
                       min_samples_split=3, min_weight_fraction_leaf=0.0,
                       n_estimators=900, n_jobs=None, oob_score=False,
                       random_state=None, verbose=5, warm_start=False)


# In[ ]:


regressor.fit(X_train, y_train)


# In[ ]:


y_pred = regressor.predict(df_test)


# In[ ]:


y_pred


# In[ ]:


pred = pd.DataFrame(y_pred)
samp = pd.read_csv('../input/house-prices-advanced-regression-techniques/test.csv')
sub = pd.concat([samp['Id'], pred], axis=1)
sub.columns = ['Id', 'SalePrice']
sub


# In[ ]:


sub.to_csv('My_Submission.csv', index=False)


# In[ ]:




