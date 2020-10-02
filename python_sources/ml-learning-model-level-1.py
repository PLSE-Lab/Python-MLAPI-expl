#!/usr/bin/env python
# coding: utf-8

# # Introduction
# **This will be your workspace for Kaggle's Machine Learning education track.**
# 
# You will build and continually improve a model to predict housing prices as you work through each tutorial.  Fork this notebook and write your code in it.
# 
# The data from the tutorial, the Melbourne data, is not available in this workspace.  You will need to translate the concepts to work with the data in this notebook, the Iowa data.
# 
# Come to the [Learn Discussion](https://www.kaggle.com/learn-forum) forum for any questions or comments. 
# 
# # Write Your Code Below
# 
# 

# In[134]:


import pandas as pd

main_file_path = '../input/train.csv'
data = pd.read_csv(main_file_path)
print(data.describe())


# In[135]:


print(data.columns)


# In[136]:


print(len(data.columns))


# In[137]:


sales_price = data.SalePrice
print(sales_price.head())


# In[138]:


my_columns = ['Neighborhood', 'SaleCondition']
two_columns = data[my_columns]
two_columns.describe()


# In[139]:


y = sales_price
predictors = ['LotArea', 'YearBuilt', 'OverallCond', 'OverallQual', '1stFlrSF', '2ndFlrSF', 'FullBath', 'BedroomAbvGr', 'TotRmsAbvGrd' ]
X = data[predictors]


# In[140]:


from sklearn.tree import DecisionTreeRegressor

iowa_model = DecisionTreeRegressor()

iowa_model.fit(X, y)


# In[141]:


print("Predictions for first 5 houses:")
print(X.head())
print(iowa_model.predict(X.head()))


# In[142]:


print(iowa_model.predict(X.tail()))


# In[143]:


from sklearn.metrics import mean_absolute_error

predicted_home_prices = iowa_model.predict(X)
mean_absolute_error(y, predicted_home_prices)


# In[144]:


from sklearn.model_selection import train_test_split

# split data into training and validation data, for both predictors and target
# The split is based on a random number generator. Supplying a numeric value to
# the random_state argument guarantees we get the same split every time we
# run this script.
train_X, val_X, train_y, val_y = train_test_split(X, y,random_state = 0)
# Define model
iowa_model = DecisionTreeRegressor()
# Fit model
iowa_model.fit(train_X, train_y)

# get predicted prices on validation data
val_predictions = iowa_model.predict(val_X)
print(mean_absolute_error(val_y, val_predictions))


# In[145]:


def get_mae(max_leaf_nodes, predictors_train, predictors_val, targ_train, targ_val):
    model = DecisionTreeRegressor(max_leaf_nodes=max_leaf_nodes, random_state=0)
    model.fit(predictors_train, targ_train)
    preds_val = model.predict(predictors_val)
    mae = mean_absolute_error(targ_val, preds_val)
    return(mae)


# In[146]:


for max_leaf_nodes in [5, 40, 50, 75, 100, 500, 5000]:
    my_mae = get_mae(max_leaf_nodes, train_X, val_X, train_y, val_y)
    print("Max leaf nodes: %d  \t\t Mean Absolute Error:  %d" %(max_leaf_nodes, my_mae))


# In[147]:


from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

iowa_forest_model = RandomForestRegressor()
iowa_forest_model.fit(train_X, train_y)
iowa_preds = iowa_forest_model.predict(val_X)
print(mean_absolute_error(val_y, iowa_preds))


# In[148]:


print(data.isnull().sum())


# In[149]:


from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split

iowa_target = data.SalePrice
iowa_predictors = data.drop(['SalePrice'], axis=1)

# For the sake of keeping the example simple, we'll use only numeric predictors. 
iowa_numeric_predictors = iowa_predictors.select_dtypes(exclude=['object'])


# In[150]:


X_train, X_test, y_train, y_test = train_test_split(iowa_numeric_predictors, 
                                                    iowa_target,
                                                    train_size=0.7, 
                                                    test_size=0.3, 
                                                    random_state=0)

def score_dataset(X_train, X_test, y_train, y_test):
    model = RandomForestRegressor()
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    return mean_absolute_error(y_test, preds)


# In[151]:


cols_with_missing = [col for col in X_train.columns
                    if X_train[col].isnull().any()]
reduced_X_train = X_train.drop(cols_with_missing, axis = 1)
reduced_X_test = X_test.drop(cols_with_missing, axis = 1)
print("MAE from dropping columns:")
print(score_dataset(reduced_X_train, reduced_X_test, y_train, y_test))


# In[152]:


from sklearn.preprocessing import Imputer

my_imputer = Imputer()
imputed_X_train = my_imputer.fit_transform(X_train)
imputed_X_test = my_imputer.transform(X_test)
print("MAE from Imputation:")
print(score_dataset(imputed_X_train, imputed_X_test, y_train, y_test))


# In[153]:


imputed_X_train_plus = X_train.copy()
imputed_X_test_plus = X_test.copy()

cols_with_missing = (col for col in X_train.columns 
                                 if X_train[col].isnull().any())
for col in cols_with_missing:
    imputed_X_train_plus[col + '_was_missing'] = imputed_X_train_plus[col].isnull()
    imputed_X_test_plus[col + '_was_missing'] = imputed_X_test_plus[col].isnull()

# Imputation
my_imputer = Imputer()
imputed_X_train_plus = my_imputer.fit_transform(imputed_X_train_plus)
imputed_X_test_plus = my_imputer.transform(imputed_X_test_plus)

print("MAE from Imputation while Track What Was Imputed:")
print(score_dataset(imputed_X_train_plus, imputed_X_test_plus, y_train, y_test))


# In[154]:


from sklearn.preprocessing import Imputer
from sklearn.pipeline import make_pipeline
from xgboost import XGBRegressor
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import itertools
plt.style.use('fivethirtyeight')

test_data = pd.read_csv('../input/test.csv')
train_data = pd.read_csv('../input/train.csv')
train_data.dropna(axis=0, subset=['SalePrice'], inplace=True)

new_predictors = [ 'MSZoning', 'LotArea',
       'OverallQual', 'OverallCond', 'YearBuilt', 
        'ExterQual', 'ExterCond', '1stFlrSF', 
        'GrLivArea',
       'SaleCondition']
#new_X = train_data[new_predictors]
new_X = train_data.drop(['SalePrice'], axis=1)
target = train_data.SalePrice

#new_test = test_data[new_predictors]
new_test = test_data

imputed_X_train_plus = new_X.copy()
imputed_X_test_plus = new_test.copy()

low_cardinality_cols = [cname for cname in imputed_X_train_plus.columns if 
                                imputed_X_train_plus[cname].nunique() < 10 and
                                imputed_X_train_plus[cname].dtype == "object"]
numeric_cols = [cname for cname in imputed_X_train_plus.columns if 
                                imputed_X_train_plus[cname].dtype in ['int64', 'float64']]
my_cols = low_cardinality_cols + numeric_cols
train_predictors = imputed_X_train_plus[my_cols]
test_predictors = imputed_X_test_plus[my_cols]

one_hot_encoded_training_predictors = pd.get_dummies(train_predictors)
one_hot_encoded_test_predictors = pd.get_dummies(test_predictors)
final_train, final_test = one_hot_encoded_training_predictors.align(one_hot_encoded_test_predictors,
                                                                    join='left', 
                                                                    axis=1)

cols_with_missing = (col for col in final_train.columns 
                                 if final_train[col].isnull().any())
for col in cols_with_missing:
    final_train[col + '_was_missing'] = final_train[col].isnull()
    final_test[col + '_was_missing'] = final_test[col].isnull()
    
#split training into training and testing    
new_X_train, new_X_test, new_y_train, new_y_test = train_test_split(final_train, 
                                                    target,
                                                    train_size=0.7, 
                                                    test_size=0.3, 
                                                    random_state=0)

my_pipeline = make_pipeline(Imputer(), XGBRegressor(n_estimators=1000, learning_rate=0.05))

# Imputation
my_imputer = Imputer()
final_train = my_imputer.fit_transform(final_train)
final_test = my_imputer.transform(final_test)


# In[155]:


#sns.heatmap(train_data[train_data.columns[1:80]].corr(),annot=True,cmap='RdYlGn')
fig=plt.gcf()
fig.set_size_inches(40,30)
#plt.show()


# In[156]:


from sklearn.ensemble import RandomForestClassifier 
#model= RandomForestClassifier(n_estimators=100,random_state=0)
model = XGBRegressor(n_estimators=1000, learning_rate=0.05)
#X_feat=final_train[final_train.columns[1:80]]
#Y_feat=final_train.SalePrice
#model.fit(final_train,target)
model.fit(final_train, target, early_stopping_rounds=5, 
             eval_set=[(final_train, target)], verbose=False)
#pd.Series(model.feature_importances_).sort_values(ascending=False)


# In[157]:


from xgboost import XGBRegressor

my_pipeline.fit(new_X_train, new_y_train)

trial = my_pipeline.predict(new_X_test)
print("Mean Absolute Error : " + str(mean_absolute_error(trial, new_y_test)))
#my_new_model = XGBRegressor(n_estimators=1000, learning_rate=0.05)
# Add silent=True to avoid printing out updates with each cycle
#my_new_model.fit(final_train, target, early_stopping_rounds=5, 
#             eval_set=[(final_train, target)], verbose=False)
#print("Mean Absolute Error : " + str(mean_absolute_error(my_new_model.predict(final_train), target)))


# In[158]:


new_predicted_prices = my_pipeline.predict(final_test)

print(new_predicted_prices)


# In[159]:


train_predictors.dtypes.sample(5)


# In[160]:


#submit_iowa = RandomForestRegressor()
#submit_iowa.fit(final_train, target)

#predicted_prices = submit_iowa.predict(final_test)

#print(predicted_prices)


# In[161]:


my_submit = pd.DataFrame({'Id': test_data.Id, 'SalePrice': new_predicted_prices})

my_submit.to_csv('submission7.csv', index = False)


# In[ ]:




