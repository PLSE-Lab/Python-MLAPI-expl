#!/usr/bin/env python
# coding: utf-8

# ## Settings

# In[ ]:


import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

train = pd.read_csv('../input/home-data-for-ml-course/train.csv')
test = pd.read_csv('../input/home-data-for-ml-course/test.csv')

model = RandomForestRegressor(n_estimators=100, criterion="mse")

features = ['Neighborhood', 'Condition1', 'YearBuilt', 'YearRemodAdd', 'TotalBsmtSF', 'CentralAir', '1stFlrSF', '2ndFlrSF', 'GrLivArea', 'FullBath', 'KitchenAbvGr', 'TotRmsAbvGrd', 'KitchenQual', 'GarageType', 'GarageYrBlt', 'GarageFinish', 'GarageCars', 'GarageQual', 'GarageCond', 'GarageArea', 'WoodDeckSF', 'OpenPorchSF', 'SaleType', 'SaleCondition']

y = train['SalePrice']
X = train[features]

train_X, val_X, train_y, val_y = train_test_split(X,y)
print('settings completed')


# ## Testing Function

# In[ ]:


from sklearn.metrics import mean_squared_log_error

def cal_err(train_X, train_y, val_X, val_y):
    model.fit(train_X, train_y)
    preds = model.predict(val_X)
    err = mean_squared_log_error(val_y, preds)
    return err


# In[ ]:


s = (train_X.dtypes == 'object')
obj_cols = list(s[s].index)
num_cols = list(set(train_X.columns) - set(obj_cols))

print("Categorical variables:")
print(obj_cols)
print("Numerical variabels:")
print(num_cols)


# ## filling NaN

# In[ ]:


# Shape of training data (num_rows, num_columns)
print(train_X.shape)

# Number of missing values in each column of training data
missing_val_count_by_column = (train_X.isnull().sum())
print(missing_val_count_by_column[missing_val_count_by_column > 0])


# In[ ]:


target_rm = ['GarageType', 'GarageFinish', 'GarageQual', 'GarageCond']

# Make copy to avoid changing original data (when imputing)
X_train_plus = train_X.copy()
X_valid_plus = val_X.copy()

# Make new columns indicating whether the house has garage
train_X['hasGarage'] = X_train_plus['GarageType'].isnull() ^ 1
val_X['hasGarage'] = X_valid_plus['GarageType'].isnull() ^ 1

train_X['GarageYrBlt'].fillna(0, inplace=True)
val_X['GarageYrBlt'].fillna(0, inplace=True)

train_X = train_X.drop(target_rm, axis=1)
val_X = val_X.drop(target_rm, axis=1)

obj_cols = list(set(obj_cols) - set(target_rm))


# ## Categorical Variables

# In[ ]:


object_nunique = list(map(lambda col: train_X[col].nunique(), obj_cols))
d = dict(zip(obj_cols, object_nunique))

# Print number of unique entries by column, in ascending order
sorted(d.items(), key=lambda x: x[1])


# attributes of which have less than 10 unique values, do One Hot Encoding<br>
# and the others, Label Encoding

# In[ ]:


from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder

target = []
for item in d:
    if d[item] < 10:
        target.append(item)

OH_encoder = OneHotEncoder(handle_unknown='ignore', sparse=False)
OH_cols_train = pd.DataFrame(OH_encoder.fit_transform(train_X[target]))
OH_cols_valid = pd.DataFrame(OH_encoder.transform(val_X[target]))

# One-hot encoding removed index; put it back
OH_cols_train.index = train_X.index
OH_cols_valid.index = val_X.index

# Remove categorical columns (will replace with one-hot encoding)
num_X_train = train_X.drop(target, axis=1)
num_X_valid = val_X.drop(target, axis=1)

# Add one-hot encoded columns to numerical features
train_X = pd.concat([num_X_train, OH_cols_train], axis=1)
val_X = pd.concat([num_X_valid, OH_cols_valid], axis=1)

# Apply label encoder to each column with categorical data
label_encoder = LabelEncoder()
train_X['Neighborhood'] = label_encoder.fit_transform(train_X['Neighborhood'])
val_X['Neighborhood'] = label_encoder.transform(val_X['Neighborhood'])


# ## Numerical Variables

# In[ ]:


# Nothing to do yet


# ## testing

# score = [0]*1001
# min1 = 10
# for i in range(10,101):
#     for j in range(10):
#         model = RandomForestRegressor(n_estimators=i, criterion="mse")
#         score[i] += cal_err(train_X, train_y, val_X, val_y)
#     if score[min1] > score[i]:
#         min = i
# 
# print('The proper value is '+str(min1))
# print('The score is '+str(score[min1] / 10))

# min2 = 100
# for i in range(100,1001,100):
#     for j in range(10):
#         model = RandomForestRegressor(n_estimators=i, criterion="mse")
#         score[i] += cal_err(train_X, train_y, val_X, val_y)
#     if score[min2] > score[i]:
#         min2 = i
# 
# print('The proper value is '+str(min2))
# print('The score is '+str(score[min2] / 10))
# 
# if score[min1] < score[min2]:
#     min_est = min1
# else:
#     min_est = min2
# 
# ori_score = score[min_est]

# score3 = [0] * 600
# min_leaf = 10
# min_est = 900
# for i in range(10,31):
#     for j in range(10):
#         model = RandomForestRegressor(n_estimators=min_est, criterion = "mse", max_leaf_nodes = i)
#         score3[i] += cal_err(train_X, train_y, val_X, val_y)
#     if score3[min_leaf] > score3[i]:
#         min3 = i
#     
# print('The proper value is '+str(min_leaf))
# print('The score is '+str(score3[min_leaf] / 10))
# print('---------------------------------------------')
# if score3[min_leaf] < ori_score:
#     model = RandomForestRegressor(n_estimators=min_est, criterion = "mse", max_leaf_nodes = min_leaf)
#     print('The final score is' + str(score3[min_leaf] / 10))
# else:
#     model = RandomForestRegressor(n_estimators=min_est, criterion = 'mse')
#     print('The final score is '+str(ori_score / 10))

# In[ ]:


print(cal_err(train_X, train_y, val_X, val_y))


# ## Submission (Not yet)

# In[ ]:


test[features].isnull().sum()


# In[ ]:


test['TotalBsmtSF'].fillna(value=test['TotalBsmtSF'].mode()[0], inplace=True)
test['KitchenQual'].fillna(value=test['KitchenQual'].mode()[0], inplace=True)
test['SaleType'].fillna(value=test['SaleType'].mode()[0], inplace=True)
test['GarageCars'].fillna(value=test['GarageCars'].mean(), inplace=True)
test['GarageArea'].fillna(value=test['GarageArea'].mean(), inplace=True)


# In[ ]:


test_X = test[features]

target_rm = ['GarageType', 'GarageFinish', 'GarageQual', 'GarageCond']

# Make copy to avoid changing original data (when imputing)
X_plus = X.copy()
test_X_plus = test_X.copy()

# Make new columns indicating whether the house has garage
X['hasGarage'] = X['GarageType'].isnull() ^ 1
test_X['hasGarage'] = test_X['GarageType'].isnull() ^ 1

X['GarageYrBlt'].fillna(0, inplace=True)
test_X['GarageYrBlt'].fillna(0, inplace=True)

X = X.drop(target_rm, axis=1)
test_X = test_X.drop(target_rm, axis=1)

obj_cols = list(set(obj_cols) - set(target_rm))

OH_encoder = OneHotEncoder(handle_unknown='ignore', sparse=False)
OH_cols_X = pd.DataFrame(OH_encoder.fit_transform(X[target]))
OH_cols_test = pd.DataFrame(OH_encoder.transform(test_X[target]))

# One-hot encoding removed index; put it back
OH_cols_X.index = X.index
OH_cols_test.index = test_X.index

# Remove categorical columns (will replace with one-hot encoding)
num_X = X.drop(target, axis=1)
num_X_test = test_X.drop(target, axis=1)

# Add one-hot encoded columns to numerical features
X = pd.concat([num_X, OH_cols_X], axis=1)
test_X = pd.concat([num_X_test, OH_cols_test], axis=1)

# Apply label encoder to each column with categorical data
label_encoder = LabelEncoder()
X['Neighborhood'] = label_encoder.fit_transform(X['Neighborhood'])
test_X['Neighborhood'] = label_encoder.transform(test_X['Neighborhood'])


# In[ ]:


model.fit(X,y)

preds = model.predict(test_X)


# In[ ]:


submission = pd.read_csv('../input/home-data-for-ml-course/sample_submission.csv')
submission.head()

submission['SalePrice'] = preds

submission.to_csv('submission_4.csv', index=False)
print('Completed')

