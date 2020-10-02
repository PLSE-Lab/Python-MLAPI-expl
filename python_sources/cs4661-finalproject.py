#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import datetime

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# # **1. Load the data**

# helper function to reduce memory usage

# In[ ]:


def reduce_mem_usage(df, verbose=True):
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    start_mem = df.memory_usage().sum() / 1024**2    
    for col in df.columns:
        col_type = df[col].dtypes
        if col_type in numerics:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)  
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)    
    end_mem = df.memory_usage().sum() / 1024**2
    if verbose: print('Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)'.format(end_mem, 100 * (start_mem - end_mem) / start_mem))
    return df


# read csv files and store them into data frames

# In[ ]:


new_transactions = pd.read_csv('/kaggle/input/elo-merchant-category-recommendation/new_merchant_transactions.csv',
                               parse_dates=['purchase_date'])

historical_transactions = pd.read_csv('/kaggle/input/elo-merchant-category-recommendation/historical_transactions.csv',
                                      parse_dates=['purchase_date'])

def read_data(input_file):
    df = pd.read_csv(input_file)
    df['first_active_month'] = pd.to_datetime(df['first_active_month'])
    df['elapsed_time'] = (datetime.date(2018, 2, 1) - df['first_active_month'].dt.date).dt.days
    return df
#_________________________________________
train = read_data('/kaggle/input/elo-merchant-category-recommendation/train.csv')
test = read_data('/kaggle/input/elo-merchant-category-recommendation/test.csv')

target = train['target']
print(target.shape)
print(test.shape)


# In[ ]:


# check on the top 5 from historical transactions

historical_transactions.head()


# # 2. Preprocessing

# One Hot Encoding

# In[ ]:


historical_transactions = pd.get_dummies(historical_transactions, columns=['category_2', 'category_3'])
new_transactions = pd.get_dummies(new_transactions, columns=['category_2', 'category_3'])


# Date processing

# In[ ]:


for df in [historical_transactions, new_transactions]:
    df['purchase_date'] = pd.to_datetime(df['purchase_date'])
    df['year'] = df['purchase_date'].dt.year
    df['weekofyear'] = df['purchase_date'].dt.weekofyear
    df['month'] = df['purchase_date'].dt.month
    df['dayofweek'] = df['purchase_date'].dt.dayofweek
    df['weekend'] = (df.purchase_date.dt.weekday >=5).astype(int)
    df['hour'] = df['purchase_date'].dt.hour
    df['authorized_flag'] = df['authorized_flag'].map({'Y':1, 'N':0})
    df['category_1'] = df['category_1'].map({'Y':1, 'N':0}) 
    #https://www.kaggle.com/c/elo-merchant-category-recommendation/discussion/73244
    df['month_diff'] = ((datetime.datetime.today() - df['purchase_date']).dt.days)//30
    df['month_diff'] += df['month_lag']


# Reduce memory usage

# In[ ]:



historical_transactions = reduce_mem_usage(historical_transactions)
new_transactions = reduce_mem_usage(new_transactions)


# # 3. Feature Engineering

# helper function to apply aggregations on existing features to create new features

# In[ ]:


def aggregate_transactions(history):
    
    history.loc[:, 'purchase_date'] = pd.DatetimeIndex(history['purchase_date']).                                      astype(np.int64) * 1e-9
    
    agg_func = {
    'authorized_flag': ['mean'],
    'category_1': ['sum', 'mean'],
    'category_2_1.0': ['mean'],
    'category_2_2.0': ['mean'],
    'category_2_3.0': ['mean'],
    'category_2_4.0': ['mean'],
    'category_2_5.0': ['mean'],
    'category_3_A': ['mean'],
    'category_3_B': ['mean'],
    'category_3_C': ['mean'],
    'merchant_id': ['nunique'],
    'merchant_category_id': ['nunique'],
    'state_id': ['nunique'],
    'city_id': ['nunique'],
    'subsector_id': ['nunique'],
    'purchase_amount': ['sum', 'mean', 'max', 'min', 'std'],
    'installments': ['sum', 'mean', 'max', 'min', 'std'],
    'purchase_date': [np.ptp, 'min', 'max'],
    'month_lag': ['mean', 'max', 'min', 'std'],
    'month_diff': ['mean'],
    'month': ['nunique'],
    'hour': ['nunique'],
    'weekofyear': ['nunique'],
    'dayofweek': ['nunique'],
    'year': ['nunique'],
    'authorized_flag': ['sum', 'mean'],
    'weekend': ['sum', 'mean']
    }
    
    agg_history = history.groupby(['card_id']).agg(agg_func)
    agg_history.columns = ['_'.join(col).strip() for col in agg_history.columns.values]
    agg_history.reset_index(inplace=True)
    
    df = (history.groupby('card_id')
          .size()
          .reset_index(name='transactions_count'))
    
    agg_history = pd.merge(df, agg_history, on='card_id', how='left')
    
    return agg_history


# history stores aggregated results from historical transactions

# In[ ]:


history = aggregate_transactions(historical_transactions)
history.columns = ['hist_' + c if c != 'card_id' else c for c in history.columns]
history[:5]


# new stores aggregated results from new merchant transactions

# In[ ]:


new = aggregate_transactions(new_transactions)
new.columns = ['new_' + c if c != 'card_id' else c for c in new.columns]
new[:5]


# # 4. Combine dataframes to train and test dataframes

# join datasets on the common id, card_id for both train and test

# In[ ]:


train = pd.merge(train, history, on='card_id', how='left')
test = pd.merge(test, history, on='card_id', how='left')

train = pd.merge(train, new, on='card_id', how='left')
test = pd.merge(test, new, on='card_id', how='left')

history[0::5]


# Impute missing values

# In[ ]:


from sklearn.impute import SimpleImputer
my_imputer = SimpleImputer()


# In[ ]:


# get features and remove any that have the incorrect data type for a data frame 
feature_cols = [col for col in train.columns if col not in ['target', 'first_active_month', 'card_id']]
X = train[feature_cols]

# impute missing values
X = my_imputer.fit_transform(X)

# get the target vector
y = train['target']


# # 5. Split test and training set from train dataframe

# In[ ]:


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.05, random_state=6)


# #  6. Train on any regression models

# Import training models

# In[ ]:


from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

reg_predictions = []


# Train using KNeighborsRegressor

# In[ ]:


myKNeighborsReg = KNeighborsRegressor(n_neighbors = 3)

myKNeighborsReg.fit(X_train, y_train)

y_predict_myKNeighborsReg = myKNeighborsReg.predict(X_test)

reg_predictions.append(y_predict_myKNeighborsReg)

# TODO: find and change a time stamp feature to a float


# Train using DecisionTreeRegressor

# In[ ]:


myDecisionTreeReg = DecisionTreeRegressor(random_state = 5)

myDecisionTreeReg.fit(X_train, y_train)

y_predict_myDecisionTreeReg = myDecisionTreeReg.predict(X_test)

reg_predictions.append(y_predict_myDecisionTreeReg)


# Train using LinearRegression

# In[ ]:


myLinearReg = LinearRegression()

myLinearReg.fit(X_train, y_train)

y_predict_myLinearReg = myLinearReg.predict(X_test)

reg_predictions.append(y_predict_myLinearReg)


# Train using RandomForestRegressor

# In[ ]:


myRandomForestReg = RandomForestRegressor(n_estimators = 9, bootstrap = True, random_state = 3)

myRandomForestReg.fit(X_train, y_train)

y_predict_myRandomForestReg = myRandomForestReg.predict(X_test)

reg_predictions.append(y_predict_myRandomForestReg)

print(X.shape)


# # 7. Check RMSE 

# In[ ]:


from sklearn import metrics

for model, y_prediction in zip(['K Nearest Neighbor: ', 'Decision Tree: ', 'Linear Regression: ', 'Random Forest: '], reg_predictions):
    mse = metrics.mean_squared_error(y_test, y_prediction)
    rmse = np.sqrt(mse)
    print(model + str(rmse))


# # 8. Dimensionality Reduction

# In[ ]:


from sklearn.decomposition import PCA
n = 45 # (n is the number of components (new features)
# after dimensionality reduction)
my_pca = PCA(n_components = n)
# (X_Train is feature matrix of training set before DR,
# X_Train_New is feature matrix of training set after DR):
X_Train_new = my_pca.fit_transform(X_train)
X_Test_new = my_pca.transform(X_test)


# In[ ]:


reg_predictions_new = []


# Train using KNeighborsRegressor

# In[ ]:


myKNeighborsReg = KNeighborsRegressor(n_neighbors = 3)

myKNeighborsReg.fit(X_Train_new, y_train)

y_predict_myKNeighborsReg = myKNeighborsReg.predict(X_Test_new)

reg_predictions_new.append(y_predict_myKNeighborsReg)


# Train using DecisionTreeRegressor

# In[ ]:


myDecisionTreeReg = DecisionTreeRegressor(random_state = 5)

myDecisionTreeReg.fit(X_Train_new, y_train)

y_predict_myDecisionTreeReg = myDecisionTreeReg.predict(X_Test_new)

reg_predictions_new.append(y_predict_myDecisionTreeReg)


# Train using LinearRegression

# In[ ]:


myLinearReg = LinearRegression()

myLinearReg.fit(X_Train_new, y_train)

y_predict_myLinearReg = myLinearReg.predict(X_Test_new)

reg_predictions_new.append(y_predict_myLinearReg)


# Train using RandomForestRegressor

# In[ ]:


myRandomForestReg = RandomForestRegressor(n_estimators = 9, bootstrap = True, random_state = 3)

myRandomForestReg.fit(X_Train_new, y_train)

y_predict_myRandomForestReg = myRandomForestReg.predict(X_Test_new)

reg_predictions_new.append(y_predict_myRandomForestReg)

print(X.shape)


# In[ ]:


for model, y_prediction in zip(['K Nearest Neighbor: ', 'Decision Tree: ', 'Linear Regression: ', 'Random Forest: '], reg_predictions_new):
    mse = metrics.mean_squared_error(y_test, y_prediction)
    rmse = np.sqrt(mse)
    print(model + str(rmse))


# # 9. Repeat steps 6, 7, and 8 on actual test dataframe

# Training and Testing on all new features

# In[ ]:


test_feature_cols = [col for col in test.columns if col not in ['target', 'first_active_month', 'card_id']]
final_test = test[feature_cols]
final_test = my_imputer.fit_transform(final_test)

reg_predictions_final = {}


# Train using KNeighborsRegressor

# In[ ]:


myKNeighborsReg = KNeighborsRegressor(n_neighbors = 3)

myKNeighborsReg.fit(X, y)

y_predict_myKNeighborsReg = myKNeighborsReg.predict(final_test)

reg_predictions_final['K Nearest Neighbor: '] = y_predict_myKNeighborsReg


# Train using DecisionTreeRegressor

# In[ ]:


myDecisionTreeReg = DecisionTreeRegressor(random_state = 5)

myDecisionTreeReg.fit(X, y)

y_predict_myDecisionTreeReg = myDecisionTreeReg.predict(final_test)

reg_predictions_final['Decision Tree: ']= y_predict_myDecisionTreeReg


# Train using LinearRegression

# In[ ]:


myLinearReg = LinearRegression()

myLinearReg.fit(X, y)

y_predict_myLinearReg = myLinearReg.predict(final_test)

reg_predictions_final['Linear Regression: '] = y_predict_myLinearReg


# Train using RandomForestRegressor

# In[ ]:


myRandomForestReg = RandomForestRegressor(n_estimators = 9, bootstrap = True, random_state = 3)

myRandomForestReg.fit(X, y)

y_predict_myRandomForestReg = myRandomForestReg.predict(final_test)

reg_predictions_final['Random Forest: '] = y_predict_myRandomForestReg

print(X.shape)


# In[ ]:


for model, y_prediction in reg_predictions_final.items():
    mse = metrics.mean_squared_error(target.iloc[:len(y_prediction)], y_prediction)
    rmse = np.sqrt(mse)
    print(model + str(rmse))


# Training and testing using dimensionality reduction

# In[ ]:


reg_predictions_final_dr = {}

n = 45

my_pca = PCA(n_components = n)

X_new = my_pca.fit_transform(X)
final_test_new = my_pca.transform(final_test)


# Train using KNeighborsRegressor

# In[ ]:


myKNeighborsReg = KNeighborsRegressor(n_neighbors = 3)

myKNeighborsReg.fit(X_new, y)

y_predict_myKNeighborsReg = myKNeighborsReg.predict(final_test_new)

reg_predictions_final_dr['K Nearest Neighbor: '] = y_predict_myKNeighborsReg


# Train using DecisionTreeRegressor

# In[ ]:


myDecisionTreeReg = DecisionTreeRegressor(random_state = 5)

myDecisionTreeReg.fit(X_new, y)

y_predict_myDecisionTreeReg = myDecisionTreeReg.predict(final_test_new)

reg_predictions_final_dr['Decision Tree: ']= y_predict_myDecisionTreeReg


# Train using LinearRegression

# In[ ]:


myLinearReg = LinearRegression()

myLinearReg.fit(X_new, y)

y_predict_myLinearReg = myLinearReg.predict(final_test_new)

reg_predictions_final_dr['Linear Regression: '] = y_predict_myLinearReg


# Train using RandomForestRegressor

# In[ ]:


myRandomForestReg = RandomForestRegressor(n_estimators = 9, bootstrap = True, random_state = 3)

myRandomForestReg.fit(X_new, y)

y_predict_myRandomForestReg = myRandomForestReg.predict(final_test_new)

reg_predictions_final_dr['Random Forest: '] = y_predict_myRandomForestReg

print(X.shape)


# In[ ]:


for model, y_prediction in reg_predictions_final_dr.items():
    mse = metrics.mean_squared_error(target.iloc[:len(y_prediction)], y_prediction)
    rmse = np.sqrt(mse)
    print(model + str(rmse))

