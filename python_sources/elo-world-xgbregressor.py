#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb
from sklearn import model_selection, preprocessing
import warnings
import time
import sys
import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import LabelEncoder

warnings.simplefilter(action='ignore', category=FutureWarning)
pd.set_option('display.max_columns', 500)


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


# In[ ]:


new_transactions = pd.read_csv('../input/new_merchant_transactions.csv',
                               parse_dates=['purchase_date'])

historical_transactions = pd.read_csv('../input/historical_transactions.csv',
                                      parse_dates=['purchase_date'])

def binarize(df):
    for col in ['authorized_flag', 'category_1']:
        df[col] = df[col].map({'Y':1, 'N':0})
    return df

historical_transactions = binarize(historical_transactions)
new_transactions = binarize(new_transactions)


# We then load the main files, formatting the dates and extracting the target:

# In[ ]:



from datetime import date

today = date.today()

def read_data(input_file):
    df = pd.read_csv(input_file)
    df['first_active_month'] = pd.to_datetime(df['first_active_month'])
    df['elapsed_time'] = (today - df['first_active_month'].dt.date).dt.days
    return df
#_________________________________________
train = read_data('../input/train.csv')
test = read_data('../input/test.csv')

target = train['target']


# <a id="2"></a> <br>
# ## Feature engineering
# 

# In[ ]:


historical_transactions['month_diff'] = ((datetime.datetime.today() - historical_transactions['purchase_date']).dt.days)//30
historical_transactions['month_diff'] += historical_transactions['month_lag']

new_transactions['month_diff'] = ((datetime.datetime.today() - new_transactions['purchase_date']).dt.days)//30
new_transactions['month_diff'] += new_transactions['month_lag']


# In[ ]:


historical_transactions[:5]


# In[ ]:



historical_transactions = pd.get_dummies(historical_transactions, columns=['category_2', 'category_3'])
new_transactions = pd.get_dummies(new_transactions, columns=['category_2', 'category_3'])

historical_transactions = reduce_mem_usage(historical_transactions)
new_transactions = reduce_mem_usage(new_transactions)

agg_fun = {'authorized_flag': ['mean']}
auth_mean = historical_transactions.groupby(['card_id']).agg(agg_fun)
auth_mean.columns = ['_'.join(col).strip() for col in auth_mean.columns.values]
auth_mean.reset_index(inplace=True)

authorized_transactions = historical_transactions[historical_transactions['authorized_flag'] == 1]
historical_transactions = historical_transactions[historical_transactions['authorized_flag'] == 0]


# In[ ]:


historical_transactions['purchase_month'] = historical_transactions['purchase_date'].dt.month
authorized_transactions['purchase_month'] = authorized_transactions['purchase_date'].dt.month
new_transactions['purchase_month'] = new_transactions['purchase_date'].dt.month


# Then I define two functions that aggregate the info contained in these two tables. The first function aggregates the function by grouping on `card_id`:

# In[ ]:


def aggregate_transactions(history):
    
    history.loc[:, 'purchase_date'] = pd.DatetimeIndex(history['purchase_date']).                                      astype(np.int64) * 1e-9
    
    agg_func = {
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
    'purchase_month': ['mean', 'max', 'min', 'std'],
    'purchase_date': [np.ptp, 'min', 'max'],
    'month_lag': ['mean', 'max', 'min', 'std'],
    'month_diff': ['mean']
    }
    
    agg_history = history.groupby(['card_id']).agg(agg_func)
    agg_history.columns = ['_'.join(col).strip() for col in agg_history.columns.values]
    agg_history.reset_index(inplace=True)
    
    df = (history.groupby('card_id')
          .size()
          .reset_index(name='transactions_count'))
    
    agg_history = pd.merge(df, agg_history, on='card_id', how='left')
    
    return agg_history


# In[ ]:


history = aggregate_transactions(historical_transactions)
history.columns = ['hist_' + c if c != 'card_id' else c for c in history.columns]
history[:5]


# In[ ]:


authorized = aggregate_transactions(authorized_transactions)
authorized.columns = ['auth_' + c if c != 'card_id' else c for c in authorized.columns]
authorized[:5]


# In[ ]:


new = aggregate_transactions(new_transactions)
new.columns = ['new_' + c if c != 'card_id' else c for c in new.columns]
new[:5]


# <a id="3"></a> <br>
# ## 3. Training the model
# We now train the model with the features we previously defined. A first step consists in merging all the dataframes:

# In[ ]:


train = pd.merge(train, history, on='card_id', how='left')
test = pd.merge(test, history, on='card_id', how='left')

train = pd.merge(train, authorized, on='card_id', how='left')
test = pd.merge(test, authorized, on='card_id', how='left')

train = pd.merge(train, new, on='card_id', how='left')
test = pd.merge(test, new, on='card_id', how='left')


# In[ ]:


#card_id is encoded from string to numeric value
from sklearn.preprocessing import LabelEncoder 
lbe = LabelEncoder()
lbe = lbe.fit(train['card_id'])
card_id = lbe.transform(train['card_id']) 
train['card_id'] = card_id 
train['card_id'].head()


# In[ ]:


lbe_activation = LabelEncoder()
lbe = lbe_activation.fit(train['first_active_month'])
first_active_month = lbe.transform(train['first_active_month']) 
train['first_active_month'] = first_active_month 
train['first_active_month'].head()


# In[ ]:


from sklearn.model_selection import train_test_split
y = train['target']
del train['target']
del train ['first_active_month']

X_train, X_test, y_train, y_test = train_test_split(train, y, 
                                                    test_size=0.3, 
                                                    random_state=1234)


# In[ ]:


from sklearn.model_selection import GridSearchCV
model = xgb.XGBRegressor()
parameters = {'nthread':[5], 
              'objective':['reg:linear'],
              'learning_rate': [.03, 0.05, .07],
              'max_depth': [5, 6, 7, 8],
              'min_child_weight': [4],
              'silent': [1],
              'subsample': [0.7],
              'colsample_bytree': [0.7],
              'n_estimators': [100]}

xgb_grid = GridSearchCV(model,
                        parameters,
                        cv = 5,
                        n_jobs = 5,
                        verbose=True) 



# In[ ]:


#this is used to get the best model and parameters
#xgb_grid.fit(X_train, y_train)


# In[ ]:


#print(xgb_grid.best_score_)


# In[ ]:


#print(xgb_grid.best_params_) 


# In[ ]:


#train the model using the best score parameters 
'''
colsample_bytree: Subsample ratio of columns when constructing each tree
learning_rate: Boosting learning rate
max_depth : Maximum tree depth for base learners
min_child_weight: Minimum sum of instance weight(hessian) needed in a child.  
n_estimators: Number of boosted trees to fit
nthread: Number of parallel threads used to run xgboost
objective: Specify the learning task and the corresponding learning objective
silent: Whether to print messages while running boosting 
subsample: Subsample ratio of the training instance
'''

'''model_final = xgb.XGBRegressor(colsample_bytree= 0.7, 
                               learning_rate= 0.03,
                               max_depth= 7, 
                               min_child_weight= 4, 
                               n_estimators= 120,
                               nthread= 4, 
                               objective= 'reg:linear',
                               silent= 1, 
                               subsample= 0.7)'''


# In[ ]:


model_final = xgb.XGBRegressor(colsample_bytree= 0.7, 
                               learning_rate= 0.05,
                               max_depth= 6, 
                               min_child_weight= 4, 
                               n_estimators= 100,
                               nthread= 5, 
                               objective= 'reg:linear',
                               silent= 1, 
                               subsample= 0.7)


# In[ ]:


model_final.fit(X_train, y_train)


# In[ ]:


# make predictions for test data
y_pred = model_final.predict(X_test)


# In[ ]:


from math import sqrt

rms = sqrt(mean_squared_error(y_test, y_pred))  
print(rms ) 


# In[ ]:


import matplotlib.pyplot as plt

xgb.plot_tree(model_final,num_trees=0)
plt.rcParams['figure.figsize'] = [100, 50]
plt.show()


# In[ ]:


xgb.plot_tree(model_final, num_trees=0, rankdir='LR')


# In[ ]:


#feature importance 
xgb.plot_importance(model_final)
plt.show()


# <a id="5"></a> <br>
# ## 5. Submission
# 

# In[ ]:


#predict  score for test data
del test['first_active_month'] 

lbe_test = LabelEncoder()
lbe_test = lbe_test.fit(test['card_id'])
card_id = lbe_test.transform(test['card_id']) 
test['card_id'] = card_id 
test['card_id'].head() 


final_pred = model_final.predict(test)


# In[ ]:


test2 = read_data('../input/test.csv')

sub_df = pd.DataFrame({"card_id":test2["card_id"].values})
sub_df["target"] = final_pred
sub_df.to_csv("submit.csv", index=False)

