#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import tensorflow as tf
from tensorflow.keras import layers

import matplotlib.pyplot as plt
import seaborn as sns
color = sns.color_palette()

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[ ]:


aisles = pd.read_csv('../input/instacart-market-basket-analysis/aisles.csv')
departments = pd.read_csv('../input/instacart-market-basket-analysis/departments.csv')
order_products_prior = pd.read_csv('../input/instacart-market-basket-analysis/order_products__prior.csv')
order_products_train = pd.read_csv('../input/instacart-market-basket-analysis/order_products__train.csv')
orders = pd.read_csv('../input/instacart-market-basket-analysis/orders.csv')
products = pd.read_csv('../input/instacart-market-basket-analysis/products.csv')


# In[ ]:


order_products_prior = pd.merge(order_products_prior, products, on='product_id', how='left')
order_products_prior = pd.merge(order_products_prior, aisles, on='aisle_id', how='left')
order_products_prior = pd.merge(order_products_prior, departments, on='department_id', how='left')
order_products_prior = pd.merge(order_products_prior, orders, on='order_id', how='left')
order_products_prior.head()


# In[ ]:


order_products_train = pd.merge(order_products_train, products, on='product_id', how='left')
order_products_train = pd.merge(order_products_train, aisles, on='aisle_id', how='left')
order_products_train = pd.merge(order_products_train, departments, on='department_id', how='left')
order_products_train = pd.merge(order_products_train, orders, on='order_id', how='left')
order_products_train.head()


# In[ ]:


# Checking that the columns are exactly the same
a, b = order_products_prior.columns, order_products_train.columns
print(a, "\n" ,b)
set(a).intersection(b)
len(a) == len(b)


# In[ ]:


# Checking to see if dataframe has any rows belonging to the test set
#'test' in order_products_prior.eval_set


# In[ ]:


# Checking to see if dataframe has any rows belonging to the test set
#'test' in order_products_train.eval_set


# In[ ]:


# Reorganizing the column order to own preference
order_products_prior = order_products_prior[['user_id', 'order_number', 'add_to_cart_order', 'order_id', 'product_id', 'aisle_id', 
                                             'product_name','department_id', 'aisle', 'department','add_to_cart_order', 'reordered', 
                                             'eval_set', 'order_dow', 'order_hour_of_day', 'days_since_prior_order']]
order_products_train = order_products_train[['user_id', 'order_number', 'add_to_cart_order', 'order_id', 'product_id', 'aisle_id', 
                                             'product_name','department_id', 'aisle', 'department','add_to_cart_order', 'reordered', 
                                             'eval_set', 'order_dow', 'order_hour_of_day', 'days_since_prior_order']]


# In[ ]:


#order_products_prior = order_products_prior.sort_values(by =['user_id', 'order_number']).reset_index()


# In[ ]:


#order_products_train = order_products_train.sort_values(by =['user_id', 'order_number']).reset_index()
#order_products_train.head(10)


# In[ ]:


#Total number of orders placed by each users
users = order_products_prior.groupby(by='user_id')['order_number'].aggregate('max').to_frame('u_num_of_orders').reset_index()


# In[ ]:


users.head()


# In[ ]:


# Average number of products bought by the user in each purchase.
#1. First getting the total number of products in each order.
total_prd_per_order = order_products_prior.groupby(by=['user_id', 'order_id'])['product_id'].aggregate('count').to_frame('total_products_per_order').reset_index()
total_prd_per_order.head(10)


# In[ ]:


#2. Getting the average products purchased by each user
avg_products = total_prd_per_order.groupby(by=['user_id'])['total_products_per_order'].mean().to_frame('u_avg_prd').reset_index()
avg_products.head()

#deleting the total_prd_per_order dataframe
del total_prd_per_order


# In[ ]:


#dow of most orders placed by each user
from scipy import stats
dow = order_products_prior.groupby(by='user_id')['order_dow'].agg(lambda x: stats.mode(x)[0]).to_frame('dow_most_orders_u').reset_index()
dow.head()


# In[ ]:


#hour of day when most orders placed by each user
from scipy import stats
hod = order_products_prior.groupby(by='user_id')['order_hour_of_day'].agg(lambda x: stats.mode(x)[0]).to_frame('hod_most_orders_u').reset_index()
hod.head()


# In[ ]:


#merging the user created features.

#1. merging avg_products with users
users = users.merge(avg_products, on='user_id', how='left')
#deleting avg_products
del avg_products
users.head()


# In[ ]:


#2. merging dow with users.
users = users.merge(dow, on='user_id', how='left')
#deleting dow
del dow
users.head()


# In[ ]:


#3. merging hod with users
users = users.merge(hod, on='user_id', how='left')
#deleting dow
del hod
users.head()


# In[ ]:


#number of time a product was purchased.
prd = order_products_prior.groupby(by='product_id')['order_id'].agg('count').to_frame('prd_count_p').reset_index()
prd.head()


# In[ ]:


#products reorder ratio.
reorder_p = order_products_prior.groupby(by='product_id')['reordered'].agg('mean').to_frame('p_reordered_ratio').reset_index()
reorder_p.head()


# In[ ]:


#merging the reorder_p with prd
prd = prd.merge(reorder_p, on='product_id', how='left')
#deleting reorder_p
del reorder_p
prd.head()


# In[ ]:


#how many times a user bought the same product.
uxp = order_products_prior.groupby(by=['user_id', 'product_id'])['order_id'].agg('count').to_frame('uxp_times_bought').reset_index()
uxp.head()


# In[ ]:


#reorder ratio of the user for each product.
reorder_uxp = order_products_prior.groupby(by=['user_id', 'product_id'])['reordered'].agg('mean').to_frame('uxp_reordered_ratio').reset_index()
reorder_uxp.head()


# In[ ]:


#merging the two dataframes into one
uxp = uxp.merge(reorder_uxp, on=['user_id', 'product_id'], how='left')
#deleting reorder_uxp
del reorder_uxp
uxp.head()


# In[ ]:


#merging users df into uxp
data = uxp.merge(users, on='user_id', how='left')
data.head()


# In[ ]:


#merging products df into data
data = data.merge(prd, on='product_id', how='left')
data.head()


# In[ ]:


#deleting unwanted dfs
del [users, prd, uxp]


# In[ ]:


#keeping only the train and test eval set from the orders dataframe.
order_future = orders.loc[((orders.eval_set == 'train') | (orders.eval_set == 'test')), ['user_id', 'eval_set', 'order_id']]
order_future.head()


# In[ ]:


#merging the order_future with the data.
data = data.merge(order_future, on='user_id', how='left')
data.head()


# In[ ]:


#preparing the train df.
data_train = data[data.eval_set == 'train']
data_train.head()


# In[ ]:


#merging the information from the order_proucts_train df into the data_train.
data_train = data_train.merge(order_products_train[['product_id', 'order_id', 'reordered']], on=['product_id', 'order_id'], how='left')
data_train.head()


# In[ ]:


#filling the NAN values
data_train.reordered.fillna(0, inplace=True)


# In[ ]:


#setting user_id and product_id as index.
data_train = data_train.set_index(['user_id', 'product_id'])

#deleting eval_set, order_id as they are not needed for training.
data_train.drop(['eval_set', 'order_id'], axis=1, inplace=True)


# In[ ]:


data_train.head()


# In[ ]:


#preparing the test dataset.
data_test = data[data.eval_set == 'test']
data_test.head()


# In[ ]:


#deleting unwanted columns
data_test.drop(['eval_set', 'order_id'], axis=1, inplace=True)


# In[ ]:


#setting user_id and product_id as index.
data_test = data_test.set_index(['user_id', 'product_id'])


# In[ ]:


data_test.head()


# In[ ]:


#deleting unwanted df
del [aisles, departments, order_products_prior, order_products_train, orders, order_future, data] #order_future, data


# In[ ]:


#resetting index
data_train.reset_index(inplace=True)
data_test.reset_index(inplace=True)


# In[ ]:


data_train.head()


# In[ ]:


#merging the aisles and department ids to with the train and test data
data_train = data_train.merge(products[['product_id', 'aisle_id']], on='product_id', how='left')
data_test = data_test.merge(products[['product_id', 'aisle_id']], on='product_id', how='left')


# In[ ]:


#department
data_train = data_train.merge(products[['product_id', 'department_id']], on='product_id', how='left')
data_test = data_test.merge(products[['product_id', 'department_id']], on='product_id', how='left')


# In[ ]:


#setting user_id and product_id as index.
data_test = data_test.set_index(['user_id', 'product_id'])
#setting user_id and product_id as index.
data_train = data_train.set_index(['user_id', 'product_id'])


# In[ ]:


y_train = data_train.reordered
X_train = data_train.drop(columns=['reordered'])


# In[ ]:


# Next step is to encode data so that it is useful and impact enough to make good predictions in our model.
# We can start very simply with an XGBoosted predictive model and see what happens.
# Next we can do some feature engineering and feature creation to boost the amount of data that gets fed into our model
# Next we can create an ensemble of models, combine them, and XGBoost the ensemble, like was done in competitions I've seen in the past
# or I can create very deep Neural Network to see if I can get even better results.


# In[ ]:


# Build XGBoost model
import xgboost as xgb

#setting boosters parameters
parameters = {
    'eavl_metric' : 'logloss',
    'max_depth' : 5,
    'colsample_bytree' : 0.4,
    'subsample' : 0.8
}

#instantiating the model
xgb_clf = xgb.XGBClassifier(objective='binary:logistic', parameters=parameters, num_boost_round=10)

# TRAIN MODEL
model = xgb_clf.fit(X_train, y_train)

#FEATURE IMPORTANCE - GRAPHICAL
xgb.plot_importance(model)


# In[ ]:


#predicting on the testing data
y_pred = xgb_clf.predict(data_test).astype('int')

#setting a threshold.
y_pred = (xgb_clf.predict_proba(data_test)[:, 1] >= 0.21).astype('int')
y_pred[0:10]


# In[ ]:


#saving the prediction as a new column in data_test
data_test['prediction'] = y_pred
data_test.head()


# In[ ]:


# Reset the index
final = data_test.reset_index()
# Keep only the required columns to create our submission file
final = final[['product_id', 'user_id', 'prediction']]

#gc.collect()
final.head()


# In[ ]:


#Creating a submission file
PATH = '/kaggle/input/instacart-market-basket-analysis/'
orders = pd.read_csv(PATH + 'orders.csv')
orders_test = orders.loc[orders.eval_set == 'test', ['user_id', 'order_id']]
orders_test.head()


# In[ ]:


#merging our prediction with orders_test
final = final.merge(orders_test, on='user_id', how='left')
final.head()


# In[ ]:


#remove user_id column
final = final.drop('user_id', axis=1)


# In[ ]:


#convert product_id as integer
final['product_id'] = final.product_id.astype(int)

## Remove all unnecessary objects
#del orders
#del orders_test
#gc.collect()

final.head()


# In[ ]:


d = dict()
for row in final.itertuples():
    if row.prediction== 1:
        try:
            d[row.order_id] += ' ' + str(row.product_id)
        except:
            d[row.order_id] = str(row.product_id)

for order in final.order_id:
    if order not in d:
        d[order] = 'None'


#We now check how the dictionary were populated (open hidden output)
#d


# In[ ]:


#Convert the dictionary into a DataFrame
sub = pd.DataFrame.from_dict(d, orient='index')

#Reset index
sub.reset_index(inplace=True)
#Set column names
sub.columns = ['order_id', 'products']

sub.head()


# In[ ]:


sub.to_csv('sub.csv', index=False)


# In[ ]:


from IPython.display import FileLink
FileLink('sub.csv')


# In[ ]:




