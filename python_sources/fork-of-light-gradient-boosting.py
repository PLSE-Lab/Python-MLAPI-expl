#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

#garbage collector
import gc
gc.enable() #enabling collection


# In[ ]:


PATH = '/kaggle/input/instacart-market-basket-analysis/'


# In[ ]:


#Reading all the datasets.
print('loading files ...')
order_products_prior = pd.read_csv(PATH + 'order_products__prior.csv')
order_products_train = pd.read_csv(PATH + 'order_products__train.csv')
orders = pd.read_csv(PATH + 'orders.csv')
products = pd.read_csv(PATH + 'products.csv', usecols=['product_id', 'aisle_id', 'department_id'])
orders.eval_set = orders.eval_set.replace({'prior': 0, 'train': 1, 'test':2})
orders.days_since_prior_order = orders.days_since_prior_order.fillna(30)
print('done loading')


# In[ ]:


#merging orders and prior datasets
prior_orders = pd.merge(orders, order_products_prior, on='order_id', how='inner')
prior_orders.head()


# In[ ]:


#deleting prior dataset
del order_products_prior
gc.collect()


# # Creating Features using user_id.

# 1. Total number of orders placed by each users. Max of the order_number column.

# In[ ]:


#number of orders placed by each user.
users = prior_orders.groupby(by='user_id')['order_number'].aggregate('max').to_frame('u_num_of_orders').reset_index()
# #converting the datatype to int.
# users.u_num_of_orders = users.u_num_of_orders.astype(np.uint8)
users.head()


# 2. Average number of products bought in each orders.

# In[ ]:


#average products in orders placed by each users.

#1. First getting the total number of products in each order.
total_prd_per_order = prior_orders.groupby(by=['user_id', 'order_id'])['product_id'].aggregate('count').to_frame('total_products_per_order').reset_index()

#2. Getting the average products purchased by each user
avg_products = total_prd_per_order.groupby(by=['user_id'])['total_products_per_order'].mean().to_frame('u_avg_prd').reset_index()
avg_products.head()

#deleting the total_prd_per_order dataframe
del [total_prd_per_order]
gc.collect()

avg_products.head()


# 3. Day of the week the users orders the most.

# In[ ]:


#dow the user has ordered most.
#importing the scipy's stats model
from scipy import stats


#execution will take approx 45sec.
dow = prior_orders.groupby(by=['user_id'])['order_dow'].aggregate(lambda x : stats.mode(x)[0]).to_frame('dow_u_most_orders')
#resetting the index
dow = dow.reset_index()
dow.head()


# 4. Hour of the day the user has placed most of his/her orders.

# In[ ]:


#hour of day the user has ordered most.

#execution will take approx 45sec.
hod = prior_orders.groupby(by=['user_id'])['order_hour_of_day'].aggregate(lambda x : stats.mode(x)[0]).to_frame('hod_u_most_orders')
#resetting the index
hod = hod.reset_index()
hod.head()


# 5. Reordered ratio of each user.

# In[ ]:


#reorder ratio of user.
reorder_u = prior_orders.groupby(by='user_id')['reordered'].aggregate('mean').to_frame('u_reorder_ratio').reset_index()
#changing the dtype.
reorder_u['u_reorder_ratio'] = reorder_u['u_reorder_ratio'].astype(np.float16)
reorder_u.head()


# 6. Average days since prior order.

# In[ ]:


#filling the NAN values with 0.
prior_orders.days_since_prior_order.fillna(0, inplace=True)


# In[ ]:


#average days between orders.
avg_days = prior_orders.groupby(by='user_id')['days_since_prior_order'].aggregate('mean').to_frame('average_days_between_orders')
#resetting index
avg_days = avg_days.reset_index()
avg_days.head()


# 7. Total items bought by user.

# In[ ]:


#total items bought.
total_item = prior_orders.groupby(by='user_id').size().to_frame('u_total_items_bought').astype(np.int16)
total_item.head()


# ** Merging all the created features into the users dataset. **

# In[ ]:


#merging users df and avg_prd
users = users.merge(avg_products, on='user_id', how='left')
#merging users df with dow
users = users.merge(dow, on='user_id', how='left')
#merging users df with hod
users = users.merge(hod, on='user_id', how='left')
#merging users df with reorder_u
users = users.merge(reorder_u, on='user_id', how='left')
#merging users df with avg_days
users = users.merge(avg_days, on='user_id', how='left')
#merging total_item df with reorder_u
users = users.merge(total_item, on='user_id', how='left')

users.head()


# In[ ]:


#deleting unwwanted df
del [reorder_u, dow, hod, avg_products, avg_days, total_item]
gc.collect()


# # Creating features using product_id.

# 1. Number of times the product has been purchased by the users.

# In[ ]:


#number of times purchased.
prd = prior_orders.groupby(by='product_id')['order_id'].aggregate('count').to_frame('p_num_of_times').reset_index()
# prd['p_num_of_times'] = prd['p_num_of_times'].astype(np.uint16)
prd.head()


# 2. Reorder ratio of each products. Number of times the product was reordered / number of times it was purchased.

# In[ ]:


#reordered ratio for each product
reorder_p = prior_orders.groupby(by='product_id')['reordered'].aggregate('mean').to_frame('p_reorder_ratio').reset_index()
# #changing dtype
# reorder_p['p_reorder_ratio'] = reorder_p['p_reorder_ratio'].astype(np.float16)
reorder_p.head()


# 3. Average add to cart order for each product.

# In[ ]:


#add to cart for each product.
add_to_cart = prior_orders.groupby(by='product_id')['add_to_cart_order'].aggregate('mean').to_frame('p_avg_cart_position').reset_index()
# #changing the dtype
# add_to_cart['p_avg_cart_position'] = add_to_cart['p_avg_cart_position'].astype(np.float16)
add_to_cart.head()


# ** Merging all the created features into the prd dataset. **

# In[ ]:


#merging reorder_p with prd.
prd = prd.merge(reorder_p, on='product_id', how='left')

#merging add_to_cart with prd.
prd = prd.merge(add_to_cart, on='product_id', how='left')

#deleting unwanted df.
del [reorder_p, add_to_cart]
gc.collect()


# In[ ]:


prd.head()


# # Creating features using user_id and product_id.

# 1. How many times a User has bought a product.

# In[ ]:


#times a user have bough a product.
uxp = prior_orders.groupby(by=['user_id', 'product_id'])['order_id'].aggregate('count').to_frame('uxp_times_bought')
#resetting index
uxp = uxp.reset_index()
# #changing the dtype.
# uxp['uxp_times_bought'] = uxp['uxp_times_bought'].astype(np.uint8)
uxp.head()


# 2. How many times a user bought a product after its first purchase.

# In[ ]:


#times a user have bough a product.
times = prior_orders.groupby(by=['user_id', 'product_id'])['order_id'].aggregate('count').to_frame('times_bought')
#resetting index
times = times.reset_index()
# #changing the dtype.
# times['times_bought'] = times['times_bought'].astype(np.uint8)
times.head()


# In[ ]:


#Total orders
total_orders = prior_orders.groupby('user_id')['order_number'].max().to_frame('total_orders').reset_index()
total_orders.head()


# In[ ]:


#Finding when the user has bought a product the first time.
first_order_num = prior_orders.groupby(by=['user_id', 'product_id'])['order_number'].aggregate('min').to_frame('first_order_num')
#resetting the index
first_order_num = first_order_num.reset_index()
first_order_num.head()


# In[ ]:


#merging both the dataframes
span = pd.merge(total_orders, first_order_num, on='user_id', how='right')
span.head()


# In[ ]:


#Calculating the order range.
# The +1 includes in the difference the first order were the product has been purchased
span['Order_Range_D'] = span.total_orders - span.first_order_num + 1
span.head()


# In[ ]:


#merging times df with the span
uxp_ratio = pd.merge(times, span, on=['user_id', 'product_id'], how='left')
uxp_ratio.head()


# In[ ]:


#calculating the ratio.
uxp_ratio['uxp_reorder_ratio'] = uxp_ratio.times_bought / uxp_ratio.Order_Range_D
uxp_ratio.head()


# In[ ]:


#dropping all the unwanted columns.
uxp_ratio.drop(['times_bought', 'total_orders', 'first_order_num', 'Order_Range_D'], axis=1, inplace=True)
uxp_ratio.head()


# In[ ]:


#deleting all the unwanted df.
del [times, span, first_order_num, total_orders]
gc.collect()


# ** Merging all the created features into the uxp dataset. **

# In[ ]:


#merging uxp_ratio with uxp.
uxp = uxp.merge(uxp_ratio, on=['user_id', 'product_id'], how='left')
#deleting uxp_ratio
del uxp_ratio
#calling garbage collector.
gc.collect()


# In[ ]:


uxp.head()


# In[ ]:


# #chacging the dtype of uxp_reorder_ratio
# uxp.uxp_reorder_ratio = uxp.uxp_reorder_ratio.astype(np.float16)


# 3. How many times a customer bought a product on its last 5 orders.

# In[ ]:


#Reversing the order number for each product.
prior_orders['order_number_back'] = prior_orders.groupby(by=['user_id'])['order_number'].transform(max) - prior_orders.order_number + 1
prior_orders.head()


# In[ ]:


#keeping only the first 5 orders from the order_number_back.
temp = prior_orders.loc[prior_orders.order_number_back <= 5]
temp.head()


# In[ ]:


#product bought by users in the last_five orders.
last_five = temp.groupby(by=['user_id', 'product_id'])['order_id'].aggregate('count').to_frame('uxp_last_five').reset_index()
last_five.head()


# In[ ]:


#ratio of the products bought in the last_five orders.
last_five['uxp_ratio_last_five'] = last_five.uxp_last_five / 5.0
# #changing the dtype.
# last_five['uxp_ratio_last_five'] = last_five['uxp_ratio_last_five'].astype(np.float16)
last_five.head()


# In[ ]:


# #changin the dtype of uxp_last_five.
# last_five['uxp_last_five'] = last_five['uxp_last_five'].astype(np.uint8)


# In[ ]:


#merging this feature with uxp df.
uxp = uxp.merge(last_five, on=['user_id', 'product_id'], how='left')

del [last_five, temp]
gc.collect()
uxp.head()


# In[ ]:


#filling the NAN values with 0.
uxp.fillna(0, inplace=True)
uxp.head(10)


# In[ ]:


uxp.info()


# # Merging users, prd and uxp dataframes.

# In[ ]:


# #Merge uxp features with the user features
# #Store the results on a new DataFrame
data = uxp.merge(users, on='user_id', how='left')
data.head()


# In[ ]:


#Merging prd features with data.
data = data.merge(prd, on='product_id', how='left')
data.head()


# In[ ]:


#deleting unwanted df.
del [users, prd, uxp]
gc.collect()


# In[ ]:


#shape of the dataset.
data.shape


# # Creating Training and Testing datasets.

# In[ ]:


#keeping only the train and test set from the orders df.
orders_future = orders.loc[((orders.eval_set == 1) | (orders.eval_set == 2)), ['user_id', 'eval_set', 'order_id']]
orders_future.head()


# In[ ]:


#merging the orders_future with data.
data = data.merge(orders_future, on='user_id', how='left')
data.head()


# In[ ]:


#Preparing training data set.
data_train = data[data.eval_set == 1]
data_train.head()


# In[ ]:


#merging the information contained in the order_products__train.csv into data_train.
data_train = data_train.merge(order_products_train[['product_id', 'order_id', 'reordered']], on=['product_id', 'order_id'], how='left')
data_train.head()


# In[ ]:


#filling the NAN values in the reordered
data_train.reordered.fillna(0, inplace=True)


# In[ ]:


# #setting user_id and product_id as index.
# data_train = data_train.set_index(['user_id', 'product_id'])

#deleting eval_set, order_id as they are not needed for training.
data_train.drop(['eval_set', 'order_id'], axis=1, inplace=True)


# In[ ]:


#head()
data_train.head()


# In[ ]:


#Preparing the test dataset.
data_test = data[data.eval_set == 2]
data_test.head()


# In[ ]:


# #setting user_id and product_id as index.
# data_test = data_test.set_index(['user_id', 'product_id'])

#deleting eval_set, order_id as they are not needed for training.
data_test.drop(['eval_set', 'order_id'], axis=1, inplace=True)


# In[ ]:


#shape of train and test.
data_train.shape, data_test.shape


# In[ ]:


# #resetting index
# data_train.reset_index()
# data_test.reset_index()

#adding aisle and department data
products = pd.read_csv(PATH + 'products.csv', usecols=['product_id', 'aisle_id', 'department_id'])

#merging product data into data_train and data_test.
data_train = data_train.merge(products, on='product_id', how='left')
data_test = data_test.merge(products, on='product_id', how='left')

#setting the index again
data_train = data_train.set_index(['user_id', 'product_id'])
data_test = data_test.set_index(['user_id', 'product_id'])


# In[ ]:


#mean encoding categorical variables.
columns_mean = ['aisle_id', 'department_id']
for col in columns_mean:
        mean = data_train.groupby(col).reordered.mean()
        data_train[col] = data_train[col].map(mean)
        data_test[col] = data_test[col].map(mean)


# In[ ]:


#deleting unwanted df and collecting garbage
del [data, orders_future, products, order_products_train]
gc.collect()


# # Model Building.

# In[ ]:


#importing the necessary packages.
from sklearn.model_selection import train_test_split, GridSearchCV, KFold
from sklearn.metrics import f1_score, classification_report
from scikitplot.metrics import plot_confusion_matrix
from scikitplot.classifiers import plot_feature_importances

#importing model packages.
import xgboost as xgb
import lightgbm as lgb


# In[ ]:


#Creating X and y variables.
X = data_train.drop(['reordered', 'uxp_ratio_last_five'], axis=1)
y = data_train.reordered

#splitting dataset into train and test split.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=10)


# In[ ]:


#deleting X, y
del [X, y]
gc.collect()


# In[ ]:


#Creating a light gradient boosting model.
# #Initializing the model
lgbm = lgb.LGBMClassifier(objective='binary', num_leaves=96, max_depth=10)

#fitting the model.
lgbm.fit(X_train, y_train)

#prediction
y_pred = (lgbm.predict_proba(X_test)[:, 1] >= 0.21).astype('int') #setting a threshold.

#Evaluation.
print('F1 Score: {}'.format(f1_score(y_pred, y_test)))
print(classification_report(y_pred, y_test))
plot_confusion_matrix(y_pred, y_test)


# In[ ]:


#plotting feature importance.
plot_feature_importances(lgbm, feature_names=data_test.columns, x_tick_rotation=90, max_num_features=20, figsize=(10,8))


# In[ ]:


#Fitting on entire data.
lgbm.fit(data_train.drop(['reordered', 'uxp_ratio_last_five'], axis=1), data_train.reordered)


# In[ ]:


#making prdeictions on the test dataset
y_pred_test = (lgbm.predict_proba(data_test.drop('uxp_ratio_last_five', axis=1))[:, 1] >= 0.21).astype('int') #setting a threshold.


# In[ ]:


#saving the prediction as a new column in data_test
data_test['prediction'] = y_pred_test
data_test.head()


# In[ ]:


# Reset the index
final = data_test.reset_index()
# Keep only the required columns to create our submission file (for chapter 6)
final = final[['product_id', 'user_id', 'prediction']]

gc.collect()
final.head()


# In[ ]:


#Creating a submission file
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
del orders
del orders_test
gc.collect()

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
        
gc.collect()

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


sub.to_csv('sub.csv', index=False, header=True)


# In[ ]:


del [X, y, X_train, y_train, y_test, X_test, lgbm, y_pred]
gc.collect()

