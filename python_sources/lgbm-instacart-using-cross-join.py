#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
import gc
import sys

aisles = pd.read_csv('../input/aisles.csv')
departments = pd.read_csv('../input/departments.csv')
order_products_prior = pd.read_csv('../input/order_products__prior.csv')
order_products_train = pd.read_csv('../input/order_products__train.csv')
orders = pd.read_csv('../input/orders.csv')
products = pd.read_csv('../input/products.csv')

shopping_cart = pd.concat(
    [
        order_products_prior,
        order_products_train
    ]
)

t = pd.merge(
    orders,
    shopping_cart,
    on = 'order_id',
    how = 'inner'
)

del shopping_cart
del order_products_prior
del order_products_train

order_id_and_number = t[['user_id', 'order_id', 'order_number','days_since_prior_order','order_dow', 'order_hour_of_day']].drop_duplicates()
order_id_and_number['days_since_prior_order'][order_id_and_number.days_since_prior_order.isnull()] = 0
order_id_and_number = order_id_and_number.sort_values(by = ['user_id', 'order_number'])
order_id_and_number.head()

unique_users = order_id_and_number['user_id'].drop_duplicates().reset_index()
unique_users = unique_users.drop('index', axis = 1)

unique_users_sample = unique_users.sample(n = 1000)

order_id_and_number = pd.merge(
    order_id_and_number,
    unique_users_sample,
    on = 'user_id',
    how = 'inner'
)

o = pd.merge(
    order_id_and_number,
    t,
    on = [
        'order_id',
        'user_id',
        'order_number',
        'order_dow',
        'order_hour_of_day'
    ],
    how = 'inner'
)

o = o.drop('days_since_prior_order_y', axis = 1)
o = o.rename({'days_since_prior_order_x':'days_since_prior_order'}, axis = 'columns')

o.head()


# In[ ]:


# COUNTS - HOUR / DAY
product_rank = t[['product_id', 'order_dow', 'order_hour_of_day']]     .groupby(['product_id', 'order_dow', 'order_hour_of_day'])     .size()     .reset_index(name = 'count')

products0 = products['product_id'].to_frame()
products0['key'] = 1

time = t[['order_dow', 'order_hour_of_day']].drop_duplicates().reset_index().drop('index', axis = 1)
time['key'] = 1

products1 = pd.merge(
    products0,
    time,
    on = 'key',
    how = 'inner'
).drop('key', axis = 1)

product_rank = pd.merge(
    products1,
    product_rank,
    on = ['product_id', 'order_dow', 'order_hour_of_day'],
    how = 'left'
)

product_rank['count'][product_rank['count'].isnull()] = 0

product_rank['sum_help'] = 1

# RANKS - HOUR / DAY - MACRO LEVEL
product_rank = product_rank.sort_values(by = ['order_dow', 'order_hour_of_day','count'], ascending = [True, True, False])

product_rank['rank_macro_day_hour'] = product_rank     .groupby(['order_dow', 'order_hour_of_day'])     .cumsum()['sum_help']

# RANKS - HOUR / DAY - PRODUCT LEVEL
product_rank = product_rank.sort_values(by = ['product_id','count'], ascending = [True, False])

product_rank['rank_product_day_hour'] = product_rank     .groupby(['product_id'])     .cumsum()['sum_help']

product_rank = product_rank.drop('sum_help', axis = 1)
product_rank = product_rank.rename({'count' : 'count_day_hour'}, axis = 'columns')

# COUNTS - DAY LEVEL
time0 = t[['order_dow']].drop_duplicates().reset_index().drop('index', axis = 1)
time0['key'] = 1

products2 = pd.merge(
    products0,
    time0,
    on = 'key',
    how = 'inner'
).drop('key', axis = 1)

product_rank_day = t[['product_id', 'order_dow']]     .groupby(['product_id', 'order_dow'])     .size()     .reset_index(name = 'count')

product_rank_day = pd.merge(
    products2,
    product_rank_day,
    on = ['product_id', 'order_dow'],
    how = 'left'
)

product_rank_day['count'][product_rank_day['count'].isnull()] = 0
product_rank_day['sum_help'] = 1

# RANKS - DAY LEVEL - MACRO LEVEL
product_rank_day = product_rank_day.sort_values(by = ['order_dow','count'], ascending = [True, False])

product_rank_day['rank_macro_day'] = product_rank_day     .groupby(['order_dow'])     .cumsum()['sum_help']

# RANKS - DAY LEVEL - PRODUCT LEVEL
product_rank_day = product_rank_day.sort_values(by = ['product_id','count'], ascending = [True, False])

product_rank_day['rank_product_day'] = product_rank_day     .groupby(['product_id'])     .cumsum()['sum_help']

product_rank_day = product_rank_day.drop('sum_help', axis = 1)
product_rank_day = product_rank_day.rename({'count' : 'count_day'}, axis = 'columns')

# COUNTS - HOUR LEVEL
time1 = t[['order_hour_of_day']].drop_duplicates().reset_index().drop('index', axis = 1)
time1['key'] = 1

products3 = pd.merge(
    products0,
    time1,
    on = 'key',
    how = 'inner'
).drop('key', axis = 1)

product_rank_hour = t[['product_id', 'order_hour_of_day']]     .groupby(['product_id', 'order_hour_of_day'])     .size()     .reset_index(name = 'count')

product_rank_hour = pd.merge(
    products3,
    product_rank_hour,
    on = ['product_id', 'order_hour_of_day'],
    how = 'left'
)

product_rank_hour['count'][product_rank_hour['count'].isnull()] = 0
product_rank_hour['sum_help'] = 1

# RANKS - HOUR LEVEL - MACRO LEVEL
product_rank_hour = product_rank_hour.sort_values(by = ['order_hour_of_day','count'], ascending = [True, False])

product_rank_hour['rank_macro_hour'] = product_rank_hour     .groupby(['order_hour_of_day'])     .cumsum()['sum_help']

# RANKS - HOUR LEVEL - PRODUCT LEVEL
product_rank_hour = product_rank_hour.sort_values(by = ['product_id','count'], ascending = [True, False])

product_rank_hour['rank_product_hour'] = product_rank_hour     .groupby(['product_id'])     .cumsum()['sum_help']

product_rank_hour = product_rank_hour.drop('sum_help', axis = 1)
product_rank_hour = product_rank_hour.rename({'count' : 'count_hour'}, axis = 'columns')

# COUNTS - HOUR LEVEL
product_rank_overall = t[['product_id']]     .groupby(['product_id'])     .size()     .reset_index(name = 'count')

product_rank_overall = pd.merge(
    products0,
    product_rank_overall,
    on = ['product_id'],
    how = 'left'
)

product_rank_overall['count'][product_rank_overall['count'].isnull()] = 0
product_rank_overall['sum_help'] = 1

# RANKS - OVERALL
product_rank_overall = product_rank_overall.sort_values(by = ['count'], ascending = [False])

product_rank_overall['rank_macro_overall'] = product_rank_overall     .cumsum()['sum_help']

product_rank_overall = product_rank_overall.drop(['sum_help', 'key'], axis = 1)
product_rank_overall = product_rank_overall.rename({'count' : 'count_overall'}, axis = 'columns')


# # cross_join = TRAIN dataset

# In[ ]:


first_order = o[['user_id', 'order_number', 'product_id']]     .groupby(['user_id', 'product_id'])     .min()['order_number']     .reset_index(name = 'first_order')

# next: user_products
user_products = o[['user_id', 'product_id']].drop_duplicates().reset_index().drop('index', axis = 1)

# next: user_orders
user_orders = o[['user_id', 'order_number', 'order_dow', 'order_hour_of_day']].drop_duplicates().reset_index().drop('index', axis = 1)

# then join on user_id, filter out when order_number < first_order.order_number
cross_join = pd.merge(
    pd.merge(
        user_products,
        user_orders,
        on = 'user_id',
        how = 'inner'
    ),
    first_order,
    on = ['user_id', 'product_id'],
    how = 'inner'
)

order_days = o[['user_id', 'order_number', 'days_since_prior_order']].drop_duplicates().reset_index().drop('index', axis = 1)

order_days['days_agg'] = order_days     .groupby('user_id')     .cumsum()['days_since_prior_order']

cross_join = pd.merge(
    cross_join,
    order_days,
    on = ['user_id', 'order_number'],
    how = 'inner'
)

product_orders = o[['user_id', 'order_number', 'product_id']]
product_orders['ordered'] = 1

cross_join = pd.merge(
    cross_join,
    product_orders,
    on = ['user_id', 'order_number', 'product_id'],
    how = 'left'
)

cross_join.loc[cross_join.ordered.isnull(), 'ordered'] = 0

cross_join = cross_join.sort_values(by = ['user_id', 'order_number'], ascending = True)

cross_join['count_previous_orders'] = cross_join     .groupby(['user_id', 'product_id'])     .cumsum()['ordered'] - cross_join['ordered']

cross_join['shift_indicator'] = cross_join     .groupby(['user_id', 'product_id'])['ordered']     .shift(1)     .fillna(0)

cross_join['last_order_number'] = 0

cross_join['prev_order_number'] = cross_join     .groupby(['user_id', 'product_id'])['order_number']     .shift(1)

cross_join.loc[cross_join.shift_indicator == 1, 'last_order_number'] = cross_join['prev_order_number']

cross_join['last_order_switch'] = 0
cross_join.loc[cross_join.last_order_number != 0, 'last_order_switch'] = 1
cross_join['bloc_party'] = cross_join     .groupby(['user_id', 'product_id'])     .cumsum()['last_order_switch']

cross_join = pd.merge(
    cross_join,
    cross_join[['user_id', 'product_id', 'bloc_party', 'last_order_number']][cross_join.last_order_number != 0] \
        .drop_duplicates() \
        .reset_index(drop = True) \
        .rename({'last_order_number' : 'bloc_last_order_number'}, axis = 'columns'),
    on = ['user_id', 'product_id', 'bloc_party'],
    how = 'inner'
) \
    .drop('last_order_number', axis = 1) \
    .rename({'bloc_last_order_number' : 'last_order_number'}, axis = 'columns')

cross_join = pd.merge(
    cross_join,
    order_days[['user_id', 'order_number', 'days_agg']].rename({'order_number' : 'last_order_number', 'days_agg' : 'days_agg_last'}, axis = 1),
    on = ['user_id', 'last_order_number'],
    how = 'inner'
)

cross_join['days_since_last_order'] = cross_join['days_agg'] - cross_join['days_agg_last']

user_add_to_cart = o[['user_id', 'product_id', 'order_number', 'add_to_cart_order']]

total_orders = user_add_to_cart     .groupby(['user_id', 'order_number'])     .size()     .reset_index(name = 'num_products_in_order')

user_add_to_cart = pd.merge(
    user_add_to_cart,
    total_orders,
    on = ['user_id', 'order_number'],
    how = 'inner'
)

user_add_to_cart['pct_add_to_cart'] = user_add_to_cart['add_to_cart_order'] / user_add_to_cart['num_products_in_order']

user_add_to_cart['total_cart_order'] = user_add_to_cart     .groupby(['user_id', 'product_id'])     .cumsum()['pct_add_to_cart']

user_add_to_cart['order_count'] = 1

user_add_to_cart['num_orders'] = user_add_to_cart     .groupby(['user_id', 'product_id'])     .cumsum()['order_count']

# denominator: replace order_number with number orders so far
user_add_to_cart['avg_add_to_cart_order'] = user_add_to_cart['total_cart_order'] / user_add_to_cart['num_orders']
cross_join = pd.merge(
    cross_join,
    user_add_to_cart.rename({'order_number' : 'last_order_number'}, axis = 1),
    on = ['user_id', 'product_id', 'last_order_number'],
    how = 'inner'
)

cross_join = cross_join.sort_values(by = ['user_id', 'order_number'])

cross_join['num_orders_three_ago'] = cross_join     .groupby(['user_id', 'product_id'])['num_orders']     .shift(3)     .fillna(0)

cross_join['num_orders_last_three'] =  cross_join['num_orders'] - cross_join['num_orders_three_ago']

product_rank['product_id'] = product_rank['product_id'].astype(int)
product_rank['order_dow'] = product_rank['order_dow'].astype(int)
product_rank['order_hour_of_day'] = product_rank['order_hour_of_day'].astype(int)

product_rank_day['product_id'] = product_rank_day['product_id'].astype(int)
product_rank_day['order_dow'] = product_rank_day['order_dow'].astype(int)

product_rank_hour['product_id'] = product_rank_hour['product_id'].astype(int)
product_rank_hour['order_hour_of_day'] = product_rank_hour['order_hour_of_day'].astype(int)

product_rank_overall['product_id'] = product_rank_overall['product_id'].astype(int)

cross_join = pd.merge(
    cross_join,
    products,
    on = 'product_id',
    how = 'inner'
).drop('product_name', axis = 1)

cross_join = pd.merge(
    cross_join,
    product_rank,
    on = ['product_id', 'order_dow', 'order_hour_of_day'],
    how = 'inner'
)

cross_join = pd.merge(
    cross_join,
    product_rank_day,
    on = ['product_id', 'order_dow'],
    how = 'inner'
)

cross_join = pd.merge(
    cross_join,
    product_rank_hour,
    on = ['product_id', 'order_hour_of_day'],
    how = 'inner'
)

cross_join = pd.merge(
    cross_join,
    product_rank_overall,
    on = ['product_id'],
    how = 'inner'
)

cross_join['order_number'] = cross_join['order_number'].astype(int)
cross_join['product_id'] = cross_join['product_id'].astype(int)
cross_join['user_id'] = cross_join['user_id'].astype(int)

del o


# In[ ]:


from sklearn.preprocessing import PolynomialFeatures

poly = cross_join[
    [
        'num_orders_last_three',
        'count_previous_orders',
        'total_cart_order',
        'days_since_last_order'
    ]
]

p = PolynomialFeatures(degree = 2).fit(poly)

poly = pd.DataFrame(
    data = p.transform(poly),
    columns = p.get_feature_names(poly.columns)
)

poly['user_id'] = cross_join['user_id']
poly['order_number'] = cross_join['order_number']
poly['product_id'] = cross_join['product_id']

cross_join = pd.merge(
    cross_join,
    poly,
    on = [
        'user_id', 
        'order_number', 
        'product_id',
        'num_orders_last_three',
        'count_previous_orders',
        'total_cart_order',
        'days_since_last_order'
    ],
    how = 'inner'
).drop('1', axis = 1)

cross_join.shape


# In[ ]:


from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import roc_curve, auc, roc_auc_score
from sklearn.metrics import f1_score

x = cross_join.copy().drop(
    [
        'ordered',
        'shift_indicator',
        'last_order_number',
        'add_to_cart_order',
        'first_order',
        'num_products_in_order',
        'bloc_party',
        'last_order_switch',
        'prev_order_number',
        'num_orders',
        'order_count',
        'num_orders_three_ago'
    ],
    axis = 1
)
y = cross_join['ordered']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.33, random_state = 0)

model = RandomForestClassifier()
model.fit(x_train, y_train)
predictions = model.predict(x_test)

predictions = pd.DataFrame(
    data = predictions,
    columns = ['prediction']
)

y_test = y_test.reset_index()
y_test = y_test.drop('index', axis = 1)
predictions['actual'] = y_test

auc = roc_auc_score(y_test, predictions['prediction'])
print('AUC: ', auc)

f1 = f1_score(predictions['actual'], predictions['prediction'], average = 'binary')
print('f1: ', f1)

corr = cross_join.corr()
corr = corr['ordered'].to_frame().reset_index()
corr.columns = ['column', 'corr']

importances = model.feature_importances_
cols = x.columns
importances = pd.DataFrame(
    data = {'column' : cols, 'importance': importances}
)

importances = pd.merge(
    importances,
    corr,
    on = 'column',
    how = 'inner'
).sort_values(by = 'corr', ascending = False)
print(importances)


# In[ ]:


import lightgbm as lgb

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.33, random_state = 0)

lgb_model = lgb.LGBMClassifier(
    n_estimators = 1000, 
    objective = 'binary', 
    class_weight = 'balanced', 
    learning_rate = 0.05, 
    reg_alpha = 0.1, 
    reg_lambda = 0.1, 
    subsample = 0.8, 
    n_jobs = -1, 
    random_state = 50
)

lgb_model.fit(
    x_train,
    y_train,
    eval_metric = 'auc'
)

p = lgb_model.predict_proba(x_test)
p_binary = lgb_model.predict(x_test)

lightGBM = roc_auc_score(y_test, p[:,1])
print('LGBM proba:', lightGBM)

lightGBM_binary = roc_auc_score(y_test, p_binary)
print('LGBM binary:', lightGBM_binary)

f1_lgbm = f1_score(y_test, p_binary, average = 'binary')
print('f1 binary: ', f1_lgbm)

lgbm_importances = lgb_model.feature_importances_
cols = x.columns
lgbm_importances = pd.DataFrame(
    data = {'column' : cols, 'importance': lgbm_importances}
).sort_values(by = 'importance', ascending = False)

print(lgbm_importances)


# In[ ]:


p = pd.DataFrame(
    data = p
)

p['actual'] = y_test

p['.1'] = 0
p.loc[p[1] >= .1, '.1'] = 1
lightGBM = roc_auc_score(y_test, p['.1'])
f1_lgbm = f1_score(y_test, p['.1'], average = 'binary')
print('f1: ', f1_lgbm)
print('LGBM: .1:', lightGBM)

p['.2'] = 0
p.loc[p[1] >= .2, '.2'] = 1
lightGBM = roc_auc_score(y_test, p['.2'])
f1_lgbm = f1_score(y_test, p['.2'], average = 'binary')
print('f1: ', f1_lgbm)
print('LGBM: .2:', lightGBM)

p['.3'] = 0
p.loc[p[1] >= .3, '.3'] = 1
lightGBM = roc_auc_score(y_test, p['.3'])
f1_lgbm = f1_score(y_test, p['.3'], average = 'binary')
print('f1: ', f1_lgbm)
print('LGBM: .3:', lightGBM)

p['.4'] = 0
p.loc[p[1] >= .4, '.4'] = 1
lightGBM = roc_auc_score(y_test, p['.4'])
f1_lgbm = f1_score(y_test, p['.4'], average = 'binary')
print('f1: ', f1_lgbm)
print('LGBM: .4:', lightGBM)

p['.5'] = 0
p.loc[p[1] >= .5, '.5'] = 1
lightGBM = roc_auc_score(y_test, p['.5'])
f1_lgbm = f1_score(y_test, p['.5'], average = 'binary')
print('f1: ', f1_lgbm)
print('LGBM: .5:', lightGBM)

p['.6'] = 0
p.loc[p[1] >= .6, '.6'] = 1
lightGBM = roc_auc_score(y_test, p['.6'])
f1_lgbm = f1_score(y_test, p['.6'], average = 'binary')
print('f1: ', f1_lgbm)
print('LGBM: .6:', lightGBM)

p['.7'] = 0
p.loc[p[1] >= .7, '.7'] = 1
lightGBM = roc_auc_score(y_test, p['.7'])
f1_lgbm = f1_score(y_test, p['.7'], average = 'binary')
print('f1: ', f1_lgbm)
print('LGBM: .7:', lightGBM)

p['.8'] = 0
p.loc[p[1] >= .8, '.8'] = 1
lightGBM = roc_auc_score(y_test, p['.8'])
f1_lgbm = f1_score(y_test, p['.8'], average = 'binary')
print('f1: ', f1_lgbm)
print('LGBM: .8:', lightGBM)

p['.9'] = 0
p.loc[p[1] >= .9, '.9'] = 1
lightGBM = roc_auc_score(y_test, p['.9'])
f1_lgbm = f1_score(y_test, p['.9'], average = 'binary')
print('f1: ', f1_lgbm)
print('LGBM: .9:', lightGBM)


# # SET UP TEST DF

# ## Initial driver DataFrame
# 
# ## Discovered cumsum() -> VERY useful tool!

# In[ ]:


test = orders[orders.eval_set == 'test']
test_unique = test['user_id'].drop_duplicates().reset_index().drop('index', axis = 1)
test_unique_orders = pd.merge(
    test_unique,
    orders,
    on = 'user_id',
    how = 'inner'
)
test_unique_orders['days_since_prior_order'][test_unique_orders.days_since_prior_order.isnull()] = 0

test_unique_orders['days'] = test_unique_orders     .groupby('user_id')     .cumsum()['days_since_prior_order']

test_driver = pd.merge(
    test_unique,
    t,
    on = 'user_id',
    how = 'inner'
)


# ## Count of times product ordered previously per user

# In[ ]:


test_counts = test_driver[['user_id', 'product_id']]     .groupby(['user_id', 'product_id'])     .size()     .reset_index(name = 'count_previous_orders')

test_counts.head()


# ## Days since last ordered

# In[ ]:


test_last_order = test_driver[['user_id', 'product_id', 'order_number']]     .groupby(['user_id', 'product_id'])     .max()['order_number']     .reset_index()

test_last_order = pd.merge(
    test[['user_id','order_number']],
    test_last_order,
    on = 'user_id',
    how = 'inner'
).rename({'order_number_x' : 'order_number_current', 'order_number_y' : 'order_number_last'}, axis = 'columns')

test_last_order = pd.merge(
    test_last_order,
    test_unique_orders[['user_id', 'order_number', 'days']],
    left_on = [
        'user_id',
        'order_number_current'
    ],
    right_on = [
        'user_id',
        'order_number'
    ],
    how = 'inner'
) \
    .rename({'days' : 'days_agg'}, axis = 'columns') \
    .drop('order_number', axis = 1)

test_last_order = pd.merge(
    test_last_order,
    test_unique_orders[['user_id', 'order_number', 'days']],
    left_on = [
        'user_id',
        'order_number_last'
    ],
    right_on = [
        'user_id',
        'order_number'
    ],
    how = 'inner'
) \
    .rename({'days' : 'days_agg_last'}, axis = 'columns') \
    .drop('order_number', axis = 1)

test_last_order['days_since_last_order'] = test_last_order['days_agg'] - test_last_order['days_agg_last']

test_last_order.head()


# # Add to cart order

# In[ ]:


add_to_cart_counts = test_driver[['user_id', 'order_number']]     .groupby(['user_id', 'order_number'])     .size()     .reset_index(name = 'total_cart_order')

add_to_cart = pd.merge(
    test_driver[['user_id', 'order_number', 'product_id','add_to_cart_order']],
    add_to_cart_counts,
    on = ['user_id', 'order_number'],
    how = 'inner'
)

add_to_cart['pct_add_to_cart'] = add_to_cart['add_to_cart_order'] / add_to_cart['total_cart_order']
last_order = add_to_cart.copy()

add_to_cart = add_to_cart[['user_id', 'product_id', 'pct_add_to_cart']]     .groupby(['user_id', 'product_id'])     .sum()['pct_add_to_cart']     .reset_index(name = 'total_cart_order')

add_to_cart = pd.merge(
    add_to_cart,
    test_counts,
    on = ['user_id', 'product_id'],
    how = 'inner'
)

add_to_cart['avg_add_to_cart_order'] = add_to_cart['total_cart_order'] / add_to_cart['count_previous_orders']

# add most recent order's pct_add_to_cart
last_order_driver = last_order[['user_id', 'product_id', 'order_number']]     .groupby(['user_id', 'product_id'])     .max()['order_number']     .reset_index(name = 'order_number')

last_order = pd.merge(
    last_order,
    last_order_driver,
    on = ['user_id', 'product_id', 'order_number'],
    how = 'inner'
)

add_to_cart = pd.merge(
    add_to_cart,
    last_order[['user_id', 'product_id', 'pct_add_to_cart']],
    on = ['user_id', 'product_id'],
    how = 'inner'
).drop('count_previous_orders', axis = 1)

add_to_cart.head()


# In[ ]:


last_user_order = test_driver[['user_id', 'order_number']]     .groupby(['user_id'])     .max()['order_number']     .reset_index(name = 'order_number_end')

last_user_order['order_number_start'] = last_user_order['order_number_end'] - 2

last_user_order.loc[last_user_order.order_number_start < 0, 'order_number_start'] = 0

recent_three_orders = pd.merge(
    test_driver,
    last_user_order,
    on = 'user_id',
    how = 'inner'
)

recent_three_orders = recent_three_orders[
    (recent_three_orders.order_number >= recent_three_orders.order_number_start) & 
    (recent_three_orders.order_number <= recent_three_orders.order_number_end)
]

recent_three_orders = recent_three_orders[['user_id', 'product_id']]     .groupby(['user_id', 'product_id'])     .size()     .reset_index(name = 'num_orders_last_three')

recent_three_orders


# # Put it all together
# ## Features:
# 1. Count of times ordered per product (test_counts)
# 2. Days since product ordered (test_last_order)
# 3. days since original order (test_unique_orders)
# 4. days_since_prior_order
# 5. order_number
# 6. product_id
# 7. user_id
# 8. order_dow
# 9. order_hour_of_day

# In[ ]:


test = pd.merge(
    test,
    test_counts,
    on = ['user_id'],
    how = 'inner'
)

test = pd.merge(
    test,
    test_last_order,
    left_on = ['user_id', 'product_id', 'order_number'],
    right_on = ['user_id', 'product_id', 'order_number_current'],
    how = 'inner'
)

test = pd.merge(
    test,
    products,
    on = 'product_id',
    how = 'inner'
).drop('product_name', axis = 'columns')

test = pd.merge(
    test,
    add_to_cart,
    on = ['user_id', 'product_id'],
    how = 'inner'
)

test = pd.merge(
    test,
    product_rank,
    on = ['product_id', 'order_dow', 'order_hour_of_day'],
    how = 'inner'
)

test = pd.merge(
    test,
    product_rank_day,
    on = ['product_id', 'order_dow'],
    how = 'inner'
)

test = pd.merge(
    test,
    product_rank_hour,
    on = ['product_id', 'order_hour_of_day'],
    how = 'inner'
)

test = pd.merge(
    test,
    product_rank_overall,
    on = ['product_id'],
    how = 'inner'
)

test = pd.merge(
    test,
    recent_three_orders,
    on = ['user_id', 'product_id'],
    how = 'left'
)

test['num_orders_last_three'][test.num_orders_last_three.isnull()] = 0

test = test.drop(
    [
        'eval_set',
        'order_number_current',
        'order_number_last',
        'order_id'
    ],
    axis = 1
)


# # Polynomial Features

# In[ ]:


x.dtypes


# In[ ]:


from sklearn.preprocessing import PolynomialFeatures

poly_test = test[
    [
        'num_orders_last_three',
        'count_previous_orders',
        'total_cart_order',
        'days_since_last_order'
    ]
]

p_test = PolynomialFeatures(degree = 2).fit(poly_test)

poly_test = pd.DataFrame(
    data = p_test.transform(poly_test),
    columns = p_test.get_feature_names(poly_test.columns)
)

poly_test['user_id'] = test['user_id']
poly_test['order_number'] = test['order_number']
poly_test['product_id'] = test['product_id']

test = pd.merge(
    test,
    poly_test,
    on = [
        'user_id', 
        'order_number', 
        'product_id',
        'num_orders_last_three',
        'count_previous_orders',
        'total_cart_order',
        'days_since_last_order'
    ],
    how = 'inner'
).drop('1', axis = 1)

test.shape


# In[ ]:


test = test[
    [
        'user_id',
        'product_id',
        'order_number',
        'order_dow',
        'order_hour_of_day',
        'days_since_prior_order',
        'days_agg',
        'count_previous_orders',
        'days_agg_last',
        'days_since_last_order',
        'pct_add_to_cart',
        'total_cart_order',
        'avg_add_to_cart_order',
        'num_orders_last_three',
        'aisle_id',
        'department_id',
        'count_day_hour',
        'rank_macro_day_hour',
        'rank_product_day_hour',
        'count_day',
        'rank_macro_day',
        'rank_product_day',
        'count_hour',
        'rank_macro_hour',
        'rank_product_hour',
        'count_overall',
        'rank_macro_overall',
        'num_orders_last_three^2',
        'num_orders_last_three count_previous_orders',
        'num_orders_last_three total_cart_order',
        'num_orders_last_three days_since_last_order',
        'count_previous_orders^2',
        'count_previous_orders total_cart_order',
        'count_previous_orders days_since_last_order',
        'total_cart_order^2',
        'total_cart_order days_since_last_order',
        'days_since_last_order^2'
    ]
]


# # Create submission file!

# In[ ]:


test['prediction'] = lgb_model.predict_proba(test)[:,1]

submit_staging = test[['user_id', 'product_id', 'order_number']][test.prediction >= .7]

submit_staging = pd.merge(
    submit_staging,
    orders[['user_id', 'order_number', 'order_id']],
    on = ['user_id', 'order_number'],
    how = 'inner'
)

test_left_side = orders[orders.eval_set == 'test']
submit_staging2 = pd.merge(
    test_left_side[['user_id', 'order_id']],
    submit_staging,
    on = ['user_id', 'order_id'],
    how = 'left'
)

submit_staging3 = submit_staging2[['order_id', 'product_id']]
submit_staging3['product_id'][submit_staging3.product_id.isnull()] = 0
submit_staging3 = submit_staging3.astype(int)
submit_staging3 = submit_staging3.astype(str)

submit = submit_staging3     .groupby('order_id')['product_id']     .apply(' '.join)     .reset_index(name = 'products')

submit.loc[submit.products == '0', 'products'] = 'None'

submit.to_csv('submit.csv', index = False)

