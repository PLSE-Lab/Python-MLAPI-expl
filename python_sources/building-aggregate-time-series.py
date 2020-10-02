#!/usr/bin/env python
# coding: utf-8
# # Building aggregated time series
# This kernel builds the aggregate time series needed for the competition. It is based on https://www.kaggle.com/szmnkrisz97/simple-quantiles-of-training-set

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm.auto import tqdm as tqdm

from ipywidgets import widgets, interactive, interact
import ipywidgets as widgets
from IPython.display import display

import os
for dirname, _, filenames in os.walk('input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# ## Reading data

# In[ ]:


train_sales = pd.read_csv('/kaggle/input/m5-forecasting-uncertainty/sales_train_validation.csv')
calendar_df = pd.read_csv('/kaggle/input/m5-forecasting-uncertainty/calendar.csv')
submission_file = pd.read_csv('/kaggle/input/m5-forecasting-uncertainty/sample_submission.csv')
sell_prices = pd.read_csv('/kaggle/input/m5-forecasting-uncertainty/sell_prices.csv')


# ## Variables to help with aggregation

# In[ ]:


total = ['Total']
train_sales['Total'] = 'Total'
train_sales['state_cat'] = train_sales.state_id + "_" + train_sales.cat_id
train_sales['state_dept'] = train_sales.state_id + "_" + train_sales.dept_id
train_sales['store_cat'] = train_sales.store_id + "_" + train_sales.cat_id
train_sales['store_dept'] = train_sales.store_id + "_" + train_sales.dept_id
train_sales['state_item'] = train_sales.state_id + "_" + train_sales.item_id
train_sales['item_store'] = train_sales.item_id + "_" + train_sales.store_id


# In[ ]:


val_eval = ['validation', 'evaluation']

# creating lists for different aggregation levels
total = ['Total']
states = ['CA', 'TX', 'WI']
num_stores = [('CA',4), ('TX',3), ('WI',3)]
stores = [x[0] + "_" + str(y + 1) for x in num_stores for y in range(x[1])]
cats = ['FOODS', 'HOBBIES', 'HOUSEHOLD']
num_depts = [('FOODS',3), ('HOBBIES',2), ('HOUSEHOLD',2)]
depts = [x[0] + "_" + str(y + 1) for x in num_depts for y in range(x[1])]
state_cats = [state + "_" + cat for state in states for cat in cats]
state_depts = [state + "_" + dept for state in states for dept in depts]
store_cats = [store + "_" + cat for store in stores for cat in cats]
store_depts = [store + "_" + dept for store in stores for dept in depts]
prods = list(train_sales.item_id.unique())
prod_state = [prod + "_" + state for prod in prods for state in states]
prod_store = [prod + "_" + store for prod in prods for store in stores]


# ## Getting aggregated sales

# In[ ]:


quants = ['0.005', '0.025', '0.165', '0.250', '0.500', '0.750', '0.835', '0.975', '0.995']
days = range(1, 1913 + 1)
time_series_columns = [f'd_{i}' for i in days]
def CreateSales( name_list, group):
    '''
    This function returns a dataframe (sales) on the aggregation level given by name list and group
    '''
    rows_ve = [(name + "_X_" + str(q) + "_" + ve, str(q)) for name in name_list for q in quants for ve in val_eval]
    sales = train_sales.groupby(group)[time_series_columns].sum() #would not be necessary for lowest level
    return sales


# In[ ]:


total = ['Total']
train_sales['Total'] = 'Total'
train_sales['state_cat'] = train_sales.state_id + "_" + train_sales.cat_id
train_sales['state_dept'] = train_sales.state_id + "_" + train_sales.dept_id
train_sales['store_cat'] = train_sales.store_id + "_" + train_sales.cat_id
train_sales['store_dept'] = train_sales.store_id + "_" + train_sales.dept_id
train_sales['state_item'] = train_sales.state_id + "_" + train_sales.item_id
train_sales['item_store'] = train_sales.item_id + "_" + train_sales.store_id


# In[ ]:


#example usage of CreateSales
sales_by_state_cats = CreateSales(state_cats, 'state_cat')
sales_by_state_cats


# ## Getting quantiles adjusted by day-of-week

# In[ ]:


cols = [i for i in train_sales.columns if i.startswith('d_')]
sales_train_s = train_sales[cols]


# In[ ]:


name = total
group_level = 'Total'
def createTrainSet(sales_train_s, name, group_level, X = False):
    sales_total = CreateSales(name, group_level)
    if(X == True):
        sales_total = sales_total.rename(index = lambda s:  s + '_X')
    sales_train_s = sales_train_s.append(sales_total)
    return(sales_train_s)


# In[ ]:


sales_train_s = pd.DataFrame()
sales_train_s = createTrainSet(sales_train_s, total, 'Total', X=True) #1
sales_train_s = createTrainSet(sales_train_s, states, 'state_id', X=True) #2
sales_train_s = createTrainSet(sales_train_s, stores, 'store_id', X=True) #3
sales_train_s = createTrainSet(sales_train_s, cats, 'cat_id', X=True) #4
sales_train_s = createTrainSet(sales_train_s, depts, 'dept_id', X=True) #5
sales_train_s = createTrainSet(sales_train_s, state_cats, 'state_cat') #6
sales_train_s = createTrainSet(sales_train_s, state_depts, 'state_dept') #7
sales_train_s = createTrainSet(sales_train_s, store_cats, 'store_cat') #8
sales_train_s = createTrainSet(sales_train_s, store_depts, 'store_dept') #9
sales_train_s = createTrainSet(sales_train_s, prods, 'item_id', X=True) #10
sales_train_s = createTrainSet(sales_train_s, prod_state, 'state_item') #11
sales_train_s = createTrainSet(sales_train_s, prod_store, 'item_store')


# In[ ]:


sales_train_s.to_csv('train_set.csv')


# In[ ]:


sales_train_s.head()

