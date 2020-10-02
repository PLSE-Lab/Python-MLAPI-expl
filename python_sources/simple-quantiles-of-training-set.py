#!/usr/bin/env python
# coding: utf-8

# # Simple Historical Quantiles

# If you like this simple model, please upvote:)

# For some reason, the Uncertainty part of the M5 forecasting competition received much lower attention than Accuracy, so much so that at the time of making this notebook (late March), only one competitor's LB score beats the best benchmark model of the organizers.
# 
# The goal of this notebook is to show that simply taking the historical quantiles on different aggregation levels is actually a good idea to start with. Sales data also has a weekly periodicity, to the predictions are adjusted by a day-of-week factor in an elementary way.
# 
# While currently this model achieves a top25 (well, this was only true until early April) score by itself, there is obviously a lot of room for improvement.

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


# In[ ]:


print("Departments: ", depts)
print("Categories by state: ", state_cats)


# In[ ]:


quants = ['0.005', '0.025', '0.165', '0.250', '0.500', '0.750', '0.835', '0.975', '0.995']
days = range(1, 1913 + 1)
time_series_columns = [f'd_{i}' for i in days]


# ## Getting aggregated sales

# In[ ]:


def create_sales(name_list, group):
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
sales_by_state_cats = create_sales(state_cats, 'state_cat')
sales_by_state_cats


# ## Getting quantiles adjusted by day-of-week

# In[ ]:


def create_quantile_dict(name_list = stores, group = 'store_id' ,X = False):
    '''
    This function writes creates sales data on given aggregation level, and then writes predictions to the global dictionary my_dict
    '''
    sales = create_sales(name_list, group)
    sales = sales.iloc[:, 1675:] #using the last few months data only
    sales_quants = pd.DataFrame(index = sales.index)
    for q in quants:
        sales_quants[q] = np.quantile(sales, float(q), axis = 1)
    full_mean = pd.DataFrame(np.mean(sales, axis = 1))
    daily_means = pd.DataFrame(index = sales.index)
    for i in range(7):
        daily_means[str(i)] = np.mean(sales.iloc[:, i::7], axis = 1)
    daily_factors = daily_means / np.array(full_mean)

    daily_factors = pd.concat([daily_factors, daily_factors, daily_factors, daily_factors], axis = 1)
    daily_factors_np = np.array(daily_factors)

    factor_df = pd.DataFrame(daily_factors_np, columns = submission_file.columns[1:])
    factor_df.index = daily_factors.index

    for i,x in enumerate(tqdm(sales_quants.index)):
        for q in quants:
            v = sales_quants.loc[x, q] * np.array(factor_df.loc[x, :])
            if X:
                my_dict[x + "_X_" + q + "_validation"] = v
                my_dict[x + "_X_" + q + "_evaluation"] = v
            else:
                my_dict[x + "_" + q + "_validation"] = v
                my_dict[x + "_" + q + "_evaluation"] = v


# In[ ]:


my_dict = {}
#adding prediction to my_dict on all 12 aggregation levels
create_quantile_dict(total, 'Total', X=True) #1
create_quantile_dict(states, 'state_id', X=True) #2
create_quantile_dict(stores, 'store_id', X=True) #3
create_quantile_dict(cats, 'cat_id', X=True) #4
create_quantile_dict(depts, 'dept_id', X=True) #5
create_quantile_dict(state_cats, 'state_cat') #6
create_quantile_dict(state_depts, 'state_dept') #7
create_quantile_dict(store_cats, 'store_cat') #8
create_quantile_dict(store_depts, 'store_dept') #9
create_quantile_dict(prods, 'item_id', X=True) #10
create_quantile_dict(prod_state, 'state_item') #11
create_quantile_dict(prod_store, 'item_store') #12


# ## Creating valid prediction df from my_dict

# In[ ]:


pred_df = pd.DataFrame(my_dict)
pred_df = pred_df.transpose()
pred_df_reset = pred_df.reset_index()
final_pred = pd.merge(pd.DataFrame(submission_file.id), pred_df_reset, left_on = 'id', right_on = 'index')
del final_pred['index']
final_pred = final_pred.rename(columns={0: 'F1', 1: 'F2', 2: 'F3', 3: 'F4', 4: 'F5', 5: 'F6', 6: 'F7', 7: 'F8', 8: 'F9',
                                        9: 'F10', 10: 'F11', 11: 'F12', 12: 'F13', 13: 'F14', 14: 'F15', 15: 'F16',
                                        16: 'F17', 17: 'F18', 18: 'F19', 19: 'F20', 20: 'F21', 21: 'F22', 
                                        22: 'F23', 23: 'F24', 24: 'F25', 25: 'F26', 26: 'F27', 27: 'F28'})
final_pred = final_pred.fillna(0)


# In[ ]:


final_pred.to_csv('submission.csv', index=False)


# In[ ]:


final_pred.head()

