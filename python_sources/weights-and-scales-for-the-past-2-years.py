#!/usr/bin/env python
# coding: utf-8

# Version 9 update: Thank you to [jhkerwin](https://www.kaggle.com/jhkerwin), who has found my error in get_weight_scale function. I had to add 5 to the indexing to account for the first index columns that come before the 'd_x' columns begin. 

# In[ ]:


from typing import Union

import numpy as np
import pandas as pd
from tqdm.notebook import tqdm_notebook as tqdm


# # This kernal relies on [WRMSSE Evaluator](https://www.kaggle.com/c/m5-forecasting-accuracy/discussion/133834) by [Sakami](https://www.kaggle.com/sakami). 

# ## Objective: obtain the weights and scaling factors for all series for different time periods for validation purposes. Use the output of this kernal to backtest your models with the proper weight and scaling factors. 
# 
# ## Output: weight_scale_*start_valid*.csv is a file that contains the weights and scaling factors for all series for a validation period starting on day *start_valid*. 
# 
# ### Example: For the weights and scales for the public validation period starting on day 1914, load the weight_scale_1914.csv file. 

# 

# ## Weights for the public validation set are provided by the M5 organizers. Lets load them in and compare them with the weights we get from the Evaluator object.

# In[ ]:


### Loading the weights
m5_weights = pd.read_csv('../input/m5methods/validation/weights_validation.csv')
m5_weights.head()


# In[ ]:


from typing import Union

import numpy as np
import pandas as pd
from tqdm.notebook import tqdm_notebook as tqdm


class WRMSSEEvaluator(object):

    def __init__(self, train_df: pd.DataFrame, valid_df: pd.DataFrame, calendar: pd.DataFrame, prices: pd.DataFrame):
        train_y = train_df.loc[:, train_df.columns.str.startswith('d_')]
        train_target_columns = train_y.columns.tolist()
        weight_columns = train_y.iloc[:, -28:].columns.tolist()

        train_df['all_id'] = 0  # for lv1 aggregation

        id_columns = train_df.loc[:, ~train_df.columns.str.startswith('d_')].columns.tolist()
        valid_target_columns = valid_df.loc[:, valid_df.columns.str.startswith('d_')].columns.tolist()

        if not all([c in valid_df.columns for c in id_columns]):
            valid_df = pd.concat([train_df[id_columns], valid_df], axis=1, sort=False)

        self.train_df = train_df
        self.valid_df = valid_df
        self.calendar = calendar
        self.prices = prices

        self.weight_columns = weight_columns
        self.id_columns = id_columns
        self.valid_target_columns = valid_target_columns

        weight_df = self.get_weight_df()
        self.weight_df = weight_df

        self.group_ids = (
            'all_id',
            'state_id',
            'store_id',
            'cat_id',
            'dept_id',
            ['state_id', 'cat_id'],
            ['state_id', 'dept_id'],
            ['store_id', 'cat_id'],
            ['store_id', 'dept_id'],
            'item_id',
            ['item_id', 'state_id'],
            ['item_id', 'store_id'],
            
        )

        for i, group_id in enumerate(tqdm(self.group_ids)):
            train_y = train_df.groupby(group_id)[train_target_columns].sum()
            scale = []
            for _, row in train_y.iterrows():
                series = row.values[np.argmax(row.values != 0):]
                scale.append(((series[1:] - series[:-1]) ** 2).mean())
            setattr(self, f'lv{i + 1}_scale', np.array(scale))
            setattr(self, f'lv{i + 1}_train_df', train_y)
            setattr(self, f'lv{i + 1}_valid_df', valid_df.groupby(group_id)[valid_target_columns].sum())

            lv_weight = weight_df.groupby(group_id)[weight_columns].sum().sum(axis=1)
            setattr(self, f'lv{i + 1}_weight', lv_weight / lv_weight.sum())

    def get_weight_df(self) -> pd.DataFrame:
        day_to_week = self.calendar.set_index('d')['wm_yr_wk'].to_dict()
        weight_df = self.train_df[['item_id', 'store_id'] + self.weight_columns].set_index(['item_id', 'store_id'])
        weight_df = weight_df.stack().reset_index().rename(columns={'level_2': 'd', 0: 'value'})
        weight_df['wm_yr_wk'] = weight_df['d'].map(day_to_week)

        weight_df = weight_df.merge(self.prices, how='left', on=['item_id', 'store_id', 'wm_yr_wk'])
        weight_df['value'] = weight_df['value'] * weight_df['sell_price']
        weight_df = weight_df.set_index(['item_id', 'store_id', 'd']).unstack(level=2)['value']
        weight_df = weight_df.loc[zip(self.train_df.item_id, self.train_df.store_id), :].reset_index(drop=True)
        weight_df = pd.concat([self.train_df[self.id_columns], weight_df], axis=1, sort=False)
        return weight_df

    def rmsse(self, valid_preds: pd.DataFrame, lv: int) -> pd.Series:
        valid_y = getattr(self, f'lv{lv}_valid_df')
        score = ((valid_y - valid_preds) ** 2).mean(axis=1)
        scale = getattr(self, f'lv{lv}_scale')
        return (score / scale).map(np.sqrt)

    def score(self, valid_preds: Union[pd.DataFrame, np.ndarray]) -> float:
        assert self.valid_df[self.valid_target_columns].shape == valid_preds.shape

        if isinstance(valid_preds, np.ndarray):
            valid_preds = pd.DataFrame(valid_preds, columns=self.valid_target_columns)

        valid_preds = pd.concat([self.valid_df[self.id_columns], valid_preds], axis=1, sort=False)

        all_scores = []
        for i, group_id in enumerate(self.group_ids):
            lv_scores = self.rmsse(valid_preds.groupby(group_id)[self.valid_target_columns].sum(), i + 1)
            weight = getattr(self, f'lv{i + 1}_weight')
            lv_scores = pd.concat([weight, lv_scores], axis=1, sort=False).prod(axis=1)
            all_scores.append(lv_scores.sum())

        return np.mean(all_scores)


# In[ ]:


get_ipython().run_cell_magic('time', '', '# Getting the data and instantiating the Evaluator object to compare the weights agianst the provided weights\ntrain_df = pd.read_csv(\'/kaggle/input/m5-forecasting-accuracy/sales_train_validation.csv\')\ncalendar = pd.read_csv(\'/kaggle/input/m5-forecasting-accuracy/calendar.csv\')\nprices = pd.read_csv(\'/kaggle/input/m5-forecasting-accuracy/sell_prices.csv\')\nfor i in range(1914, 1942):\n    train_df[f"d_{i}"] = 0\n\ntrain_fold_df = train_df.iloc[:, :-28]\nvalid_fold_df = train_df.iloc[:, -28:].copy()\n\ne = WRMSSEEvaluator(train_fold_df, valid_fold_df, calendar, prices)\ndel train_fold_df, train_df, calendar, prices')


# In[ ]:


### Lets get all the weights from the object in one array to compare to m5_weights
e_weights = e.lv1_weight
for i in range(1,12):
    e_weights = np.append(e_weights, getattr(e, f'lv{i + 1}_weight').values)


# In[ ]:


#### Lets see where the biggest discrepency is between the weights
(m5_weights.Weight - e_weights).max()


# Thats not good! we are not matching the given weights

# In[ ]:


# .0057 is way too hight. Lets make a function to find
# out where the problem is. The function will show 
# the max discrepency by level.
def show_diff(lv, mw):
    L = lv
    mw3 = mw[mw.Level_id == f'Level{L}']['Weight']
    mw3
    es3 = getattr(e,f'lv{L}_weight').values
    res = (es3 - mw3).max()
    return res


# In[ ]:


for i in range(12):
    print(f'lv_{i+1} difference is {show_diff(i+1, m5_weights)}')


# We can see the problem is in level 11. Lets look at the data.

# In[ ]:


e.lv11_weight


# In[ ]:


m5_weights[m5_weights.Level_id == f'Level11']


# We can see that the problem is the ordering. The evaluator is sorting by food item, then state, while the m5_weights are sorted by state, then food item. Lets fix the m5_weights

# In[ ]:


# We get the level 11 porting of m5_weights isolated and sorted correctly 
temp_df = m5_weights[m5_weights.Level_id == f'Level11'].sort_values(['Agg_Level_2', 'Agg_Level_1'])
# We also want to know to max and min of the index values so we can 
# reset the index correctly 
print(temp_df.index.min(), temp_df.index.max())


# In[ ]:


# Replace the index so we can put the temp_df 
# into m5_weights easily
temp_df.index = range(3203,12350)
# Insert the temp_df into m5_weights
m5_weights = pd.concat([m5_weights[m5_weights.Level_id != 'Level11'], temp_df]).sort_index()


# In[ ]:


# Check if this new m5_weights dataframe lines up 
# with our evaluators weights
for i in range(12):
    print(f'lv_{i+1} difference is {show_diff(i+1, m5_weights)}')


# Fantastic! All the errors are tiny enough to ignor (I think e-8 is small enough)

# In[ ]:


# Lets also add the scaling factor onto this dataframe as well 
# First make a copy of m5_weights so that we can use it over again. 
m5_w = m5_weights.copy()
# Get the scale for each series from the evaluator
e_scales = e.lv1_scale
for i in range(1,12):
    e_scales = np.append(e_scales, getattr(e, f'lv{i + 1}_scale'))
    
# Now add a scale column to m5_w
m5_w['scale'] = e_scales 

# Make Weight column lowercase cus I like it that way
m5_w = m5_w.rename(columns={'Weight': 'weight'})

# Make a csv file, weight_scale_1914.csv, denoting that 
# we have the weights and the scales for all series 
# for a validation period starting on 1914
m5_w.to_csv('weight_scale_1914.csv')


# In[ ]:


m5_w


# Thats fantastic! We fixed level 11 and now we know that we can produce weights with our evaluator, and then save them simply by concatenating the column onto the updated m5_weights dataframe. The same process can also be done for the scaling factor, but we can't double check it against any given scales for now. I will check at least one set of scaling factors in another notebook. For now, lets make a function to produce weights and scales any validation set starting on day START_VAL

# In[ ]:


get_ipython().run_cell_magic('time', '', '\n# Start by loading in the needed datasets \ntrain_df = pd.read_csv(\'/kaggle/input/m5-forecasting-accuracy/sales_train_validation.csv\')\ncalendar = pd.read_csv(\'/kaggle/input/m5-forecasting-accuracy/calendar.csv\')\nprices = pd.read_csv(\'/kaggle/input/m5-forecasting-accuracy/sell_prices.csv\')\nfor i in range(1914, 1942): # This part needed only to get data for public test period\n    train_df[f"d_{i}"] = 0')


# In[ ]:


def get_weight_scale(start_valid, train_df, calendar, prices):
    """Gets the weight and scale for every series, given the end of know data (end_train), 
     is one day before the input start_valid"""

    # Adding + 5 in order to account for the index columns. Error found by 
    # https://www.kaggle.com/jhkerwin
    train_fold_df = train_df.iloc[:, :start_valid + 5]
    valid_fold_df = train_df.iloc[:, start_valid + 5:start_valid + 28 + 5].copy()

    e = WRMSSEEvaluator(train_fold_df, valid_fold_df, calendar, prices)
    del train_fold_df

    # Get a fresh copy of the index columns of m5_weights
    w = m5_weights.iloc[:, :3].copy()

    # Get all the weights for each series
    e_weights = e.lv1_weight
    for i in range(1,12):
        e_weights = np.append(e_weights, getattr(e, f'lv{i + 1}_weight').values)

    # Get the scale for each series from the evaluator
    e_scales = e.lv1_scale
    for i in range(1,12):
        e_scales = np.append(e_scales, getattr(e, f'lv{i + 1}_scale'))

    # Add the weight and scale columns
    w['weight'] = e_weights
    w['scale'] = e_scales

    # Create a csv file with the data 
    w.to_csv(f'weight_scale_{start_valid}.csv')


# In[ ]:


# Lets get the weight_scale_x dataframe for every 28 day period for the
# past 27 periods (roughly 2 years) to get the plenty of validation period
# information, including the public validation and final test time of year 
# for the past 2 years 
for i in range(28): 
    get_weight_scale(1914 - (28 * i), train_df, calendar, prices)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




