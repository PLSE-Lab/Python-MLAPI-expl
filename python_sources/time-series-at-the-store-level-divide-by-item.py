#!/usr/bin/env python
# coding: utf-8

# # Concept: model each store as a time series, then divide the store predictions by items. 
# 
# After looking at the data a bit, it looks like item sales are random. Sales at the store level, however, show seasonality, and look more like time series. Can we make predictions at the store level and then break those into item level based on each item's overall sales?
# 
# Note that by taking the time series approach, there's a lot of data I *didn't* utilize (price, events, etc.). So it would be an interesting next step to see how those variables could be included.

# ### Outline:
# 1. Import, melt data
# 2. For each store, find the ratio at which each item contributes to overall sales
# 3. Build an ARMA model for each store and decompose predictions based on (2)
# 4. Utilize the notebook recently posted by Vopani (https://www.kaggle.com/rohanrao/m5-how-to-get-your-public-lb-score-rank) to estimate public leaderboard score
# 5. Submit predictions

# ## 1. Import data

# In[ ]:


# Import libraries

import pandas as pd
import numpy as np


# In[ ]:


# Import data

sample = pd.read_csv('/kaggle/input/m5-forecasting-accuracy/sample_submission.csv')
evaluation = pd.read_csv('/kaggle/input/m5-forecasting-accuracy/sales_train_evaluation.csv')

# Break into the training and validation set

df_train = evaluation.iloc[:, :-28]
df_valid = evaluation.iloc[:, -28:]


# In[ ]:


df_train.head()


# Melting the dataframe will make it so days are rows, rather than columns. It is needed later when sales are aggregated by stores.

# In[ ]:


# Melt the dataframe

d_cols = ['d_' + str(i + 1) for i in range(1913)]

ts_df = pd.melt(frame = df_train, 
                     id_vars = ['id', 'item_id', 'cat_id', 'store_id'],
                     var_name = 'd',
                     value_vars = d_cols,
                     value_name = 'sales')


# ## 2. For each store, find how much each item contributes to overall sales

# In[ ]:


# Aggregate total sales by store

store_level = ts_df[['store_id', 'sales']].groupby('store_id').agg('sum')

# Aggregate total sales by store and item

store_item_level = ts_df[['store_id', 'item_id', 'sales']].groupby(['store_id', 'item_id']).agg('sum')

store_level = store_level.reset_index()
store_item_level = store_item_level.reset_index()


# In[ ]:


# For each store, figure out how much each item contributes to overall sales

totals = store_level.merge(store_item_level, on = 'store_id', how = 'left')
totals['ratio'] = totals.apply(lambda row: row['sales_y'] / row['sales_x'], axis = 1)

# Make two copies of this dataframe, which will u

totals['id'] = totals['item_id'] + '_' + totals['store_id'] + '_' + 'evaluation'


# Now we'll build a dataframe that we'll ultimately use for the submission file

# In[ ]:


# Make two copies of the dataframe
totals1 = totals.copy()
totals2 = totals.copy()

# Add an 'id' column
totals1['id'] = totals1['item_id'] + '_' + totals1['store_id'] + '_' + 'evaluation'
totals2['id'] = totals2['item_id'] + '_' + totals2['store_id'] + '_' + 'validation'

# Concatenate vertically
totals = pd.concat([totals2, totals1], axis = 0)
totals = totals.reset_index().drop('index', axis = 1)

join_df = sample.merge(totals, on = 'id', how = 'inner')


# ## 3. Build an ARMA model for each store and decompose predictions based on (2)

# In[ ]:


# Get a list of all the stores and all the items
stores = df_train['store_id'].unique()
items = df_train['item_id'].unique()

# Get a list of the days we're predicting for
f_cols = ['F' + str(i + 1) for i in range(28)]


# In[ ]:


from statsmodels.tsa.arima_model import ARIMA
import matplotlib.pyplot as plt


# This is where the magic happens!
# 
# The orders of the ARMA model are hard-coded, found through experimenting. This notebook isn't about the theory of time series as much as the approach of building a larger time series and decomposing it, so I'll leave out the details of experimenting.
# 
# Do note that it takes a minute to run. Commenting out the '%%capture' at the top of the cell will give status updates.

# In[ ]:


get_ipython().run_cell_magic('capture', '', "# surpress output from fitting ARMA models\n\n# Make an ARMA model for each store\n\nfor store in stores:\n    \n    print('starting store: ' + store)\n    \n    # Generate a time series by grouping the sales data by store\n    store_data = ts_df[ts_df['store_id'] == store]\n    store_data_sales = store_data[['d', 'sales']].groupby('d').agg(sum)\n    store_data_sales = store_data_sales.reset_index()\n    store_data_sales['d'] = store_data_sales['d'].apply(lambda x: int(x.split('_')[1]))\n    store_data_sales = store_data_sales.sort_values('d')\n    ts = store_data_sales['sales']\n    ts.index = store_data_sales['d']\n\n    # Make an ARMA model\n    if store in ['WI_1', 'WI_3']:\n        order = (7, 0, 2)\n    elif store == 'WI_2':\n        order = (6, 0, 2)\n    else:\n        order = (10, 0, 2)\n    \n    model = ARIMA(ts[-100:], order = order)\n    results_AR = model.fit(disp = -1)\n\n    # b holds the predictions. Note that we predict for 56 additional days (28 validation and 28 evaluation)\n    a = results_AR.predict(start = 0, end = len(ts[-100:]) + 56 - 2)\n    b = a[-56:]\n    b.index = ts[-56:].index + 56 \n    \n    # Add the predictions to our dataframe that we'll ultimately use for submission\n    df = join_df[(join_df['store_id'] == store) & (join_df['id'].str.contains('validation'))]\n    \n    for i, row in df.iterrows():\n        print('working on row ' + str(i))\n        join_df.loc[i, f_cols] = [b.values[j]*row['ratio'] for j in range(28)]\n        join_df.loc[i + 30490, f_cols] = [b.values[j + 28]*row['ratio'] for j in range(28)]\n        \n# Supress output (from ARMA model generation)\n;")


# ## 4. Utilize the notebook recently posted by Vopani (https://www.kaggle.com/rohanrao/m5-how-to-get-your-public-lb-score-rank) to estimate public leaderboard score

# The code below is taken from the notebook above, and the evaluation metric was eventually defined here: https://www.kaggle.com/c/m5-forecasting-accuracy/discussion/133834

# In[ ]:


from typing import Union
from tqdm.notebook import tqdm_notebook as tqdm

## evaluation metric
## from https://www.kaggle.com/c/m5-forecasting-accuracy/discussion/133834 and edited to get scores at all levels
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
            ['item_id', 'store_id']
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

        group_ids = []
        all_scores = []
        for i, group_id in enumerate(self.group_ids):
            lv_scores = self.rmsse(valid_preds.groupby(group_id)[self.valid_target_columns].sum(), i + 1)
            weight = getattr(self, f'lv{i + 1}_weight')
            lv_scores = pd.concat([weight, lv_scores], axis=1, sort=False).prod(axis=1)
            group_ids.append(group_id)
            all_scores.append(lv_scores.sum())

        return group_ids, all_scores  


# In[ ]:


## public LB rank
def get_lb_rank(score):
    """
    Get rank on public LB as of 2020-05-31 23:59:59
    """
    df_lb = pd.read_csv("../input/m5-accuracy-final-public-lb/m5-forecasting-accuracy-publicleaderboard-rank.csv")

    return (df_lb.Score <= score).sum() + 1


# In[ ]:


## reading data
df_calendar = pd.read_csv("../input/m5-forecasting-accuracy/calendar.csv")
df_prices = pd.read_csv("../input/m5-forecasting-accuracy/sell_prices.csv")
df_sample_submission = pd.read_csv("../input/m5-forecasting-accuracy/sample_submission.csv")
df_sample_submission["order"] = range(df_sample_submission.shape[0])

evaluator = WRMSSEEvaluator(df_train, df_valid, df_calendar, df_prices)


# In[ ]:


# The validation predictions are in the first 30490 rows of our prediction dataframe

preds_valid = join_df.loc[:30490,].drop(['store_id', 'sales_x', 'item_id', 'sales_y', 'ratio'], axis = 1)
preds_valid = preds_valid[preds_valid.id.str.contains("validation")]
preds_valid = preds_valid.merge(df_sample_submission[["id", "order"]], on = "id").sort_values("order").drop(["id", "order"], axis = 1).reset_index(drop = True)
preds_valid.rename(columns = {
    "F1": "d_1914", "F2": "d_1915", "F3": "d_1916", "F4": "d_1917", "F5": "d_1918", "F6": "d_1919", "F7": "d_1920",
    "F8": "d_1921", "F9": "d_1922", "F10": "d_1923", "F11": "d_1924", "F12": "d_1925", "F13": "d_1926", "F14": "d_1927",
    "F15": "d_1928", "F16": "d_1929", "F17": "d_1930", "F18": "d_1931", "F19": "d_1932", "F20": "d_1933", "F21": "d_1934",
    "F22": "d_1935", "F23": "d_1936", "F24": "d_1937", "F25": "d_1938", "F26": "d_1939", "F27": "d_1940", "F28": "d_1941"
}, inplace = True)


# In[ ]:


## evaluating random submission
groups, scores = evaluator.score(preds_valid)

score_public_lb = np.mean(scores)
score_public_rank = get_lb_rank(score_public_lb)

for i in range(len(groups)):
    print(f"Score for group {groups[i]}: {round(scores[i], 5)}")

print(f"\nPublic LB Score: {round(score_public_lb, 5)}")
print(f"Public LB Rank: {score_public_rank}")


# ## 5. Submit predictions

# In[ ]:


submission = join_df.drop(['store_id', 'sales_x', 'item_id', 'sales_y', 'ratio'], axis = 1)
submission.to_csv('submission.csv', index = False)


# ### Thanks for looking at my kernel! I didn't have as much time as I would have liked to for this one, but I tried to get something together quickly :)
