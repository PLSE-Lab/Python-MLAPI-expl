#!/usr/bin/env python
# coding: utf-8

# ## M5 Weights
# ![](https://i.imgur.com/yRPR0pY.png)
# 
# The evaluation metric of this competition is quite unique in the way that the weights of observations is dependent on historic values of the time series and hence changes for every validation / test dataset.
# 
# **This means that the public LB and the private LB have different weights.** Now that the validation data is released, we can calculate the weights used for the test data and also compare the weights between the public LB and private LB.
# 
# This notebook demonstrates some of the changes / differences between the weights and hopefully it can be used to ensure models don't overfit to the public LB. The final weights are also saved in the output of the notebook.
# 
# I've also shared a notebook of how you can deep dive into analyzing your submission on the public LB: https://www.kaggle.com/rohanrao/m5-anatomy-of-the-public-lb
# 

# In[ ]:


## importing packages
import numpy as np
import pandas as pd

from bokeh.layouts import column, row
from bokeh.models import ColumnDataSource, LinearAxis, Range1d
from bokeh.models.tools import HoverTool
from bokeh.plotting import figure, output_notebook, show
from bokeh.transform import dodge

from math import pi
from typing import Union
from tqdm.notebook import tqdm

output_notebook()


# ## Weights calculation
# Thanks to [sakami](https://www.kaggle.com/sakami) for providing a neat class for calculation of weights [here](https://www.kaggle.com/c/m5-forecasting-accuracy/discussion/133834).
# 

# In[ ]:


## evaluation metric
## from https://www.kaggle.com/c/m5-forecasting-accuracy/discussion/133834
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
            'cat_id',
            'state_id',
            'dept_id',
            'store_id',
            'item_id',
            ['state_id', 'cat_id'],
            ['state_id', 'dept_id'],
            ['store_id', 'cat_id'],
            ['store_id', 'dept_id'],
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

    def score(self, valid_preds: Union[pd.DataFrame, np.ndarray]):
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


## reading data
df_train_full =  pd.read_csv("../input/m5-forecasting-accuracy/sales_train_evaluation.csv")
df_calendar = pd.read_csv("../input/m5-forecasting-accuracy/calendar.csv")
df_prices = pd.read_csv("../input/m5-forecasting-accuracy/sell_prices.csv")

df_train = df_train_full.iloc[:, :-28]
df_valid = df_train_full.iloc[:, -28:]

df_test = df_valid.copy() + np.random.randint(100, size = df_valid.shape)
df_test.columns = ["d_" + str(x) for x in range(1942, 1970)]

evaluator_1 = WRMSSEEvaluator(df_train, df_valid, df_calendar, df_prices)
evaluator_2 = WRMSSEEvaluator(df_train_full, df_test, df_calendar, df_prices)


# ## 1. Store-Item
# Let's explore how different are the weights at store-item level.
# 

# In[ ]:


df_old = evaluator_1.lv12_weight.reset_index().rename(columns = {0: "weight_old"})
df_new = evaluator_2.lv12_weight.reset_index().rename(columns = {0: "weight_new"})

df = df_old.merge(df_new, on = ["store_id", "item_id"])
df["store_item_id"] = df.store_id + "-" + df.item_id
df["weight_diff"] = df.weight_new - df.weight_old
df["weight_perc"] = (df.weight_new - df.weight_old) * 100 / df.weight_old

source = ColumnDataSource(df)

tooltips_1 = [
    ("Store-Item", "@store_item_id"),
    ("Old Weight", "@weight_old{0.0000}"),
    ("New Weight", "@weight_new{0.0000}")
]

v1 = figure(plot_width = 600, plot_height = 300, tooltips = tooltips_1, title = "Old vs New Weights for Store-Item")
v1.circle("weight_old", "weight_new", source = source, size = 10, color = "steelblue", alpha = 0.6, legend_label = "Store-Item")

v1.xaxis.axis_label = "Old Weight"
v1.yaxis.axis_label = "New Weight"

v1.legend.location = "top_left"


df_diff_bottom = df.sort_values("weight_diff").head(10)
df_diff_bottom["weight_diff_min"] = df_diff_bottom.weight_diff
df_diff_top = df.sort_values("weight_diff", ascending = False).head(10).sort_values("weight_diff")
df_diff_top["weight_diff_max"] = df_diff_top.weight_diff

df_diff = pd.concat([df_diff_bottom, df_diff_top])

source_2 = ColumnDataSource(df_diff)

tooltips_2 = [
    ("Store-Item", "@store_item_id"),
    ("Old Weight", "@weight_old{0.0000}"),
    ("New Weight", "@weight_new{0.0000}"),
    ("Weight Absolute Change", "@weight_diff{0.0000}")
]

max_diff = max(abs(np.nanmin(df_diff.weight_diff_min.values)), np.nanmax(df_diff.weight_diff_max.values))

v2 = figure(plot_width = 650, plot_height = 400, y_range = df_diff.store_item_id, x_range = Range1d(-max_diff * 1.1, max_diff * 1.1), tooltips = tooltips_2, title = "Top Absolute Weight Change for Store-Item")
v2.hbar("store_item_id", right = "weight_diff_min", source = source_2, height = 0.75, color = "red", alpha = 0.6)
v2.hbar("store_item_id", right = "weight_diff_max", source = source_2, height = 0.75, color = "green", alpha = 0.6)

v2.xaxis.axis_label = "Absolute Weight Difference"
v2.yaxis.axis_label = "Store-Item"


df_perc_bottom = df[df.weight_old != 0].sort_values("weight_perc").head(10)
df_perc_bottom["weight_perc_min"] = df_perc_bottom.weight_perc
df_perc_top = df[df.weight_old != 0].sort_values("weight_perc", ascending = False).head(10).sort_values("weight_perc")
df_perc_top["weight_perc_max"] = df_perc_top.weight_perc

df_perc = pd.concat([df_perc_bottom, df_perc_top])

source_3 = ColumnDataSource(df_perc)

tooltips_3 = [
    ("Store-Item", "@store_item_id"),
    ("Old Weight", "@weight_old{0.000000}"),
    ("New Weight", "@weight_new{0.000000}"),
    ("Weight Percentage Change", "@weight_perc{0.00}%")
]

max_perc = max(abs(np.nanmin(df_perc.weight_perc_min.values)), np.nanmax(df_perc.weight_perc_max.values))

v3 = figure(plot_width = 650, plot_height = 400, y_range = df_perc.store_item_id, x_range = Range1d(-max_perc * 1.1, max_perc * 1.1), tooltips = tooltips_3, title = "Top Percentage Weight Change for Store-Item")
v3.hbar("store_item_id", right = "weight_perc_min", source = source_3, height = 0.75, color = "red", alpha = 0.6)
v3.hbar("store_item_id", right = "weight_perc_max", source = source_3, height = 0.75, color = "green", alpha = 0.6)

v3.xaxis.axis_label = "Percentage Weight Difference"
v3.yaxis.axis_label = "Store-Item"


df_top_old = df.sort_values("weight_old", ascending = False).head(20)

source_4 = ColumnDataSource(df_top_old)

tooltips_4 = [
    ("Store-Item", "@store_item_id"),
    ("Old Weight", "@weight_old{0.000000}"),
    ("New Weight", "@weight_new{0.000000}")
]

v4 = figure(plot_width = 650, plot_height = 400, x_range = df_top_old.store_item_id, tooltips = tooltips_4, title = "Top Store-Item with old weights")
v4.vbar(x = dodge("store_item_id", -0.15, range = v4.x_range), top = "weight_old", source = source_4, width = 0.2, color = "orange", alpha = 0.6, legend_label = "Old Weight")
v4.vbar(x = dodge("store_item_id", 0.15, range = v4.x_range), top = "weight_new", source = source_4, width = 0.2, color = "maroon", alpha = 0.6, legend_label = "New Weight")

v4.xaxis.major_label_orientation = pi / 2

v4.xaxis.axis_label = "Store-Item"
v4.yaxis.axis_label = "Weight"


df_top_new = df.sort_values("weight_new", ascending = False).head(20)

source_5 = ColumnDataSource(df_top_new)

tooltips_5 = [
    ("Store-Item", "@store_item_id"),
    ("Old Weight", "@weight_old{0.000000}"),
    ("New Weight", "@weight_new{0.000000}")
]

v5 = figure(plot_width = 650, plot_height = 400, x_range = df_top_new.store_item_id, tooltips = tooltips_5, title = "Top Store-Item with new weights")
v5.vbar(x = dodge("store_item_id", -0.15, range = v5.x_range), top = "weight_old", source = source_5, width = 0.2, color = "orange", alpha = 0.6, legend_label = "Old Weight")
v5.vbar(x = dodge("store_item_id", 0.15, range = v5.x_range), top = "weight_new", source = source_5, width = 0.2, color = "maroon", alpha = 0.6, legend_label = "New Weight")

v5.xaxis.major_label_orientation = pi / 2

v5.xaxis.axis_label = "Store-Item"
v5.yaxis.axis_label = "Weight"


show(column(v1, v2, v3, v4, v5))


# In[ ]:


df.to_csv("weights_store_item.csv", index = False)
df.head()


# ## 2. State-Item
# Let's explore how different are the weights at state-item level.

# In[ ]:


df_old = evaluator_1.lv11_weight.reset_index().rename(columns = {0: "weight_old"})
df_new = evaluator_2.lv11_weight.reset_index().rename(columns = {0: "weight_new"})

df = df_old.merge(df_new, on = ["state_id", "item_id"])
df["state_item_id"] = df.state_id + "-" + df.item_id
df["weight_diff"] = df.weight_new - df.weight_old
df["weight_perc"] = (df.weight_new - df.weight_old) * 100 / df.weight_old

source = ColumnDataSource(df)

tooltips_1 = [
    ("State-Item", "@state_item_id"),
    ("Old Weight", "@weight_old{0.0000}"),
    ("New Weight", "@weight_new{0.0000}")
]

v1 = figure(plot_width = 600, plot_height = 300, tooltips = tooltips_1, title = "Old vs New Weights for State-Item")
v1.circle("weight_old", "weight_new", source = source, size = 10, color = "steelblue", alpha = 0.6, legend_label = "state-Item")

v1.xaxis.axis_label = "Old Weight"
v1.yaxis.axis_label = "New Weight"

v1.legend.location = "top_left"


df_diff_bottom = df.sort_values("weight_diff").head(10)
df_diff_bottom["weight_diff_min"] = df_diff_bottom.weight_diff
df_diff_top = df.sort_values("weight_diff", ascending = False).head(10).sort_values("weight_diff")
df_diff_top["weight_diff_max"] = df_diff_top.weight_diff

df_diff = pd.concat([df_diff_bottom, df_diff_top])

source_2 = ColumnDataSource(df_diff)

tooltips_2 = [
    ("State-Item", "@state_item_id"),
    ("Old Weight", "@weight_old{0.0000}"),
    ("New Weight", "@weight_new{0.0000}"),
    ("Weight Absolute Change", "@weight_diff{0.0000}")
]

max_diff = max(abs(np.nanmin(df_diff.weight_diff_min.values)), np.nanmax(df_diff.weight_diff_max.values))

v2 = figure(plot_width = 650, plot_height = 400, y_range = df_diff.state_item_id, x_range = Range1d(-max_diff * 1.1, max_diff * 1.1), tooltips = tooltips_2, title = "Top Absolute Weight Change for State-Item")
v2.hbar("state_item_id", right = "weight_diff_min", source = source_2, height = 0.75, color = "red", alpha = 0.6)
v2.hbar("state_item_id", right = "weight_diff_max", source = source_2, height = 0.75, color = "green", alpha = 0.6)

v2.xaxis.axis_label = "Absolute Weight Difference"
v2.yaxis.axis_label = "State-Item"


df_perc_bottom = df[df.weight_old != 0].sort_values("weight_perc").head(10)
df_perc_bottom["weight_perc_min"] = df_perc_bottom.weight_perc
df_perc_top = df[df.weight_old != 0].sort_values("weight_perc", ascending = False).head(10).sort_values("weight_perc")
df_perc_top["weight_perc_max"] = df_perc_top.weight_perc

df_perc = pd.concat([df_perc_bottom, df_perc_top])

source_3 = ColumnDataSource(df_perc)

tooltips_3 = [
    ("State-Item", "@state_item_id"),
    ("Old Weight", "@weight_old{0.000000}"),
    ("New Weight", "@weight_new{0.000000}"),
    ("Weight Percentage Change", "@weight_perc{0.00}%")
]

max_perc = max(abs(np.nanmin(df_perc.weight_perc_min.values)), np.nanmax(df_perc.weight_perc_max.values))

v3 = figure(plot_width = 650, plot_height = 400, y_range = df_perc.state_item_id, x_range = Range1d(-max_perc * 1.1, max_perc * 1.1), tooltips = tooltips_3, title = "Top Percentage Weight Change for State-Item")
v3.hbar("state_item_id", right = "weight_perc_min", source = source_3, height = 0.75, color = "red", alpha = 0.6)
v3.hbar("state_item_id", right = "weight_perc_max", source = source_3, height = 0.75, color = "green", alpha = 0.6)

v3.xaxis.axis_label = "Percentage Weight Difference"
v3.yaxis.axis_label = "State-Item"


df_top_old = df.sort_values("weight_old", ascending = False).head(20)

source_4 = ColumnDataSource(df_top_old)

tooltips_4 = [
    ("State-Item", "@state_item_id"),
    ("Old Weight", "@weight_old{0.000000}"),
    ("New Weight", "@weight_new{0.000000}")
]

v4 = figure(plot_width = 650, plot_height = 400, x_range = df_top_old.state_item_id, tooltips = tooltips_4, title = "Top State-Item with old weights")
v4.vbar(x = dodge("state_item_id", -0.15, range = v4.x_range), top = "weight_old", source = source_4, width = 0.2, color = "orange", alpha = 0.6, legend_label = "Old Weight")
v4.vbar(x = dodge("state_item_id", 0.15, range = v4.x_range), top = "weight_new", source = source_4, width = 0.2, color = "maroon", alpha = 0.6, legend_label = "New Weight")

v4.xaxis.major_label_orientation = pi / 2

v4.xaxis.axis_label = "State-Item"
v4.yaxis.axis_label = "Weight"


df_top_new = df.sort_values("weight_new", ascending = False).head(20)

source_5 = ColumnDataSource(df_top_new)

tooltips_5 = [
    ("State-Item", "@state_item_id"),
    ("Old Weight", "@weight_old{0.000000}"),
    ("New Weight", "@weight_new{0.000000}")
]

v5 = figure(plot_width = 650, plot_height = 400, x_range = df_top_new.state_item_id, tooltips = tooltips_5, title = "Top State-Item with new weights")
v5.vbar(x = dodge("state_item_id", -0.15, range = v5.x_range), top = "weight_old", source = source_5, width = 0.2, color = "orange", alpha = 0.6, legend_label = "Old Weight")
v5.vbar(x = dodge("state_item_id", 0.15, range = v5.x_range), top = "weight_new", source = source_5, width = 0.2, color = "maroon", alpha = 0.6, legend_label = "New Weight")

v5.xaxis.major_label_orientation = pi / 2

v5.xaxis.axis_label = "State-Item"
v5.yaxis.axis_label = "Weight"


show(column(v1, v2, v3, v4, v5))


# In[ ]:


df.to_csv("weights_state_item.csv", index = False)
df.head()


# ## 3. Store-Department
# Let's explore how different are the weights at store-department level.
# 

# In[ ]:


df_old = evaluator_1.lv10_weight.reset_index().rename(columns = {0: "weight_old"})
df_new = evaluator_2.lv10_weight.reset_index().rename(columns = {0: "weight_new"})

df = df_old.merge(df_new, on = ["store_id", "dept_id"])
df["store_dept_id"] = df.store_id + "-" + df.dept_id
df["weight_diff"] = df.weight_new - df.weight_old
df["weight_perc"] = (df.weight_new - df.weight_old) * 100 / df.weight_old

source = ColumnDataSource(df)

tooltips_1 = [
    ("Store-Department", "@store_dept_id"),
    ("Old Weight", "@weight_old{0.0000}"),
    ("New Weight", "@weight_new{0.0000}")
]

v1 = figure(plot_width = 600, plot_height = 300, tooltips = tooltips_1, title = "Old vs New Weights for Store-Department")
v1.circle("weight_old", "weight_new", source = source, size = 10, color = "steelblue", alpha = 0.6, legend_label = "Store-Department")

v1.xaxis.axis_label = "Old Weight"
v1.yaxis.axis_label = "New Weight"

v1.legend.location = "top_left"


df_diff_bottom = df.sort_values("weight_diff").head(10)
df_diff_bottom["weight_diff_min"] = df_diff_bottom.weight_diff
df_diff_top = df.sort_values("weight_diff", ascending = False).head(10).sort_values("weight_diff")
df_diff_top["weight_diff_max"] = df_diff_top.weight_diff

df_diff = pd.concat([df_diff_bottom, df_diff_top])

source_2 = ColumnDataSource(df_diff)

tooltips_2 = [
    ("Store-Department", "@store_dept_id"),
    ("Old Weight", "@weight_old{0.0000}"),
    ("New Weight", "@weight_new{0.0000}"),
    ("Weight Absolute Change", "@weight_diff{0.0000}")
]

max_diff = max(abs(np.nanmin(df_diff.weight_diff_min.values)), np.nanmax(df_diff.weight_diff_max.values))

v2 = figure(plot_width = 650, plot_height = 400, y_range = df_diff.store_dept_id, x_range = Range1d(-max_diff * 1.1, max_diff * 1.1), tooltips = tooltips_2, title = "Top Absolute Weight Change for Store-Department")
v2.hbar("store_dept_id", right = "weight_diff_min", source = source_2, height = 0.75, color = "red", alpha = 0.6)
v2.hbar("store_dept_id", right = "weight_diff_max", source = source_2, height = 0.75, color = "green", alpha = 0.6)

v2.xaxis.axis_label = "Absolute Weight Difference"
v2.yaxis.axis_label = "Store-Department"


df_perc_bottom = df[df.weight_old != 0].sort_values("weight_perc").head(10)
df_perc_bottom["weight_perc_min"] = df_perc_bottom.weight_perc
df_perc_top = df[df.weight_old != 0].sort_values("weight_perc", ascending = False).head(10).sort_values("weight_perc")
df_perc_top["weight_perc_max"] = df_perc_top.weight_perc

df_perc = pd.concat([df_perc_bottom, df_perc_top])

source_3 = ColumnDataSource(df_perc)

tooltips_3 = [
    ("Store-Department", "@store_dept_id"),
    ("Old Weight", "@weight_old{0.000000}"),
    ("New Weight", "@weight_new{0.000000}"),
    ("Weight Percentage Change", "@weight_perc{0.00}%")
]

max_perc = max(abs(np.nanmin(df_perc.weight_perc_min.values)), np.nanmax(df_perc.weight_perc_max.values))

v3 = figure(plot_width = 650, plot_height = 400, y_range = df_perc.store_dept_id, x_range = Range1d(-max_perc * 1.1, max_perc * 1.1), tooltips = tooltips_3, title = "Top Percentage Weight Change for Store-Department")
v3.hbar("store_dept_id", right = "weight_perc_min", source = source_3, height = 0.75, color = "red", alpha = 0.6)
v3.hbar("store_dept_id", right = "weight_perc_max", source = source_3, height = 0.75, color = "green", alpha = 0.6)

v3.xaxis.axis_label = "Percentage Weight Difference"
v3.yaxis.axis_label = "Store-Department"


show(column(v1, v2, v3))


# In[ ]:


df.to_csv("weights_store_dept.csv", index = False)
df.head()


# ## 4. Store-Category
# Let's explore how different are the weights at store-category level.
# 

# In[ ]:


df_old = evaluator_1.lv9_weight.reset_index().rename(columns = {0: "weight_old"})
df_new = evaluator_2.lv9_weight.reset_index().rename(columns = {0: "weight_new"})

df = df_old.merge(df_new, on = ["store_id", "cat_id"])
df["store_cat_id"] = df.store_id + "-" + df.cat_id
df["weight_diff"] = df.weight_new - df.weight_old
df["weight_perc"] = (df.weight_new - df.weight_old) * 100 / df.weight_old

source = ColumnDataSource(df)

tooltips_1 = [
    ("Store-Category", "@store_cat_id"),
    ("Old Weight", "@weight_old{0.0000}"),
    ("New Weight", "@weight_new{0.0000}")
]

v1 = figure(plot_width = 600, plot_height = 300, tooltips = tooltips_1, title = "Old vs New Weights for Store-Category")
v1.circle("weight_old", "weight_new", source = source, size = 10, color = "steelblue", alpha = 0.6, legend_label = "Store-Category")

v1.xaxis.axis_label = "Old Weight"
v1.yaxis.axis_label = "New Weight"

v1.legend.location = "top_left"


df_diff_bottom = df.sort_values("weight_diff").head(10)
df_diff_bottom["weight_diff_min"] = df_diff_bottom.weight_diff
df_diff_top = df.sort_values("weight_diff", ascending = False).head(10).sort_values("weight_diff")
df_diff_top["weight_diff_max"] = df_diff_top.weight_diff

df_diff = pd.concat([df_diff_bottom, df_diff_top])

source_2 = ColumnDataSource(df_diff)

tooltips_2 = [
    ("Store-Category", "@store_cat_id"),
    ("Old Weight", "@weight_old{0.0000}"),
    ("New Weight", "@weight_new{0.0000}"),
    ("Weight Absolute Change", "@weight_diff{0.0000}")
]

max_diff = max(abs(np.nanmin(df_diff.weight_diff_min.values)), np.nanmax(df_diff.weight_diff_max.values))

v2 = figure(plot_width = 650, plot_height = 400, y_range = df_diff.store_cat_id, x_range = Range1d(-max_diff * 1.1, max_diff * 1.1), tooltips = tooltips_2, title = "Top Absolute Weight Change for Store-Category")
v2.hbar("store_cat_id", right = "weight_diff_min", source = source_2, height = 0.75, color = "red", alpha = 0.6)
v2.hbar("store_cat_id", right = "weight_diff_max", source = source_2, height = 0.75, color = "green", alpha = 0.6)

v2.xaxis.axis_label = "Absolute Weight Difference"
v2.yaxis.axis_label = "Store-Category"


df_perc_bottom = df[df.weight_old != 0].sort_values("weight_perc").head(10)
df_perc_bottom["weight_perc_min"] = df_perc_bottom.weight_perc
df_perc_top = df[df.weight_old != 0].sort_values("weight_perc", ascending = False).head(10).sort_values("weight_perc")
df_perc_top["weight_perc_max"] = df_perc_top.weight_perc

df_perc = pd.concat([df_perc_bottom, df_perc_top])

source_3 = ColumnDataSource(df_perc)

tooltips_3 = [
    ("Store-Category", "@store_cat_id"),
    ("Old Weight", "@weight_old{0.000000}"),
    ("New Weight", "@weight_new{0.000000}"),
    ("Weight Percentage Change", "@weight_perc{0.00}%")
]

max_perc = max(abs(np.nanmin(df_perc.weight_perc_min.values)), np.nanmax(df_perc.weight_perc_max.values))

v3 = figure(plot_width = 650, plot_height = 400, y_range = df_perc.store_cat_id, x_range = Range1d(-max_perc * 1.1, max_perc * 1.1), tooltips = tooltips_3, title = "Top Percentage Weight Change for Store-Category")
v3.hbar("store_cat_id", right = "weight_perc_min", source = source_3, height = 0.75, color = "red", alpha = 0.6)
v3.hbar("store_cat_id", right = "weight_perc_max", source = source_3, height = 0.75, color = "green", alpha = 0.6)

v3.xaxis.axis_label = "Percentage Weight Difference"
v3.yaxis.axis_label = "Store-Category"


show(column(v1, v2, v3))


# In[ ]:


df.to_csv("weights_store_cat.csv", index = False)
df.head()


# ## 5. State-Department
# Let's explore how different are the weights at state-department level.

# In[ ]:


df_old = evaluator_1.lv8_weight.reset_index().rename(columns = {0: "weight_old"})
df_new = evaluator_2.lv8_weight.reset_index().rename(columns = {0: "weight_new"})

df = df_old.merge(df_new, on = ["state_id", "dept_id"])
df["state_dept_id"] = df.state_id + "-" + df.dept_id
df["weight_diff"] = df.weight_new - df.weight_old
df["weight_perc"] = (df.weight_new - df.weight_old) * 100 / df.weight_old

source = ColumnDataSource(df)

tooltips_1 = [
    ("State-Department", "@state_dept_id"),
    ("Old Weight", "@weight_old{0.0000}"),
    ("New Weight", "@weight_new{0.0000}")
]

v1 = figure(plot_width = 600, plot_height = 300, tooltips = tooltips_1, title = "Old vs New Weights for State-Department")
v1.circle("weight_old", "weight_new", source = source, size = 10, color = "steelblue", alpha = 0.6, legend_label = "State-Department")

v1.xaxis.axis_label = "Old Weight"
v1.yaxis.axis_label = "New Weight"

v1.legend.location = "top_left"


df_diff_bottom = df.sort_values("weight_diff").head(10)
df_diff_bottom["weight_diff_min"] = df_diff_bottom.weight_diff
df_diff_top = df.sort_values("weight_diff", ascending = False).head(10).sort_values("weight_diff")
df_diff_top["weight_diff_max"] = df_diff_top.weight_diff

df_diff = pd.concat([df_diff_bottom, df_diff_top])

source_2 = ColumnDataSource(df_diff)

tooltips_2 = [
    ("State-Department", "@state_dept_id"),
    ("Old Weight", "@weight_old{0.0000}"),
    ("New Weight", "@weight_new{0.0000}"),
    ("Weight Absolute Change", "@weight_diff{0.0000}")
]

max_diff = max(abs(np.nanmin(df_diff.weight_diff_min.values)), np.nanmax(df_diff.weight_diff_max.values))

v2 = figure(plot_width = 650, plot_height = 400, y_range = df_diff.state_dept_id, x_range = Range1d(-max_diff * 1.1, max_diff * 1.1), tooltips = tooltips_2, title = "Top Absolute Weight Change for State-Department")
v2.hbar("state_dept_id", right = "weight_diff_min", source = source_2, height = 0.75, color = "red", alpha = 0.6)
v2.hbar("state_dept_id", right = "weight_diff_max", source = source_2, height = 0.75, color = "green", alpha = 0.6)

v2.xaxis.axis_label = "Absolute Weight Difference"
v2.yaxis.axis_label = "State-Department"


df_perc_bottom = df[df.weight_old != 0].sort_values("weight_perc").head(10)
df_perc_bottom["weight_perc_min"] = df_perc_bottom.weight_perc
df_perc_top = df[df.weight_old != 0].sort_values("weight_perc", ascending = False).head(10).sort_values("weight_perc")
df_perc_top["weight_perc_max"] = df_perc_top.weight_perc

df_perc = pd.concat([df_perc_bottom, df_perc_top])

source_3 = ColumnDataSource(df_perc)

tooltips_3 = [
    ("State-Department", "@state_dept_id"),
    ("Old Weight", "@weight_old{0.000000}"),
    ("New Weight", "@weight_new{0.000000}"),
    ("Weight Percentage Change", "@weight_perc{0.00}%")
]

max_perc = max(abs(np.nanmin(df_perc.weight_perc_min.values)), np.nanmax(df_perc.weight_perc_max.values))

v3 = figure(plot_width = 650, plot_height = 400, y_range = df_perc.state_dept_id, x_range = Range1d(-max_perc * 1.1, max_perc * 1.1), tooltips = tooltips_3, title = "Top Percentage Weight Change for State-Department")
v3.hbar("state_dept_id", right = "weight_perc_min", source = source_3, height = 0.75, color = "red", alpha = 0.6)
v3.hbar("state_dept_id", right = "weight_perc_max", source = source_3, height = 0.75, color = "green", alpha = 0.6)

v3.xaxis.axis_label = "Percentage Weight Difference"
v3.yaxis.axis_label = "State-Department"


show(column(v1, v2, v3))


# In[ ]:


df.to_csv("weights_state_dept.csv", index = False)
df.head()


# ## 6. State-Category
# Let's explore how different are the weights at state-category level.

# In[ ]:


df_old = evaluator_1.lv7_weight.reset_index().rename(columns = {0: "weight_old"})
df_new = evaluator_2.lv7_weight.reset_index().rename(columns = {0: "weight_new"})

df = df_old.merge(df_new, on = ["state_id", "cat_id"])
df["state_cat_id"] = df.state_id + "-" + df.cat_id
df["weight_diff"] = df.weight_new - df.weight_old
df["weight_perc"] = (df.weight_new - df.weight_old) * 100 / df.weight_old

source = ColumnDataSource(df)

tooltips_1 = [
    ("State-Category", "@state_cat_id"),
    ("Old Weight", "@weight_old{0.0000}"),
    ("New Weight", "@weight_new{0.0000}")
]

v1 = figure(plot_width = 600, plot_height = 300, tooltips = tooltips_1, title = "Old vs New Weights for State-Category")
v1.circle("weight_old", "weight_new", source = source, size = 10, color = "steelblue", alpha = 0.6, legend_label = "State-Category")

v1.xaxis.axis_label = "Old Weight"
v1.yaxis.axis_label = "New Weight"

v1.legend.location = "top_left"


df_diff_bottom = df[df.weight_diff < 0].sort_values("weight_diff")
df_diff_bottom["weight_diff_min"] = df_diff_bottom.weight_diff
df_diff_top = df[df.weight_diff >= 0].sort_values("weight_diff")
df_diff_top["weight_diff_max"] = df_diff_top.weight_diff

df_diff = pd.concat([df_diff_bottom, df_diff_top])

source_2 = ColumnDataSource(df_diff)

tooltips_2 = [
    ("State-Category", "@state_cat_id"),
    ("Old Weight", "@weight_old{0.0000}"),
    ("New Weight", "@weight_new{0.0000}"),
    ("Weight Absolute Change", "@weight_diff{0.0000}")
]

max_diff = max(abs(np.nanmin(df_diff.weight_diff_min.values)), np.nanmax(df_diff.weight_diff_max.values))

v2 = figure(plot_width = 650, plot_height = 400, y_range = df_diff.state_cat_id, x_range = Range1d(-max_diff * 1.1, max_diff * 1.1), tooltips = tooltips_2, title = "Top Absolute Weight Change for State-Category")
v2.hbar("state_cat_id", right = "weight_diff_min", source = source_2, height = 0.75, color = "red", alpha = 0.6)
v2.hbar("state_cat_id", right = "weight_diff_max", source = source_2, height = 0.75, color = "green", alpha = 0.6)

v2.xaxis.axis_label = "Absolute Weight Difference"
v2.yaxis.axis_label = "State-Category"


df_perc_bottom = df[df.weight_perc < 0].sort_values("weight_perc")
df_perc_bottom["weight_perc_min"] = df_perc_bottom.weight_perc
df_perc_top = df[df.weight_perc >= 0].sort_values("weight_perc")
df_perc_top["weight_perc_max"] = df_perc_top.weight_perc

df_perc = pd.concat([df_perc_bottom, df_perc_top])

source_3 = ColumnDataSource(df_perc)

tooltips_3 = [
    ("State-Category", "@state_cat_id"),
    ("Old Weight", "@weight_old{0.000000}"),
    ("New Weight", "@weight_new{0.000000}"),
    ("Weight Percentage Change", "@weight_perc{0.00}%")
]

max_perc = max(abs(np.nanmin(df_perc.weight_perc_min.values)), np.nanmax(df_perc.weight_perc_max.values))

v3 = figure(plot_width = 650, plot_height = 400, y_range = df_perc.state_cat_id, x_range = Range1d(-max_perc * 1.1, max_perc * 1.1), tooltips = tooltips_3, title = "Top Percentage Weight Change for State-Category")
v3.hbar("state_cat_id", right = "weight_perc_min", source = source_3, height = 0.75, color = "red", alpha = 0.6)
v3.hbar("state_cat_id", right = "weight_perc_max", source = source_3, height = 0.75, color = "green", alpha = 0.6)

v3.xaxis.axis_label = "Percentage Weight Difference"
v3.yaxis.axis_label = "State-Category"


show(column(v1, v2, v3))


# In[ ]:


df.to_csv("weights_state_cat.csv", index = False)
df.head()


# ## 7. Item
# Let's explore how different are the weights at item level.

# In[ ]:


df_old = evaluator_1.lv6_weight.reset_index().rename(columns = {0: "weight_old"})
df_new = evaluator_2.lv6_weight.reset_index().rename(columns = {0: "weight_new"})

df = df_old.merge(df_new, on = "item_id")
df["weight_diff"] = df.weight_new - df.weight_old
df["weight_perc"] = (df.weight_new - df.weight_old) * 100 / df.weight_old

source = ColumnDataSource(df)

tooltips_1 = [
    ("Item", "@item_id"),
    ("Old Weight", "@weight_old{0.0000}"),
    ("New Weight", "@weight_new{0.0000}")
]

v1 = figure(plot_width = 600, plot_height = 300, tooltips = tooltips_1, title = "Old vs New Weights for Item")
v1.circle("weight_old", "weight_new", source = source, size = 10, color = "steelblue", alpha = 0.6, legend_label = "Item")

v1.xaxis.axis_label = "Old Weight"
v1.yaxis.axis_label = "New Weight"

v1.legend.location = "top_left"


df_diff_bottom = df.sort_values("weight_diff").head(10)
df_diff_bottom["weight_diff_min"] = df_diff_bottom.weight_diff
df_diff_top = df.sort_values("weight_diff", ascending = False).head(10).sort_values("weight_diff")
df_diff_top["weight_diff_max"] = df_diff_top.weight_diff

df_diff = pd.concat([df_diff_bottom, df_diff_top])

source_2 = ColumnDataSource(df_diff)

tooltips_2 = [
    ("Item", "@item_id"),
    ("Old Weight", "@weight_old{0.0000}"),
    ("New Weight", "@weight_new{0.0000}"),
    ("Weight Absolute Change", "@weight_diff{0.0000}")
]

max_diff = max(abs(np.nanmin(df_diff.weight_diff_min.values)), np.nanmax(df_diff.weight_diff_max.values))

v2 = figure(plot_width = 650, plot_height = 400, y_range = df_diff.item_id, x_range = Range1d(-max_diff * 1.1, max_diff * 1.1), tooltips = tooltips_2, title = "Top Absolute Weight Change for Item")
v2.hbar("item_id", right = "weight_diff_min", source = source_2, height = 0.75, color = "red", alpha = 0.6)
v2.hbar("item_id", right = "weight_diff_max", source = source_2, height = 0.75, color = "green", alpha = 0.6)

v2.xaxis.axis_label = "Absolute Weight Difference"
v2.yaxis.axis_label = "Item"


df_perc_bottom = df[df.weight_old != 0].sort_values("weight_perc").head(10)
df_perc_bottom["weight_perc_min"] = df_perc_bottom.weight_perc
df_perc_top = df[df.weight_old != 0].sort_values("weight_perc", ascending = False).head(10).sort_values("weight_perc")
df_perc_top["weight_perc_max"] = df_perc_top.weight_perc

df_perc = pd.concat([df_perc_bottom, df_perc_top])

source_3 = ColumnDataSource(df_perc)

tooltips_3 = [
    ("Item", "@item_id"),
    ("Old Weight", "@weight_old{0.000000}"),
    ("New Weight", "@weight_new{0.000000}"),
    ("Weight Percentage Change", "@weight_perc{0.00}%")
]

max_perc = max(abs(np.nanmin(df_perc.weight_perc_min.values)), np.nanmax(df_perc.weight_perc_max.values))

v3 = figure(plot_width = 650, plot_height = 400, y_range = df_perc.item_id, x_range = Range1d(-max_perc * 1.1, max_perc * 1.1), tooltips = tooltips_3, title = "Top Percentage Weight Change for Item")
v3.hbar("item_id", right = "weight_perc_min", source = source_3, height = 0.75, color = "red", alpha = 0.6)
v3.hbar("item_id", right = "weight_perc_max", source = source_3, height = 0.75, color = "green", alpha = 0.6)

v3.xaxis.axis_label = "Percentage Weight Difference"
v3.yaxis.axis_label = "Item"


show(column(v1, v2, v3))


# In[ ]:


df.to_csv("weights_item.csv", index = False)
df.head()


# ## 8. Store
# Let's explore how different are the weights at store level.

# In[ ]:


df_old = evaluator_1.lv5_weight.reset_index().rename(columns = {0: "weight_old"})
df_new = evaluator_2.lv5_weight.reset_index().rename(columns = {0: "weight_new"})

df = df_old.merge(df_new, on = "store_id")
df["weight_diff"] = df.weight_new - df.weight_old
df["weight_perc"] = (df.weight_new - df.weight_old) * 100 / df.weight_old

source = ColumnDataSource(df)

tooltips_1 = [
    ("Store", "@store_id"),
    ("Old Weight", "@weight_old{0.0000}"),
    ("New Weight", "@weight_new{0.0000}")
]

v1 = figure(plot_width = 600, plot_height = 300, tooltips = tooltips_1, title = "Old vs New Weights for Store")
v1.circle("weight_old", "weight_new", source = source, size = 10, color = "steelblue", alpha = 0.6, legend_label = "Store")

v1.xaxis.axis_label = "Old Weight"
v1.yaxis.axis_label = "New Weight"

v1.legend.location = "top_left"


df_diff_bottom = df[df.weight_diff < 0].sort_values("weight_diff")
df_diff_bottom["weight_diff_min"] = df_diff_bottom.weight_diff
df_diff_top = df[df.weight_diff >= 0].sort_values("weight_diff")
df_diff_top["weight_diff_max"] = df_diff_top.weight_diff

df_diff = pd.concat([df_diff_bottom, df_diff_top])

source_2 = ColumnDataSource(df_diff)

tooltips_2 = [
    ("Store", "@store_id"),
    ("Old Weight", "@weight_old{0.0000}"),
    ("New Weight", "@weight_new{0.0000}"),
    ("Weight Absolute Change", "@weight_diff{0.0000}")
]

max_diff = max(abs(np.nanmin(df_diff.weight_diff_min.values)), np.nanmax(df_diff.weight_diff_max.values))

v2 = figure(plot_width = 650, plot_height = 400, y_range = df_diff.store_id, x_range = Range1d(-max_diff * 1.1, max_diff * 1.1), tooltips = tooltips_2, title = "Top Absolute Weight Change for Store")
v2.hbar("store_id", right = "weight_diff_min", source = source_2, height = 0.75, color = "red", alpha = 0.6)
v2.hbar("store_id", right = "weight_diff_max", source = source_2, height = 0.75, color = "green", alpha = 0.6)

v2.xaxis.axis_label = "Absolute Weight Difference"
v2.yaxis.axis_label = "Store"


df_perc_bottom = df[df.weight_perc < 0].sort_values("weight_perc")
df_perc_bottom["weight_perc_min"] = df_perc_bottom.weight_perc
df_perc_top = df[df.weight_perc >= 0].sort_values("weight_perc")
df_perc_top["weight_perc_max"] = df_perc_top.weight_perc

df_perc = pd.concat([df_perc_bottom, df_perc_top])

source_3 = ColumnDataSource(df_perc)

tooltips_3 = [
    ("Store", "@store_id"),
    ("Old Weight", "@weight_old{0.000000}"),
    ("New Weight", "@weight_new{0.000000}"),
    ("Weight Percentage Change", "@weight_perc{0.00}%")
]

max_perc = max(abs(np.nanmin(df_perc.weight_perc_min.values)), np.nanmax(df_perc.weight_perc_max.values))

v3 = figure(plot_width = 650, plot_height = 400, y_range = df_perc.store_id, x_range = Range1d(-max_perc * 1.1, max_perc * 1.1), tooltips = tooltips_3, title = "Top Percentage Weight Change for Store")
v3.hbar("store_id", right = "weight_perc_min", source = source_3, height = 0.75, color = "red", alpha = 0.6)
v3.hbar("store_id", right = "weight_perc_max", source = source_3, height = 0.75, color = "green", alpha = 0.6)

v3.xaxis.axis_label = "Percentage Weight Difference"
v3.yaxis.axis_label = "Store"


show(column(v1, v2, v3))


# In[ ]:


df.to_csv("weights_store.csv", index = False)
df.head()


# ## 9. Department
# Let's explore how different are the weights at department level.

# In[ ]:


df_old = evaluator_1.lv4_weight.reset_index().rename(columns = {0: "weight_old"})
df_new = evaluator_2.lv4_weight.reset_index().rename(columns = {0: "weight_new"})

df = df_old.merge(df_new, on = "dept_id")
df["weight_diff"] = df.weight_new - df.weight_old
df["weight_perc"] = (df.weight_new - df.weight_old) * 100 / df.weight_old

source = ColumnDataSource(df)

tooltips_1 = [
    ("Department", "@dept_id"),
    ("Old Weight", "@weight_old{0.0000}"),
    ("New Weight", "@weight_new{0.0000}")
]

v1 = figure(plot_width = 600, plot_height = 300, tooltips = tooltips_1, title = "Old vs New Weights for Department")
v1.circle("weight_old", "weight_new", source = source, size = 10, color = "steelblue", alpha = 0.6, legend_label = "Department")

v1.xaxis.axis_label = "Old Weight"
v1.yaxis.axis_label = "New Weight"

v1.legend.location = "top_left"


df_diff_bottom = df[df.weight_diff < 0].sort_values("weight_diff")
df_diff_bottom["weight_diff_min"] = df_diff_bottom.weight_diff
df_diff_top = df[df.weight_diff >= 0].sort_values("weight_diff")
df_diff_top["weight_diff_max"] = df_diff_top.weight_diff

df_diff = pd.concat([df_diff_bottom, df_diff_top])

source_2 = ColumnDataSource(df_diff)

tooltips_2 = [
    ("Department", "@dept_id"),
    ("Old Weight", "@weight_old{0.0000}"),
    ("New Weight", "@weight_new{0.0000}"),
    ("Weight Absolute Change", "@weight_diff{0.0000}")
]

max_diff = max(abs(np.nanmin(df_diff.weight_diff_min.values)), np.nanmax(df_diff.weight_diff_max.values))

v2 = figure(plot_width = 650, plot_height = 400, y_range = df_diff.dept_id, x_range = Range1d(-max_diff * 1.1, max_diff * 1.1), tooltips = tooltips_2, title = "Top Absolute Weight Change for Department")
v2.hbar("dept_id", right = "weight_diff_min", source = source_2, height = 0.75, color = "red", alpha = 0.6)
v2.hbar("dept_id", right = "weight_diff_max", source = source_2, height = 0.75, color = "green", alpha = 0.6)

v2.xaxis.axis_label = "Absolute Weight Difference"
v2.yaxis.axis_label = "Department"


df_perc_bottom = df[df.weight_perc < 0].sort_values("weight_perc")
df_perc_bottom["weight_perc_min"] = df_perc_bottom.weight_perc
df_perc_top = df[df.weight_perc >= 0].sort_values("weight_perc")
df_perc_top["weight_perc_max"] = df_perc_top.weight_perc

df_perc = pd.concat([df_perc_bottom, df_perc_top])

source_3 = ColumnDataSource(df_perc)

tooltips_3 = [
    ("Department", "@dept_id"),
    ("Old Weight", "@weight_old{0.000000}"),
    ("New Weight", "@weight_new{0.000000}"),
    ("Weight Percentage Change", "@weight_perc{0.00}%")
]

max_perc = max(abs(np.nanmin(df_perc.weight_perc_min.values)), np.nanmax(df_perc.weight_perc_max.values))

v3 = figure(plot_width = 650, plot_height = 400, y_range = df_perc.dept_id, x_range = Range1d(-max_perc * 1.1, max_perc * 1.1), tooltips = tooltips_3, title = "Top Percentage Weight Change for Department")
v3.hbar("dept_id", right = "weight_perc_min", source = source_3, height = 0.75, color = "red", alpha = 0.6)
v3.hbar("dept_id", right = "weight_perc_max", source = source_3, height = 0.75, color = "green", alpha = 0.6)

v3.xaxis.axis_label = "Percentage Weight Difference"
v3.yaxis.axis_label = "Department"


show(column(v1, v2, v3))


# In[ ]:


df.to_csv("weights_dept.csv", index = False)
df.head()


# ## 10. State
# Let's explore how different are the weights at state level.

# In[ ]:


df_old = evaluator_1.lv3_weight.reset_index().rename(columns = {0: "weight_old"})
df_new = evaluator_2.lv3_weight.reset_index().rename(columns = {0: "weight_new"})

df = df_old.merge(df_new, on = "state_id")
df["weight_diff"] = df.weight_new - df.weight_old
df["weight_perc"] = (df.weight_new - df.weight_old) * 100 / df.weight_old

source = ColumnDataSource(df)

tooltips_1 = [
    ("State", "@state_id"),
    ("Old Weight", "@weight_old{0.0000}"),
    ("New Weight", "@weight_new{0.0000}")
]

v1 = figure(plot_width = 600, plot_height = 300, tooltips = tooltips_1, title = "Old vs New Weights for State")
v1.circle("weight_old", "weight_new", source = source, size = 10, color = "steelblue", alpha = 0.6, legend_label = "State")

v1.xaxis.axis_label = "Old Weight"
v1.yaxis.axis_label = "New Weight"

v1.legend.location = "top_left"


df_diff_bottom = df[df.weight_diff < 0].sort_values("weight_diff")
df_diff_bottom["weight_diff_min"] = df_diff_bottom.weight_diff
df_diff_top = df[df.weight_diff >= 0].sort_values("weight_diff")
df_diff_top["weight_diff_max"] = df_diff_top.weight_diff

df_diff = pd.concat([df_diff_bottom, df_diff_top])

source_2 = ColumnDataSource(df_diff)

tooltips_2 = [
    ("State", "@state_id"),
    ("Old Weight", "@weight_old{0.0000}"),
    ("New Weight", "@weight_new{0.0000}"),
    ("Weight Absolute Change", "@weight_diff{0.0000}")
]

max_diff = max(abs(np.nanmin(df_diff.weight_diff_min.values)), np.nanmax(df_diff.weight_diff_max.values))

v2 = figure(plot_width = 650, plot_height = 400, y_range = df_diff.state_id, x_range = Range1d(-max_diff * 1.1, max_diff * 1.1), tooltips = tooltips_2, title = "Top Absolute Weight Change for State")
v2.hbar("state_id", right = "weight_diff_min", source = source_2, height = 0.75, color = "red", alpha = 0.6)
v2.hbar("state_id", right = "weight_diff_max", source = source_2, height = 0.75, color = "green", alpha = 0.6)

v2.xaxis.axis_label = "Absolute Weight Difference"
v2.yaxis.axis_label = "State"


df_perc_bottom = df[df.weight_perc < 0].sort_values("weight_perc")
df_perc_bottom["weight_perc_min"] = df_perc_bottom.weight_perc
df_perc_top = df[df.weight_perc >= 0].sort_values("weight_perc")
df_perc_top["weight_perc_max"] = df_perc_top.weight_perc

df_perc = pd.concat([df_perc_bottom, df_perc_top])

source_3 = ColumnDataSource(df_perc)

tooltips_3 = [
    ("State", "@state_id"),
    ("Old Weight", "@weight_old{0.000000}"),
    ("New Weight", "@weight_new{0.000000}"),
    ("Weight Percentage Change", "@weight_perc{0.00}%")
]

max_perc = max(abs(np.nanmin(df_perc.weight_perc_min.values)), np.nanmax(df_perc.weight_perc_max.values))

v3 = figure(plot_width = 650, plot_height = 400, y_range = df_perc.state_id, x_range = Range1d(-max_perc * 1.1, max_perc * 1.1), tooltips = tooltips_3, title = "Top Percentage Weight Change for State")
v3.hbar("state_id", right = "weight_perc_min", source = source_3, height = 0.75, color = "red", alpha = 0.6)
v3.hbar("state_id", right = "weight_perc_max", source = source_3, height = 0.75, color = "green", alpha = 0.6)

v3.xaxis.axis_label = "Percentage Weight Difference"
v3.yaxis.axis_label = "State"


show(column(v1, v2, v3))


# In[ ]:


df.to_csv("weights_state.csv", index = False)
df.head()


# ## 11. Category
# Let's explore how different are the weights at category level.

# In[ ]:


df_old = evaluator_1.lv2_weight.reset_index().rename(columns = {0: "weight_old"})
df_new = evaluator_2.lv2_weight.reset_index().rename(columns = {0: "weight_new"})

df = df_old.merge(df_new, on = "cat_id")
df["weight_diff"] = df.weight_new - df.weight_old
df["weight_perc"] = (df.weight_new - df.weight_old) * 100 / df.weight_old

source = ColumnDataSource(df)

tooltips_1 = [
    ("Category", "@cat_id"),
    ("Old Weight", "@weight_old{0.0000}"),
    ("New Weight", "@weight_new{0.0000}")
]

v1 = figure(plot_width = 600, plot_height = 300, tooltips = tooltips_1, title = "Old vs New Weights for Category")
v1.circle("weight_old", "weight_new", source = source, size = 10, color = "steelblue", alpha = 0.6, legend_label = "Category")

v1.xaxis.axis_label = "Old Weight"
v1.yaxis.axis_label = "New Weight"

v1.legend.location = "top_left"


df_diff_bottom = df[df.weight_diff < 0].sort_values("weight_diff")
df_diff_bottom["weight_diff_min"] = df_diff_bottom.weight_diff
df_diff_top = df[df.weight_diff >= 0].sort_values("weight_diff")
df_diff_top["weight_diff_max"] = df_diff_top.weight_diff

df_diff = pd.concat([df_diff_bottom, df_diff_top])

source_2 = ColumnDataSource(df_diff)

tooltips_2 = [
    ("Category", "@cat_id"),
    ("Old Weight", "@weight_old{0.0000}"),
    ("New Weight", "@weight_new{0.0000}"),
    ("Weight Absolute Change", "@weight_diff{0.0000}")
]

max_diff = max(abs(np.nanmin(df_diff.weight_diff_min.values)), np.nanmax(df_diff.weight_diff_max.values))

v2 = figure(plot_width = 650, plot_height = 400, y_range = df_diff.cat_id, x_range = Range1d(-max_diff * 1.1, max_diff * 1.1), tooltips = tooltips_2, title = "Top Absolute Weight Change for Category")
v2.hbar("cat_id", right = "weight_diff_min", source = source_2, height = 0.75, color = "red", alpha = 0.6)
v2.hbar("cat_id", right = "weight_diff_max", source = source_2, height = 0.75, color = "green", alpha = 0.6)

v2.xaxis.axis_label = "Absolute Weight Difference"
v2.yaxis.axis_label = "Category"


df_perc_bottom = df[df.weight_perc < 0].sort_values("weight_perc")
df_perc_bottom["weight_perc_min"] = df_perc_bottom.weight_perc
df_perc_top = df[df.weight_perc >= 0].sort_values("weight_perc")
df_perc_top["weight_perc_max"] = df_perc_top.weight_perc

df_perc = pd.concat([df_perc_bottom, df_perc_top])

source_3 = ColumnDataSource(df_perc)

tooltips_3 = [
    ("Category", "@cat_id"),
    ("Old Weight", "@weight_old{0.000000}"),
    ("New Weight", "@weight_new{0.000000}"),
    ("Weight Percentage Change", "@weight_perc{0.00}%")
]

max_perc = max(abs(np.nanmin(df_perc.weight_perc_min.values)), np.nanmax(df_perc.weight_perc_max.values))

v3 = figure(plot_width = 650, plot_height = 400, y_range = df_perc.cat_id, x_range = Range1d(-max_perc * 1.1, max_perc * 1.1), tooltips = tooltips_3, title = "Top Percentage Weight Change for Category")
v3.hbar("cat_id", right = "weight_perc_min", source = source_3, height = 0.75, color = "red", alpha = 0.6)
v3.hbar("cat_id", right = "weight_perc_max", source = source_3, height = 0.75, color = "green", alpha = 0.6)

v3.xaxis.axis_label = "Percentage Weight Difference"
v3.yaxis.axis_label = "Category"


show(column(v1, v2, v3))


# In[ ]:


df.to_csv("weights_cat.csv", index = False)
df.head()


# ## Avoiding the LB pitfall
# There are some valuable insights to take back from the changes in weights, especially at the more granular levels. Feel free to explore and inculcate some of them into your models to ensure the model performs well on the private LB.
# 
