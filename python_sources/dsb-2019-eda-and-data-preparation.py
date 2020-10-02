#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import math
import os
import csv
import json
import gc
from functools import partial
from pathlib import Path
from multiprocessing import Pool

import pandas as pd
import numpy as np
from tqdm.notebook import tqdm
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from tqdm import tqdm_notebook
from pandas_summary import DataFrameSummary
import seaborn as sns
from pandas.io.json import json_normalize
from hashlib import md5

tqdm.pandas()


# In[ ]:


def compare_cat_frequency(df1, df2, column, df1_name='train', df2_name='test', top_n=25, figsize=(20, 16)):
    fig, (ax, ax2) = plt.subplots(ncols=2, sharey=True, figsize=figsize)
    
    ax.yaxis.tick_right()

    df1_values = df1[column].value_counts()[:25]
    df1_values.plot.barh(ax=ax, title=f'{df1_name} {column} distribution')

    test_df[column].value_counts()[df1_values.keys()].plot.barh(ax=ax2, title=f'{df2_name} {column} distribution')

    plt.tight_layout()
    plt.show()


def plot_grid_hist(df, columns, test_df=None, ncols=3, figsize=(20, 12)):
    sns.set_palette("Spectral_r")

    fig, axes = plt.subplots(nrows=len(columns) // ncols, ncols=ncols, figsize=figsize)

    count = 0
    for ax_row in axes:
        for ax in ax_row:
            count += 1
            try:
                key = columns[count]
                print(key)
                ax.hist(df[key], label='Train', edgecolor='black', linewidth=0, bins=100, histtype='stepfilled', density=True)
                if test_df is not None:
                    ax.hist(test_df[key], label='Test', bins=100, linewidth=1, linestyle='dashed', alpha = 0.5, histtype='stepfilled', density=True)
                    ax.legend()
                ax.set_title(f'Distribution of {key}')
            except IndexError:
                continue


# # DSB 2019: EDA and data preparation
# 
# The goal of this notebook is to explore the dataset from the 2019 Data Science Bowl Kaggle competition and prepare it ready it for a sequence model.
# 
# ## Competition overview
# 
# The dataset is provided by the PBS Measure Up app and is a timeseries of a user's activity across game sessions. The goal is to predict how they will do at an evaluation based using the features.
# 
# It appears the organisers are looking to use the model to improve the games they develop and maybe give a user the best possible experience based on their performance.

# In[ ]:


DATA_PATH = Path('/kaggle/input/data-science-bowl-2019/')
OUTPUT_PATH = Path('/kaggle/working/')


# ## Loading data

# Based on some earlier analysis I've done, I'm setting the datatypes of a few fields to minimise memory consumption.

# In[ ]:


TRAIN_DTYPES = {
    'event_count': np.uint16,
    'event_code': np.uint16,
    'game_type': np.uint32
}


# In[ ]:


get_ipython().run_cell_magic('time', '', "\ntrain_df = pd.read_csv(\n    DATA_PATH/'train.csv', parse_dates=['timestamp'],\n    dtype=TRAIN_DTYPES\n).sort_values('timestamp')")


# In[ ]:


get_ipython().run_cell_magic('time', '', "\ntest_df = pd.read_csv(\n    DATA_PATH/'test.csv',\n    parse_dates=['timestamp'],\n    dtype=TRAIN_DTYPES\n).sort_values('timestamp')")


# ## Column overview
# 
# I'll use `DataFrameSummary` from [pandas-summary](https://github.com/mouradmourafiq/pandas-summary) to get a high-level overview of the data we're working with.

# In[ ]:


DataFrameSummary(train_df).columns_stats


# In[ ]:


DataFrameSummary(test_df).columns_stats


# So, it appears that there are no missing values. However, the `event_data` is a JSON field, so I'm sure when that's expanded it will be a different story.

# ### timestamp distribution
# 
# Since time seems to be a key feature in this dataset, it will be interesting to see how the train and test set differ in their time distribution.**

# In[ ]:


sns.set_palette("Spectral_r")

plt.figure(figsize=(14, 4))
plt.title("timestamp frequency")

plt.hist(train_df.timestamp, edgecolor='black', linewidth=1.2, label='Train', histtype='stepfilled', density=True)
plt.hist(test_df.timestamp, edgecolor='black', linewidth=1.2, linestyle='dashed', label='Test', alpha = 0.5, histtype='stepfilled', density=True)

plt.xticks(rotation=70)
plt.legend()
plt.show()


# So the train and test are roughly the same time period across a ~3 month window.

# ### event_time distribution

# In[ ]:


import matplotlib.patheffects as pe

sns.set_palette("RdBu_r")

plt.figure(figsize=(14, 4))
plt.title("event_time frequency")
plt.hist(train_df.game_time, label='Train', bins=100, path_effects=[pe.Stroke(linewidth=2, foreground='black'), pe.Normal()])
plt.hist(test_df.game_time, label='Test', bins=100, path_effects=[pe.Stroke(linewidth=2, foreground='black'), pe.Normal()])

plt.legend()
plt.show()


# Hard to see from this plot but it appears the event_time are mostly very short with some very long times.

# ### title distribution

# In[ ]:


compare_cat_frequency(train_df, test_df, column='title')


# So, titles played is quite similar across train and test set, albeit the train set has a lot more examples.

# ### Distribution of other categorical fields

# In[ ]:


for column in ['event_code', 'event_id', 'world', 'type']:
    compare_cat_frequency(train_df, test_df, column=column)


# ### Set categorical columns

# To save space, I'll use the Pandas categorical type for categorical columns. I'll concat the train and test datasets together to ensure the types have the full range of values.

# In[ ]:


all_df = pd.concat([train_df, test_df], axis=0)

WORLD_VALS = all_df.world.unique()
TITLE_VALS = all_df.title.unique()
TYPE_VALS = all_df.type.unique()
EVENT_CODE = all_df.event_code.unique()
EVENT_ID = all_df.event_id.unique()


# In[ ]:


def set_categorical(df):
    df.world = pd.Categorical(df.world, categories=WORLD_VALS)
    df.title = pd.Categorical(df.title, categories=TITLE_VALS)
    df.type = pd.Categorical(df.type, categories=TYPE_VALS)
    df.event_code = pd.Categorical(df.event_code, categories=EVENT_CODE)
    df.event_id = pd.Categorical(df.event_id, categories=EVENT_ID)
    return df


# In[ ]:


train_df = set_categorical(train_df)
test_df = set_categorical(test_df)


# In[ ]:


train_df.dtypes


# In[ ]:


del all_df
gc.collect()


# ### Event data

# ### Flattening

# The event data is in represented as JSON objects for each row. The first step to performing analysis on it will be to flatten it into a sparse matrix.
# 
# Since it's a lot of information, I'm going to start by analysis a sample 100 thousands rows. Since the event ordering is essentially, I'm going to sample by taking a slice of the first 100k rows, not randomly sampling.

# In[ ]:


def flatten_json(nested_json):
    nested_json = json.loads(nested_json)

    out = {}

    def _flatten(x, name=''):
        if type(x) is dict:
            for a in x: _flatten(x[a], name + a + '_')
        elif type(x) is list:
            i = 0
            for a in x:
                _flatten(a, name + str(i) + '_')
                i += 1
        else:
            out[name[:-1]] = x

    _flatten(nested_json)

    return out


# In[ ]:


train_df_sample = train_df.sample(n=100_000)


# In[ ]:


get_ipython().run_cell_magic('time', '', '\ntrain_event_data_norm_sample = json_normalize(train_df_sample.event_data.progress_apply(flatten_json))')


# ## Missing values

# Show percentage of missing values across each field.

# In[ ]:


train_event_na_perc = (
    train_event_data_norm_sample.isna().sum().sort_values() /
     len(train_event_data_norm_sample))


# In[ ]:


train_event_na_perc


# I'm going to limit the columns to just the ones that are in at least 5% of rows. There's a lot of very rare fields and I feel like they're going to need some feature engineering. That seems like something to be working on toward the end of the competition.
# 
# Since `event_count`, `event_code` and `game_time` are already provided, that seems like a column that can be excluded too.

# In[ ]:


columns_to_include = train_event_na_perc[train_event_na_perc <= 0.95].keys()
columns_to_include = [c for c in columns_to_include if c not in ('event_count', 'event_code', 'game_time')]


# In[ ]:


DataFrameSummary(train_event_data_norm_sample[columns_to_include]).columns_stats


# In[ ]:


ed_summary = DataFrameSummary(train_event_data_norm_sample[columns_to_include]).summary(); ed_summary


# Let's look at the distribution of numeric types.

# In[ ]:


numeric_cols = ed_summary.T[ed_summary.T.types == 'numeric'].index
path_effects = [pe.Stroke(linewidth=1, foreground='black'), pe.Normal()]


# In[ ]:


plot_grid_hist(train_event_data_norm_sample, columns=list(numeric_cols)[:16])


# Next up, I want to load the whole dataset joined to the `event_data` DataFrame, but I intend to convert some of the large columns to a lookup using the hash of the data.

# In[ ]:


columns_to_include


# In[ ]:


DESCRIPTIONS = []
SOURCE_CATS = []
IDENTIFIER_CATS = set([])
MEDIA_TYPE_CATS = set([])
COORD_STAGE_HEIGHT = set([])
COORD_STAGE_WIDTH = set([])


def do_event_data(event_data: dict, output_file: str):
    csv_file = open(f'{OUTPUT_PATH}/{output_file}', 'w')
    csv_writer = csv.writer(csv_file, delimiter=',')

    for data in tqdm(event_data.values, total=len(event_data)):
        row_flattened = flatten_json(data)

        # map description to its hash.
        desc = row_flattened.get('description')
        if desc:
            if desc not in DESCRIPTIONS:
                DESCRIPTIONS.append(desc)
            row_flattened['description'] = DESCRIPTIONS.index(desc)
            
        source = row_flattened.get('source')
        if source:
            source = str(source).lower()
            if source not in SOURCE_CATS:
                SOURCE_CATS.append(source)
            row_flattened['source'] = SOURCE_CATS.index(source)
            
        for col, l in [
            ('identifier', IDENTIFIER_CATS),
            ('media_type', MEDIA_TYPE_CATS),
            ('coordinates_stage_height', COORD_STAGE_HEIGHT),
            ('coordinates_stage_width', COORD_STAGE_WIDTH)
        ]:
            value = row_flattened.get(col)
            if value: l.add(value)

        csv_writer.writerow(row_flattened.get(k, None) for k in columns_to_include)


# In[ ]:


do_event_data(train_df.event_data, 'train_event_data.csv')


# In[ ]:


do_event_data(test_df.event_data, 'test_event_data.csv')


# In[ ]:


dtypes = dict(
    source=pd.CategoricalDtype(list(range(len(SOURCE_CATS)))),
    media_type=pd.CategoricalDtype(MEDIA_TYPE_CATS),
    identifier=pd.CategoricalDtype(IDENTIFIER_CATS),
    description=pd.CategoricalDtype(list(range(len(DESCRIPTIONS)))),
    coordinates_stage_height=pd.CategoricalDtype(list(range(len(COORD_STAGE_HEIGHT)))),
    coordinates_stage_width=pd.CategoricalDtype(list(range(len(COORD_STAGE_WIDTH))))
)


# In[ ]:


get_ipython().run_cell_magic('time', '', "\ntrain_event_data = pd.read_csv(\n    OUTPUT_PATH/'train_event_data.csv', names=columns_to_include, header=None, dtype=dtypes)")


# In[ ]:


get_ipython().run_cell_magic('time', '', "\ntest_event_data = pd.read_csv(OUTPUT_PATH/'test_event_data.csv', names=columns_to_include, header=None, dtype=dtypes)")


# In[ ]:


numeric_cols_revised = ['round', 'coordinates_x', 'coordinates_y', 'duration', 'total_duration', 'level', 'size', 'weight']


# In[ ]:


plot_grid_hist(
    train_event_data.sample(n=500_000),
    test_df=test_event_data,
    columns=list(numeric_cols_revised),
    ncols=2, figsize=(16, 20))


# In[ ]:


def join_event_data(df, df_event):
    return pd.concat([
        df[[i for i in df.columns if i != 'event_data']].reset_index(drop=True),
        df_event.reset_index(drop=True)], axis=1).reset_index(drop=True)


# In[ ]:


train_df_comb = join_event_data(train_df, train_event_data)
test_df_comb = join_event_data(test_df, test_event_data)


# The last thing I want to do is sort by `installation_id` and `timestamp` which should make convert the data into a sequence nice and easy in future.

# In[ ]:


train_df_comb = train_df_comb.sort_values(['installation_id', 'timestamp']).reset_index(drop=True)
test_df_comb = test_df_comb.sort_values(['installation_id', 'timestamp']).reset_index(drop=True)


# In[ ]:


train_df_comb.to_feather(OUTPUT_PATH/'train.fth')
test_df_comb.to_feather(OUTPUT_PATH/'test.fth')


# In[ ]:


del train_df_comb
del test_df_comb
del train_df
del test_df
del train_event_data
del test_event_data
del train_event_data_norm_sample
del train_df_sample

gc.collect()


# ## Add labels

# I'll add a column for the assessment titles and use it to understand how many assessments happen per `installation_id`.
# 
# Based on the following bit of information: "Assessment attempts are captured in event_code 4100 for all assessments except for Bird Measurer, which uses event_code 4110."

# In[ ]:


train_df_comb = pd.read_feather(OUTPUT_PATH/'train.fth')
test_df_comb = pd.read_feather(OUTPUT_PATH/'test.fth')


# In[ ]:


train_labels = pd.read_csv(DATA_PATH/'train_labels.csv')


# So, it sounds like we want to calculate `num_correct`, `num_incorect`, `accuracy` and use that to set the `accuracy_group`.
# 
# I'll start by adding an assessment column to determine if the activity is an assessment.

# I'll start by adding a `attempt` column.

# In[ ]:


# thanks to https://www.kaggle.com/artgor/oop-approach-to-fe-and-models

def set_attempt_label(df):
    df['attempt'] = 0
    df.loc[
        (df['title'] == 'Bird Measurer (Assessment)') &
        (df['event_code'] == 4110), 'attempt'] = 1

    df.loc[
        (df['title'] != 'Bird Measurer (Assessment)') &
        (df['event_code'] == 4100) & (df['type'] == 'Assessment'), 'attempt'] = 1

    return df
    
train_df_comb = set_attempt_label(train_df_comb)
test_df_comb = set_attempt_label(test_df_comb)


# Let's group by game session and installation id for assessments

# In[ ]:


def get_accuracy_group(row):
    if row.correct == 0:
        return 0
    
    if row.attempt > 2:
        return 1
    
    if row.attempt == 2:
        return 2
    
    if row.attempt == 1:
        return 3


def get_labels(df):
    num_correct = df[df.attempt == 1].groupby(['game_session', 'installation_id']).correct.sum().astype(int)
    num_attempts = df[df.attempt == 1].groupby(['game_session', 'installation_id']).attempt.sum().astype(int)
    titles = df[df.attempt == 1].groupby(['game_session', 'installation_id']).title.agg(lambda x: x.iloc[0])
    labels_joined = num_correct.to_frame().join(num_attempts).join(titles).reset_index()
    labels_joined['accuracy_group'] = labels_joined.apply(get_accuracy_group, axis=1)
    return labels_joined


# In[ ]:


train_labels_joined = get_labels(train_df_comb)
test_labels_joined = get_labels(test_df_comb)


# In[ ]:


train_labels_joined.accuracy_group.value_counts().plot.bar(title='Train labels dist')


# In[ ]:


test_labels_joined.accuracy_group.value_counts().plot.bar(title='Test labels dist')


# ### Storing sequence start and end positions

# In order to build a disk space efficient sequence model, I'm going to store the start and end position of each sequence, based on the assumption that the train and test data is sorted by `installation_id` and `timestamp`.

# In[ ]:


def _do_installation_id(inp, df):
    (installation_id, row) = inp

    game_sessions = row.game_session.unique()

    filtered_rows = df[df.installation_id == installation_id]

    start_idx = filtered_rows.head(1).index[0]

    output = []
    for game_session in game_sessions:
        assessment_row = filtered_rows[(filtered_rows.game_session == game_session) & (filtered_rows.event_code == 2000)]
        output.append((installation_id, game_session, start_idx, assessment_row.index[0]))

    return output


def add_start_and_end_pos(labels, df):
    labels_grouped = labels.groupby('installation_id')
    
    labels['start_idx'] = -1
    labels['end_idx'] = -1
    
    for row in tqdm(labels_grouped, total=len(labels_grouped)):
        results = _do_installation_id(row, df=df)

        for (installation_id, game_session, start_pos, end_pos) in results:
            filt = (labels.installation_id == installation_id) & (labels.game_session == game_session)

            labels.loc[filt, 'start_idx'] = start_pos
            labels.loc[filt, 'end_idx'] = end_pos


# In[ ]:


add_start_and_end_pos(train_labels_joined, train_df_comb)
add_start_and_end_pos(test_labels_joined, test_df_comb)


# In[ ]:


train_labels_joined.to_feather(OUTPUT_PATH/'train_labels.fth')
test_labels_joined.to_feather(OUTPUT_PATH/'test_labels.fth')

