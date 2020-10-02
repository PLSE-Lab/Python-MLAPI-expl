#!/usr/bin/env python
# coding: utf-8

# [Lesson Video Link](https://course.fast.ai/videos/?lesson=6)
# 
# [Lesson resources and updates](https://forums.fast.ai/t/lesson-6-official-resources-and-updates/31441)
# 
# [Lesson chat](https://forums.fast.ai/t/lesson-6-in-class-discussion/31440)
# 
# [Further discussion thread](https://forums.fast.ai/t/lesson-6-advanced-discussion/31442)
# 
# Note: This is a mirror of the FastAI Lesson 6 Nb. 
# Please thank the amazing team behind fast.ai for creating these, I've merely created a mirror of the same here
# For complete info on the course, visit course.fast.ai

# In[ ]:


import os
import re
import pandas as pd
import numpy as np
from pathlib import Path


# In[ ]:


get_ipython().run_line_magic('reload_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')


# In[ ]:


from fastai import *
#from fastai.basics import *


# # Rossmann

# ## Data preparation / Feature engineering

# In addition to the provided data, we will be using external datasets put together by participants in the Kaggle competition. You can download all of them [here](http://files.fast.ai/part2/lesson14/rossmann.tgz). Then you shold untar them in the dirctory to which `PATH` is pointing below.
# 
# For completeness, the implementation used to put them together is included below.

# In[ ]:


PATH=Path('../input/')
table_names = ['train', 'store', 'store_states', 'state_names', 'googletrend', 'weather', 'test']
tables = [pd.read_csv(PATH/f'{fname}.csv', low_memory=False) for fname in table_names]
train, store, store_states, state_names, googletrend, weather, test = tables
len(train),len(test)


# We turn state Holidays to booleans, to make them more convenient for modeling. We can do calculations on pandas fields using notation very similar (often identical) to numpy.

# In[ ]:


train.StateHoliday = train.StateHoliday!='0'
test.StateHoliday = test.StateHoliday!='0'


# `join_df` is a function for joining tables on specific fields. By default, we'll be doing a left outer join of `right` on the `left` argument using the given fields for each table.
# 
# Pandas does joins using the `merge` method. The `suffixes` argument describes the naming convention for duplicate fields. We've elected to leave the duplicate field names on the left untouched, and append a "\_y" to those on the right.

# In[ ]:


def join_df(left, right, left_on, right_on=None, suffix='_y'):
    if right_on is None: right_on = left_on
    return left.merge(right, how='left', left_on=left_on, right_on=right_on, 
                      suffixes=("", suffix))


# Join weather/state names.

# In[ ]:


weather = join_df(weather, state_names, "file", "StateName")


# In pandas you can add new columns to a dataframe by simply defining it. We'll do this for googletrends by extracting dates and state names from the given data and adding those columns.
# 
# We're also going to replace all instances of state name 'NI' to match the usage in the rest of the data: 'HB,NI'. This is a good opportunity to highlight pandas indexing. We can use `.loc[rows, cols]` to select a list of rows and a list of columns from the dataframe. In this case, we're selecting rows w/ statename 'NI' by using a boolean list `googletrend.State=='NI'` and selecting "State".

# In[ ]:


googletrend['Date'] = googletrend.week.str.split(' - ', expand=True)[0]
googletrend['State'] = googletrend.file.str.split('_', expand=True)[2]
googletrend.loc[googletrend.State=='NI', "State"] = 'HB,NI'


# The following extracts particular date fields from a complete datetime for the purpose of constructing categoricals.
# 
# You should *always* consider this feature extraction step when working with date-time. Without expanding your date-time into these additional fields, you can't capture any trend/cyclical behavior as a function of time at any of these granularities. We'll add to every table with a date field.

# In[ ]:


def add_datepart(df, fldname, drop=True, time=False):
    "Helper function that adds columns relevant to a date."
    fld = df[fldname]
    fld_dtype = fld.dtype
    if isinstance(fld_dtype, pd.core.dtypes.dtypes.DatetimeTZDtype):
        fld_dtype = np.datetime64

    if not np.issubdtype(fld_dtype, np.datetime64):
        df[fldname] = fld = pd.to_datetime(fld, infer_datetime_format=True)
    targ_pre = re.sub('[Dd]ate$', '', fldname)
    attr = ['Year', 'Month', 'Week', 'Day', 'Dayofweek', 'Dayofyear',
            'Is_month_end', 'Is_month_start', 'Is_quarter_end', 'Is_quarter_start', 'Is_year_end', 'Is_year_start']
    if time: attr = attr + ['Hour', 'Minute', 'Second']
    for n in attr: df[targ_pre + n] = getattr(fld.dt, n.lower())
    df[targ_pre + 'Elapsed'] = fld.astype(np.int64) // 10 ** 9
    if drop: df.drop(fldname, axis=1, inplace=True)


# In[ ]:


add_datepart(weather, "Date", drop=False)
add_datepart(googletrend, "Date", drop=False)
add_datepart(train, "Date", drop=False)
add_datepart(test, "Date", drop=False)


# The Google trends data has a special category for the whole of the Germany - we'll pull that out so we can use it explicitly.

# In[ ]:


trend_de = googletrend[googletrend.file == 'Rossmann_DE']


# Now we can outer join all of our data into a single dataframe. Recall that in outer joins everytime a value in the joining field on the left table does not have a corresponding value on the right table, the corresponding row in the new table has Null values for all right table fields. One way to check that all records are consistent and complete is to check for Null values post-join, as we do here.
# 
# *Aside*: Why note just do an inner join?
# If you are assuming that all records are complete and match on the field you desire, an inner join will do the same thing as an outer join. However, in the event you are wrong or a mistake is made, an outer join followed by a null-check will catch it. (Comparing before/after # of rows for inner join is equivalent, but requires keeping track of before/after row #'s. Outer join is easier.)

# In[ ]:


store = join_df(store, store_states, "Store")
len(store[store.State.isnull()])


# In[ ]:


joined = join_df(train, store, "Store")
joined_test = join_df(test, store, "Store")
len(joined[joined.StoreType.isnull()]),len(joined_test[joined_test.StoreType.isnull()])


# In[ ]:


joined = join_df(joined, googletrend, ["State","Year", "Week"])
joined_test = join_df(joined_test, googletrend, ["State","Year", "Week"])
len(joined[joined.trend.isnull()]),len(joined_test[joined_test.trend.isnull()])


# In[ ]:


joined = joined.merge(trend_de, 'left', ["Year", "Week"], suffixes=('', '_DE'))
joined_test = joined_test.merge(trend_de, 'left', ["Year", "Week"], suffixes=('', '_DE'))
len(joined[joined.trend_DE.isnull()]),len(joined_test[joined_test.trend_DE.isnull()])


# In[ ]:


joined = join_df(joined, weather, ["State","Date"])
joined_test = join_df(joined_test, weather, ["State","Date"])
len(joined[joined.Mean_TemperatureC.isnull()]),len(joined_test[joined_test.Mean_TemperatureC.isnull()])


# In[ ]:


for df in (joined, joined_test):
    for c in df.columns:
        if c.endswith('_y'):
            if c in df.columns: df.drop(c, inplace=True, axis=1)


# Next we'll fill in missing values to avoid complications with `NA`'s. `NA` (not available) is how Pandas indicates missing values; many models have problems when missing values are present, so it's always important to think about how to deal with them. In these cases, we are picking an arbitrary *signal value* that doesn't otherwise appear in the data.

# In[ ]:


for df in (joined,joined_test):
    df['CompetitionOpenSinceYear'] = df.CompetitionOpenSinceYear.fillna(1900).astype(np.int32)
    df['CompetitionOpenSinceMonth'] = df.CompetitionOpenSinceMonth.fillna(1).astype(np.int32)
    df['Promo2SinceYear'] = df.Promo2SinceYear.fillna(1900).astype(np.int32)
    df['Promo2SinceWeek'] = df.Promo2SinceWeek.fillna(1).astype(np.int32)


# Next we'll extract features "CompetitionOpenSince" and "CompetitionDaysOpen". Note the use of `apply()` in mapping a function across dataframe values.

# In[ ]:


for df in (joined,joined_test):
    df["CompetitionOpenSince"] = pd.to_datetime(dict(year=df.CompetitionOpenSinceYear, 
                                                     month=df.CompetitionOpenSinceMonth, day=15))
    df["CompetitionDaysOpen"] = df.Date.subtract(df.CompetitionOpenSince).dt.days


# We'll replace some erroneous / outlying data.

# In[ ]:


for df in (joined,joined_test):
    df.loc[df.CompetitionDaysOpen<0, "CompetitionDaysOpen"] = 0
    df.loc[df.CompetitionOpenSinceYear<1990, "CompetitionDaysOpen"] = 0


# We add "CompetitionMonthsOpen" field, limiting the maximum to 2 years to limit number of unique categories.

# In[ ]:


for df in (joined,joined_test):
    df["CompetitionMonthsOpen"] = df["CompetitionDaysOpen"]//30
    df.loc[df.CompetitionMonthsOpen>24, "CompetitionMonthsOpen"] = 24
joined.CompetitionMonthsOpen.unique()


# Same process for Promo dates. You may need to install the `isoweek` package first.

# In[ ]:


# If needed, uncomment:
# ! pip install isoweek


# In[ ]:


from isoweek import Week
for df in (joined,joined_test):
    df["Promo2Since"] = pd.to_datetime(df.apply(lambda x: Week(
        x.Promo2SinceYear, x.Promo2SinceWeek).monday(), axis=1).astype(pd.datetime))
    df["Promo2Days"] = df.Date.subtract(df["Promo2Since"]).dt.days


# In[ ]:


for df in (joined,joined_test):
    df.loc[df.Promo2Days<0, "Promo2Days"] = 0
    df.loc[df.Promo2SinceYear<1990, "Promo2Days"] = 0
    df["Promo2Weeks"] = df["Promo2Days"]//7
    df.loc[df.Promo2Weeks<0, "Promo2Weeks"] = 0
    df.loc[df.Promo2Weeks>25, "Promo2Weeks"] = 25
    df.Promo2Weeks.unique()


# In[ ]:


joined.to_pickle('joined')
joined_test.to_pickle('joined_test')


# ## Durations

# It is common when working with time series data to extract data that explains relationships across rows as opposed to columns, e.g.:
# * Running averages
# * Time until next event
# * Time since last event
# 
# This is often difficult to do with most table manipulation frameworks, since they are designed to work with relationships across columns. As such, we've created a class to handle this type of data.
# 
# We'll define a function `get_elapsed` for cumulative counting across a sorted dataframe. Given a particular field `fld` to monitor, this function will start tracking time since the last occurrence of that field. When the field is seen again, the counter is set to zero.
# 
# Upon initialization, this will result in datetime na's until the field is encountered. This is reset every time a new store is seen. We'll see how to use this shortly.

# In[ ]:


def get_elapsed(fld, pre):
    day1 = np.timedelta64(1, 'D')
    last_date = np.datetime64()
    last_store = 0
    res = []

    for s,v,d in zip(df.Store.values,df[fld].values, df.Date.values):
        if s != last_store:
            last_date = np.datetime64()
            last_store = s
        if v: last_date = d
        res.append(((d-last_date).astype('timedelta64[D]') / day1))
    df[pre+fld] = res


# We'll be applying this to a subset of columns:

# In[ ]:


columns = ["Date", "Store", "Promo", "StateHoliday", "SchoolHoliday"]


# In[ ]:


#df = train[columns]
df = train[columns].append(test[columns])


# Let's walk through an example.
# 
# Say we're looking at School Holiday. We'll first sort by Store, then Date, and then call `add_elapsed('SchoolHoliday', 'After')`:
# This will apply to each row with School Holiday:
# * A applied to every row of the dataframe in order of store and date
# * Will add to the dataframe the days since seeing a School Holiday
# * If we sort in the other direction, this will count the days until another holiday.

# In[ ]:


fld = 'SchoolHoliday'
df = df.sort_values(['Store', 'Date'])
get_elapsed(fld, 'After')
df = df.sort_values(['Store', 'Date'], ascending=[True, False])
get_elapsed(fld, 'Before')


# We'll do this for two more fields.

# In[ ]:


fld = 'StateHoliday'
df = df.sort_values(['Store', 'Date'])
get_elapsed(fld, 'After')
df = df.sort_values(['Store', 'Date'], ascending=[True, False])
get_elapsed(fld, 'Before')


# In[ ]:


fld = 'Promo'
df = df.sort_values(['Store', 'Date'])
get_elapsed(fld, 'After')
df = df.sort_values(['Store', 'Date'], ascending=[True, False])
get_elapsed(fld, 'Before')


# We're going to set the active index to Date.

# In[ ]:


df = df.set_index("Date")


# Then set null values from elapsed field calculations to 0.

# In[ ]:


columns = ['SchoolHoliday', 'StateHoliday', 'Promo']


# In[ ]:


for o in ['Before', 'After']:
    for p in columns:
        a = o+p
        df[a] = df[a].fillna(0).astype(int)


# Next we'll demonstrate window functions in pandas to calculate rolling quantities.
# 
# Here we're sorting by date (`sort_index()`) and counting the number of events of interest (`sum()`) defined in `columns` in the following week (`rolling()`), grouped by Store (`groupby()`). We do the same in the opposite direction.

# In[ ]:


bwd = df[['Store']+columns].sort_index().groupby("Store").rolling(7, min_periods=1).sum()


# In[ ]:


fwd = df[['Store']+columns].sort_index(ascending=False
                                      ).groupby("Store").rolling(7, min_periods=1).sum()


# Next we want to drop the Store indices grouped together in the window function.
# 
# Often in pandas, there is an option to do this in place. This is time and memory efficient when working with large datasets.

# In[ ]:


bwd.drop('Store',1,inplace=True)
bwd.reset_index(inplace=True)


# In[ ]:


fwd.drop('Store',1,inplace=True)
fwd.reset_index(inplace=True)


# In[ ]:


df.reset_index(inplace=True)


# Now we'll merge these values onto the df.

# In[ ]:


df = df.merge(bwd, 'left', ['Date', 'Store'], suffixes=['', '_bw'])
df = df.merge(fwd, 'left', ['Date', 'Store'], suffixes=['', '_fw'])


# In[ ]:


df.drop(columns,1,inplace=True)


# In[ ]:


df.head()


# It's usually a good idea to back up large tables of extracted / wrangled features before you join them onto another one, that way you can go back to it easily if you need to make changes to it.

# In[ ]:


df.to_pickle('df')


# In[ ]:


df["Date"] = pd.to_datetime(df.Date)


# In[ ]:


df.columns


# In[ ]:


joined = pd.read_pickle('joined')
joined_test = pd.read_pickle('joined_test')


# In[ ]:


joined = join_df(joined, df, ['Store', 'Date'])


# In[ ]:


joined_test = join_df(joined_test, df, ['Store', 'Date'])


# The authors also removed all instances where the store had zero sale / was closed. We speculate that this may have cost them a higher standing in the competition. One reason this may be the case is that a little exploratory data analysis reveals that there are often periods where stores are closed, typically for refurbishment. Before and after these periods, there are naturally spikes in sales that one might expect. By ommitting this data from their training, the authors gave up the ability to leverage information about these periods to predict this otherwise volatile behavior.

# In[ ]:


joined = joined[joined.Sales!=0]


# We'll back this up as well.

# In[ ]:


joined.reset_index(inplace=True)
joined_test.reset_index(inplace=True)


# In[ ]:


joined.to_pickle('train_clean')
joined_test.to_pickle('test_clean')

