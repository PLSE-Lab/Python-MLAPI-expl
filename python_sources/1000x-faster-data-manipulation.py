#!/usr/bin/env python
# coding: utf-8

# # 1000x faster data manipulation!
# 
# **Disclaimer: this is not my(ceshine'e) original work. I just took the notebook by Nathan Cheever from [here](https://gitlab.com/cheevahagadog/talks-demos-n-such/-/blob/master/PyGotham2019/PyGotham-updated.ipynb) and made it run on Kaggle.**
# 
# Update: programatically compare the execution times. 
# 
# ## Vectorizing with pandas and NumPy
# 
# This talk was originally given at PyGotham 2019. Original video [here](https://youtu.be/nxWginnBklU).
# 
# Slides [here](https://docs.google.com/presentation/d/1X7CheRfv0n4_I21z4bivvsHt6IDxkuaiAuCclSzia1E/edit?usp=sharing)

# In[ ]:


import pandas as pd
import numpy as np
import re
import time

from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"

pd.set_option('max_columns', 15)
pd.set_option('chained_assignment', None)


# ## Read in and setup the data 
# This CSV is an example of the data I worked on while practicing learning how to vectorize former `.apply` functoins. The data here have been scrubbed and scrambled to not relate to any real-life data.

# In[ ]:


get_ipython().system('wget https://gitlab.com/cheevahagadog/talks-demos-n-such/-/raw/master/PyGotham2019/sample_data_pygotham2019.csv')


# In[ ]:


df = pd.read_csv('sample_data_pygotham2019.csv', 
                 parse_dates=['Date Created', 'Original Record: Date Created'])


# ### Data inspection 

# In[ ]:


df.shape
df.head(5)


# In[ ]:


df.dtypes


# # First attempt at vectorizing with conditionals

# In[ ]:


def set_lead_status(row):
    if row['Current Status'] == '- None -':
        return row['Status at Time of Lead']
    else:
        return row['Current Status']


# In[ ]:


get_ipython().run_cell_magic('timeit', '-r 3 -o -q -n 1', 'test = df.apply(set_lead_status, axis=1)')


# In[ ]:


baseline_time = np.mean(_.all_runs) / _.loops


# In[ ]:


# Or another way to do it...
start = time.time()

test = df.apply(set_lead_status, axis=1)

end = time.time()
print(round(end - start, 2))


# Kinda slow. So I thought I could vectorize by taking what Sofia Hiesler said in her talk: just pass in the columns and operate on them at the same time....

# In[ ]:


def set_lead_status(col1, col2):
    if col1 == '- None -':
        return col2
    else:
        return col1


# In[ ]:


try:
    test1 = set_lead_status(df['Current Status'], df['Status at Time of Lead'])
except ValueError as e:
    print("Error:", e)


# # Enter `numpy.where()`
# [Documentation](https://docs.scipy.org/doc/numpy/reference/generated/numpy.where.html)
# Similar to Excel 'IF' function, you give it a condition, except this condition can handle the truthiness for the entire array/column. Then give it what to return if elements on that boolean array are true or false.

# In[ ]:


get_ipython().run_cell_magic('timeit', '-o -r 3 -q', "# Pandas Series Vectorized baby!!\n\n# you can pass the output directly into a pandas Series\n\ndf['normalized_status'] = np.where(\n    df['Status at Time of Lead'] == '- None -',   # <-- condition\n    df['Current Status'],                         # <-- return if true\n    df['Status at Time of Lead']                  # <-- return if false\n)")


# In[ ]:


npwhere_time = np.mean(_.all_runs) / _.loops
npwhere_time


# In[ ]:


print(f"`np.where` is {round(baseline_time / npwhere_time, 1)}x faster than `.apply`")


# Sofia mentions in her talk that you can go even faster by accessing the underlying NumPy arrays from your pandas series. This makes it faster b/c now there's only a NumPy array of your data to pass to C, with no need for handling all the stuff attached to a pandas Series that makes them so convenient to work with.

# In[ ]:


get_ipython().run_cell_magic('timeit', '-o -r 3 -q ', "# NumPy Vectorized baby!!\n\ndf['normalized_status'] = np.where(\n    df['Status at Time of Lead'].values == '- None -',\n    df['Current Status'].values, \n    df['Status at Time of Lead'].values\n)")


# In[ ]:


npwhere2_time = np.mean(_.all_runs) / _.loops


# In[ ]:


print(f"`np.where` w/ numpy vectorization is {round(baseline_time / npwhere2_time, 1)}x faster than `.apply`")


# In[ ]:


# %%timeit
# test = df.apply(works_but_slow, axis=1, raw=True)  # There is a significant speed improvement from using raw=True 
# # with pd.DataFrame.apply versus without. This option feeds NumPy arrays to the custom function instead of pd.Series objects.
# # https://stackoverflow.com/questions/52673285/performance-of-pandas-apply-vs-np-vectorize-to-create-new-column-from-existing-c


# # What about `numpy.vectorize()`?
# 
# This is a function that will take a Python function and turn it into a NumPy ufunc, so it can handle vectorized approaches. It _vectorizes_ your function, not necessarily how that function applies to your data. Big difference there.
# 
# Resources:
#  - https://docs.scipy.org/doc/numpy/reference/generated/numpy.vectorize.html
#  - https://www.experfy.com/blog/why-you-should-forget-loops-and-embrace-vectorization-for-data-science
# 

# In[ ]:


# Here is our previous function that I tried to vectorize but couldn't due to the ValueError. 
def works_fast_maybe(col1, col2):
    if col1 == '- None -':
        return col2
    else:
        return col1


# In[ ]:


# with the np.vectorize method --> returns a vectorized callable
vectfunc = np.vectorize(works_fast_maybe) #otypes=[np.float],cache=False)


# In[ ]:


get_ipython().run_cell_magic('timeit', '-o -r 3 -q', "test3 = list(vectfunc(df['Status at Time of Lead'], df['Current Status']))")


# Some guy on Medium thought this was faster -- using index setting -- but it turns out it's actually not vectorizing

# In[ ]:


def can_I_go_any_faster(status_at_time, current_status):
    # this works fine if you're setting static values
    df['test'] = 'test'# status_at_time
    df.loc[status_at_time == '- None -', 'test'] = current_status  # <-- ValueError, indexes don't match!
    df.loc[status_at_time != '- None -', 'test'] = status_at_time


# In[ ]:


get_ipython().run_cell_magic('timeit', '-r 3 -q -o', "test4 = can_I_go_any_faster(df['Status at Time of Lead'], df['Current Status'])")


# In[ ]:


def can_I_go_any_faster2(status_at_time, current_status):
    # this works fine if you're setting static values
    df['test'] = 'test'# status_at_time
    df.loc[status_at_time == '- None -', 'test'] = 'statys1_isNone' 
    df.loc[status_at_time != '- None -', 'test'] = 'statys2_notNone'


# In[ ]:


get_ipython().run_cell_magic('timeit', '', "test5 = can_I_go_any_faster2(df['Status at Time of Lead'], df['Current Status'])")


# # Multiple conditions

# ### lead_category

# In[ ]:


list1 = ['LEAD-3 Flame No Contact', 'LEAD-Campaign', 'LEAD-Claim', 'LEAD-Contact Program', 
         'LEAD-General Pool', 'LEAD-No Contact', 'LEAD-Nurture', 'LEAD-Unqualified', 'PROSPECT-LOST']

list2 = ['- None -', 'CLIENT-Closed-Sold', 'CLIENT-Handoff', 'CLIENT-Implementation', 'CLIENT-Implementation (EMR)',
         'CLIENT-Live', 'CLIENT-Partner', 'CLIENT-Referring Consultant', 'CLIENT-Transferred', 'LEAD-Engaged', 
         'LEAD-Long-Term Opportunity', 'PROSPECT-CURRENT', 'PROSPECT-LONG TERM', 'PROSPECT-NO DECISION']

# apply version
def lead_category(row):
    if row['Original Record: Date Created'] == row['Date Created']:
        return 'New Lead'
    elif row['normalized_status'].startswith('CLI'):
        return 'Client Lead'
    elif row['normalized_status'] in list1:
        return 'MTouch Lead'
    elif row['normalized_status'] in list2:
        return 'EMTouch Lead'
    return 'NA'


# In[ ]:


get_ipython().run_cell_magic('timeit', '-r 3 -o -q', "df['lead_category0'] = df.apply(lead_category, axis=1)")


# In[ ]:


baseline_time = np.mean(_.all_runs) / _.loops


# You can call a `np.where` for every condition, and it will run fine. But it gets a little hard to read after a while.

# In[ ]:


get_ipython().run_cell_magic('timeit', '-r 3 -o -q', "df['lead_category'] = \\\n    np.where(df['Original Record: Date Created'].values == df['Date Created'].values, 'New Lead', \n            np.where(df['normalized_status'].str.startswith('CLI').values, 'Client Lead', \n                    np.where(df['normalized_status'].isin(list1).values, 'MTouch Lead', \n                            np.where(df['normalized_status'].isin(list2).values, 'EMTouch Lead', \n                                     'NA') \n                                  )\n                         )\n                )")


# # Enter `numpy.select()`
# Cleaner (and even faster!) and doing multiple nested `np.where` calls for each conditions.
# Order of operations matter!
# 
# [Documentation](https://docs.scipy.org/doc/numpy/reference/generated/numpy.select.html)

# In[ ]:


get_ipython().run_cell_magic('timeit', '-r 3 -o -q', "conditions = [\n    df['Original Record: Date Created'].values == df['Date Created'].values,\n    df['normalized_status'].str.startswith('CLI').values,\n    df['normalized_status'].isin(list1).values,\n    df['normalized_status'].isin(list2).values\n]\n\nchoices = [\n    'New Lead', \n    'Client Lead', \n    'MTouch Lead',\n    'EMTouch Lead'\n]\n\n\ndf['lead_category1'] = np.select(conditions, choices, default='NA')  # Order of operations matter!")


# In[ ]:


npselect_time = np.mean(_.all_runs) / _.loops


# In[ ]:


# Their output logic is the same
(df.lead_category == df.lead_category1).all()


# In[ ]:


print(f"`np.select` w/ numpy vectorization is {round(baseline_time / npselect_time, 2)}x faster than nested .apply()")


# ## What about nested multiple conditionals? Can we vectorize that?
# Yes!

# In[ ]:


# This is something you might think you can't vectorize, but you sure can!
def sub_conditional(row):
    if row['Inactive'] == 'No':
        if row['Providers'] == 0:
            return 'active_no_providers'
        elif row['Providers'] < 5:
            return 'active_small'
        else:
            return 'active_normal'
    elif row['duplicate_leads']:
        return 'is_dup'
    else:
        if row['bad_leads']:
            return 'active_bad'
        else:
            return 'active_good'


# In[ ]:


get_ipython().run_cell_magic('timeit', '-r 3 -q -o', "# Let's time how long it takes to apply a nested multiple condition func\ndf['lead_type'] = df.apply(sub_conditional, axis=1)")


# In[ ]:


baseline_time = np.mean(_.all_runs) / _.loops


# In[ ]:


get_ipython().run_cell_magic('timeit', '-r 3 -o -q', "\n# With np.select, could do .values here for additional speed, but left out to avoid too much text\nconditions = [\n    ((df['Inactive'] == 'No') & (df['Providers'] == 0)),\n    ((df['Inactive'] == 'No') & (df['Providers'] < 5)),\n    df['Inactive'] == 'No',\n    df['duplicate_leads'],  # <-- you can also just evaluate boolean arrays\n    df['bad_leads'],\n]\n\nchoices = [\n    'active_no_providers',\n    'active_small',\n    'active_normal',\n    'is_dup',\n    'active_bad',\n]\n\ndf['lead_type_vec'] = np.select(conditions, choices, default='NA')")


# In[ ]:


npselect_time = np.mean(_.all_runs) / _.loops


# In[ ]:


mask = (
    ((df['Inactive'] == 'No') & (df['Providers'] == 0))
    & ((df['Inactive'] == 'No') & (df['Providers'] < 5))
    & (df['Inactive'] == 'No')
    & (df['duplicate_leads'])  # <-- you can also just evaluate boolean arrays
    & df['bad_leads']
)


# In[ ]:


print(f"`np.select` is {round(baseline_time / npselect_time, 2)} faster than nested .apply()")


# In[ ]:


get_ipython().run_cell_magic('timeit', '-r 3 -o -q', "\n# With np.select\nconditions = [\n    ((df['Inactive'].values == 'No') & (df['Providers'].values == 0)),\n    ((df['Inactive'].values == 'No') & (df['Providers'].values < 5)),\n    df['Inactive'].values == 'No',\n    df['duplicate_leads'].values,  # <-- you can also just evaluate boolean arrays\n    df['bad_leads'].values,\n]\n\nchoices = [\n    'active_no_providers',\n    'active_small',\n    'active_normal',\n    'is_dup',\n    'active_bad',\n]\n\ndf['lead_type_vec'] = np.select(conditions, choices, default='NA')")


# In[ ]:


npselect2_time = np.mean(_.all_runs) / _.loops


# In[ ]:


print(f"`np.select` w/ vectorization is {round(baseline_time / npselect2_time, 2)} faster than nested .apply()")


# # What about more complicated things?
# Of course this is just a tiny sample of things you might encounter, but I thought they could be useful to see how vectorization can still apply even with otherwise difficult cases.

# #### Strings

# In[ ]:


df.head(2)


# In[ ]:


# Doing a regex search to find string patterns

def find_paid_nonpaid(s):
    if re.search(r'non.*?paid', s, re.I):
        return 'non-paid'
    elif re.search(r'Buyerzone|^paid\s+', s, re.I):
        return 'paid'
    else:
        return 'unknown'


# In[ ]:


get_ipython().run_cell_magic('timeit', '-r 3 -o -q', "# our old friend .apply()\ndf['lead_source_paid_unpaid'] = df['Lead Source'].apply(find_paid_nonpaid)")


# In[ ]:


baseline_time = np.mean(_.all_runs) / _.loops


# How does `np.vectorize()` do for strings?

# In[ ]:


get_ipython().run_cell_magic('timeit', '-r 3 -o -q', "\nvect_str = np.vectorize(find_paid_nonpaid)\n\ndf['lead_source_paid_unpaid1'] = vect_str(df['Lead Source'])")


# In[ ]:


get_ipython().run_cell_magic('timeit', '-r 3 -o -q', "# How does a list comprehension do?\ndf['lead_source_paid_unpaid2'] = ['non-paid' if re.search(r'non.*?paid', s, re.I) \n                                  else 'paid' if re.search(r'Buyerzone|^paid\\s+', s, re.I) \n                                  else 'unknown' for s in df['Lead Source']]")


# pandas provides the `.str()` methods for working with strings.

# In[ ]:


get_ipython().run_cell_magic('timeit', '-r 3 -o -q', "# How does this compare?\nconditions = [\n    df['Lead Source'].str.contains(r'non.*?paid', case=False, na=False),\n    df['Lead Source'].str.contains(r'Buyerzone|^paid\\s+', case=False, na=False),\n]\n\nchoices = [\n    'non-paid',\n    'paid'\n]\n\ndf['lead_source_paid_unpaid1'] = np.select(conditions, choices, default='unknown')")


# *what about not searching strings?*

# In[ ]:


get_ipython().run_cell_magic('timeit', '-r 3 -o -q', "df['lowerls'] = df['Lead Source'].apply(lambda x: x.lower())")


# In[ ]:


get_ipython().run_cell_magic('timeit', '-r 3 -o -q', "df['lowerls1'] = df['Lead Source'].str.lower()")


# #### Dictionary lookups

# In[ ]:


channel_dict = {
    'Billing Service': 'BS', 'Consultant': 'PD', 'Educational': 'PD', 
    'Enterprise': 'PD', 'Hospital': 'PD', 'IPA': 'PD', 'MBS': 'RCM', 
    'MSO': 'PD', 'Medical practice': 'PD', 'Other': 'PD', 'Partner': 'PD',
    'PhyBillers': 'BS', 'Practice': 'PD', 'Purchasing Group': 'PD',
    'Reseller': 'BS', 'Vendor': 'PD', '_Other': 'PD', 'RCM': 'RCM'
}

def a_dict_lookup(row):
    if row['Providers'] > 7:
        return 'Upmarket'
    else:
        channel = channel_dict.get(row['Category'])
        return channel


# In[ ]:


get_ipython().run_cell_magic('timeit', '-o -r 3 -q -n 1', "df['dict_lookup'] = df.apply(a_dict_lookup, axis=1)")


# In[ ]:


baseline_time = np.mean(_.all_runs) / _.loops


# In[ ]:


get_ipython().run_cell_magic('timeit', '-r 3 -o -q', "df['dict_lookup1'] = np.where(\n    df['Providers'].values > 7, \n    'Upmarket',\n    df['Category'].map(channel_dict)\n)")


# In[ ]:


npwhere_time = np.mean(_.all_runs) / _.loops


# In[ ]:


baseline_time / npwhere_time


# In[ ]:


get_ipython().run_cell_magic('timeit', '-r 3 -o -q', "channel_values = df['Category'].map(channel_dict)\ndf['dict_lookup1'] = np.where(\n    df['Providers'] > 7, \n    'Upmarket',\n    channel_values\n)")


# In[ ]:


get_ipython().run_cell_magic('timeit', '-r 3 -o -q', "# Using np.vectorize to vectorize a dictionary .get() method works, but is slower than .map()\nchannel_values = np.vectorize(channel_dict.get)(df['Category'].values)\ndf['dict_lookup2'] = np.where(\n    df['Providers'] > 7, \n    'Upmarket',\n    channel_values\n)")


# In[ ]:


print((df['dict_lookup'] == df['dict_lookup1']).all())
print((df['dict_lookup'] == df['dict_lookup2']).all())


# #### Dates

# In[ ]:


df.head(2)


# In[ ]:


# make a new column called 'Start Date' for dummy testing
# ONly do a fraction so we have some NaN values
df['Start Date'] = df['Date Created'].sample(frac=0.8)


# In[ ]:


def weeks_to_complete(row) -> float:
    """Calculate the number of weeks between two dates"""
    if pd.isnull(row['Start Date']):
        return (row['Original Record: Date Created'] -  row['Date Created']).days / 7
    else:
        return (row['Date Created'] - row['Start Date']).days / 7


# In[ ]:


get_ipython().run_cell_magic('timeit', '-r 3 -o -q -n 1', 'wtc1 = df.apply(weeks_to_complete, axis=1)')


# In[ ]:


baseline_time = np.mean(_.all_runs) / _.loops


# One approach to vectorization is to use pandas `.dt` datetime accessors. They have lots of goodies..

# In[ ]:


get_ipython().run_cell_magic('timeit', '-r 3 -o -q', "wtc2 = np.where(\n    df['Start Date'].isnull().values,\n    (df['Original Record: Date Created'].values - df['Date Created']).dt.days / 7,\n    (df['Date Created'].values - df['Start Date']).dt.days / 7\n)")


# In[ ]:


npwhere_time = np.mean(_.all_runs) / _.loops


# Another approach is to do ndarray type casting, converting our series into numpy timedelta arrays. This way is faster, but more verbose and kinda more code for basically the samething.

# In[ ]:


get_ipython().run_cell_magic('timeit', '-r 3 -o -q', "wtc3 = np.where(\n    df['Start Date'].isnull().values,\n    ((df['Original Record: Date Created'].values - df['Date Created'].values).astype('timedelta64[D]') / np.timedelta64(1, 'D')) / 7,\n    ((df['Date Created'].values - df['Start Date'].values).astype('timedelta64[D]') / np.timedelta64(1, 'D')) / 7\n)")


# In[ ]:


npwhere2_time = np.mean(_.all_runs) / _.loops


# In[ ]:


# How much faster is our last way over .apply()?
baseline_time / npwhere2_time


# #### Needing values on other rows for the logic
# This example comes from a task I had to recreate a function like this in Excel:
# ```excel
# =IF(A2=A1, IF(L2-L1 < 5, 0, 1), 1))
# ```
# Where the `A` column is for ids, and the `L` column is for dates.

# In[ ]:


def time_looper(df):
    """ Using a plain Python for loop"""
    output = []
    for i, row in enumerate(range(0, len(df))):
        if i > 0:
            
            # compare the current id to the row above
            if df.iloc[i]['Internal ID'] == df.iloc[i-1]['Internal ID']:
                
                # compare the current date to the row above
                if (df.iloc[i]['Date Created'] - df.iloc[i-1]['Original Record: Date Created']).days < 5:
                    output.append(0)
                else:
                    output.append(1)
            else:
                output.append(1)
        else:
            output.append(np.nan)
    return output


# In[ ]:


def time_looper2(df):
    """Using pandas dataframe `.iterrows()` method for iterating over rows"""
    output = []
    for i, row in df.iterrows():
        if i > 0:
            if df.iloc[i]['Internal ID'] == df.iloc[i-1]['Internal ID']:
                if (df.iloc[i]['Date Created'] - df.iloc[i-1]['Original Record: Date Created']).days < 5:
                    output.append(0)
                else:
                    output.append(1)
            else:
                output.append(1)
        else:
            output.append(np.nan)
    return output


# In[ ]:


df.sort_values(['Internal ID', 'Date Created'], inplace=True)


# In[ ]:


get_ipython().run_cell_magic('timeit', '-r 3 -o -q -n 1', "df['time_col_raw_for'] = time_looper(df)")


# In[ ]:


baseline_time = np.mean(_.all_runs) / _.loops


# In[ ]:


get_ipython().run_cell_magic('timeit', '-r 3 -o -q -n 1', "df['time_col_iterrows'] = time_looper2(df)")


# In[ ]:


iterrows_time =  np.mean(_.all_runs) / _.loops


# Our approach to vectorizing this unfortunate situation was two-fold:
#  1. Using the pandas `.shift()` function, we moved previous values down so they're on the same axis as what we're comparing them to 
#  2. `np.select()` for the vectorized conditional logic check
#  

# In[ ]:


get_ipython().run_cell_magic('timeit', '-o -r 3 -q', "previous_id = df['Internal ID'].shift(1).fillna(0).astype(int)\nprevious_date = df['Original Record: Date Created'].shift(1).fillna(pd.Timestamp('1900'))\n\nconditions = [\n    ((df['Internal ID'].values ==  previous_id) & \n     (df['Date Created'] - previous_date).astype('timedelta64[D]') < 5),\n    df['Internal ID'].values ==  previous_id\n]\nchoices = [0, 1]\ndf['time_col1'] = np.select(conditions, choices, default=1)")


# In[ ]:


shift_time = np.mean(_.all_runs) / _.loops


# In[ ]:


baseline_time / shift_time


# In[ ]:


# TODO: figure out what's going on here
(df['time_col1'] == df['time_col_iterrows']).all()


# # Other alternatives

# ## A _parallel_ apply func
# Source: https://towardsdatascience.com/make-your-own-super-pandas-using-multiproc-1c04f41944a1

# In[ ]:


from multiprocessing import Pool


# In[ ]:


def p_apply(df, func, cores=4):
    """Pass in your dataframe and the func to apply to it"""
    df_split = np.array_split(df, cores)
    pool = Pool(n_cores)
    df = pd.concat(pool.map(func, df_split))
    pool.close()
    pool.join()
    return df


# In[ ]:


# df = p_apply(df, func=some_big_function)


# ## Dask
# https://docs.dask.org

# In[ ]:




