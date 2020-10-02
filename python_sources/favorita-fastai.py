#!/usr/bin/env python
# coding: utf-8

# ****### Favorita 2016 10 weeks (28-06 to 31-08 : 65 days inclusive)
# 
# Discussed on http://forums.fast.ai/t/corporacion-favorita-grocery-sales-forecasting/8359/10

# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')
get_ipython().run_line_magic('reload_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')


# In[ ]:


from fastai.structured import *
from fastai.column_data import *
np.set_printoptions(threshold=50, edgeitems=20)
import datetime

PATH = '../input/'


# In[ ]:


# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", PATH]).decode("utf8"))

# Any results you write to the current directory are saved as output.


# ## Create datasets

# In[ ]:


table_names = ['train', 'stores', 'items', 'transactions', 
               'holidays_events', 'oil', 'test', 'sample_submission']


# We'll be using the popular data manipulation framework pandas. Among other things, pandas allows you to manipulate tables/data frames in python as one would in a database.
# 
# We're going to go ahead and load all of our csv's as dataframes into the list tables.

# In[ ]:


tables = [pd.read_csv(f'{PATH}{fname}.csv', low_memory=False) for fname in table_names]


# In[ ]:


from IPython.display import HTML


# We can use `head()` to get a quick look at the contents of each table:
# * **train**: includes the target unit_sales by date, store_nbr, and item_nbr and a unique id to label rows
# 
# * **stores**: metadata; including city, state, type, and cluster; cluster is a grouping of similar stores
# 
# * **items**: metadata; including family, class, and perishable; perishable have a score weight of 1.25; otherwise, the weight is 1.0
# 
# * **transactions**: count of sales transactions for each date, store_nbr combination. Only included for the training data timeframe
# 
# * **holidays_events**: metadata; Pay special attention to the transferred column.
#     * A holiday that is transferred officially falls on that calendar day, but was moved to another date by the government. A transferred day is more like a normal day than a holiday. To find the day that it was actually celebrated, look for the corresponding row where type is Transfer.
#     * For example, the holiday Independencia de Guayaquil was transferred from 2012-10-09 to 2012-10-12, which means it was celebrated on 2012-10-12. Days that are type Bridge are extra days that are added to a holiday (e.g., to extend the break across a long weekend). These are frequently made up by the type Work Day which is a day not normally scheduled for work (e.g., Saturday) that is meant to payback the Bridge.
#     * Additional holidays are days added a regular calendar holiday, for example, as typically happens around Christmas (making Christmas Eve a holiday).
# 
# * **oil**: Daily oil price. Includes values during both the train and test data timeframe. (Ecuador is an oil-dependent country and it's economical health is highly vulnerable to shocks in oil prices.)
# 
# * **test**: Test data, with the date, store_nbr, item_nbr combinations that are to be predicted, along with the onpromotion information.
#     * NOTE: The test data has a small number of items that are not contained in the training data. Part of the exercise will be to predict a new item sales based on similar products.
#     * The public / private leaderboard split is based on time. All items in the public split are also included in the private split.
#     
# * **sample_submission**: sample submission file in the correct format. It is highly recommend you zip your submission file before uploading!

# In[ ]:


for t in tables: display(t.head(), t.shape)


# In[ ]:


# The following returns summarized aggregate information to each table accross each field.
for t in tables: display(DataFrameSummary(t).summary())


# ## Data Cleaning / Feature Engineering

# As a structured data problem, we necessarily have to go through all the cleaning and feature engineering, even though we're using a neural network.

# In[ ]:


train, stores, items, transactions, holidays_events, oil, test, sample_submission = tables


# In[ ]:


len(train),len(test)


# We turn state OnPromotion to booleans, to make them more convenient for modeling. We can do calculations on pandas fields using notation very similar (often identical) to numpy.
train.onpromotion = train.onpromotion!='0'
test.onpromotion = test.onpromotion!='0'
# #### Optimizing the Date format

# The following extracts particular date fields from a complete datetime for the purpose of constructing categoricals.
# 
# You should *always* consider this feature extraction step when working with date-time. Without expanding your date-time into these additional fields, you can't capture any trend/cyclical behavior as a function of time at any of these granularities. We'll add to every table with a date field.
# 
# **note**: Dayofweek starts at 0, Dayofyear starts at 1

# In[ ]:


add_datepart(train, "date", drop=False)

%add_datepart
# In[ ]:


add_datepart(transactions, "date", drop=False)


# In[ ]:


add_datepart(holidays_events, "date", drop=False)
add_datepart(oil, "date", drop=False)
add_datepart(test, "date", drop=False)


# In[ ]:


for t in tables: display(t.head(), t.shape)


# #### Reducing data to the last 10 weeks for the training set (16 days needed for Validation/Test)

# In[ ]:


# If done on all train data, results in 125m rows. So, we're taking a small sample of the last 8 weeks:
train_mask_10w = (train['date'] >= '2016-06-28') & (train['date'] <= '2016-08-31')
print(train.shape)


# In[ ]:


train =  train[train_mask_10w]
print(train.shape)


# In[ ]:


train.head()


# In[ ]:


transactions_mask_10w = (transactions['date'] >= '2016-06-28') & (transactions['date'] <= '2016-08-31')
print(transactions.shape)


# In[ ]:


transactions =  transactions[transactions_mask_10w]
print(transactions.shape)


# In[ ]:


transactions.head()


# In[ ]:


holidays_events_mask_10w = (holidays_events['date'] >= '2016-06-28') & (holidays_events['date'] <= '2016-08-31')
print(holidays_events.shape)


# In[ ]:


holidays_events =  holidays_events[holidays_events_mask_10w]
print(holidays_events.shape)


# In[ ]:


holidays_events.head()


# In[ ]:


oil_mask_10w = (oil['date'] >= '2016-06-28') & (oil['date'] <= '2016-08-31')
print(oil.shape)


# In[ ]:


oil =  oil[oil_mask_10w]
print(oil.shape)


# In[ ]:


oil.head()


# ### Join the tables

# `join_df` is a function for joining tables on specific fields. By default, we'll be doing a left outer join of `right` on the `left` argument using the given fields for each table.
# 
# Pandas does joins using the `merge` method. The `suffixes` argument describes the naming convention for duplicate fields. We've elected to leave the duplicate field names on the left untouched, and append a "\_y" to those on the right.

# In[ ]:


def join_df(left, right, left_on, right_on=None, suffix='_y'):
    if right_on is None: right_on = left_on
    return left.merge(right, how='left', left_on=left_on, right_on=right_on, 
                      suffixes=("", suffix))


# Now we can outer join all of our data into a single dataframe. Recall that in outer joins everytime a value in the joining field on the left table does not have a corresponding value on the right table, the corresponding row in the new table has Null values for all right table fields. One way to check that all records are consistent and complete is to check for Null values post-join, as we do here.
# 
# *Aside*: Why not just do an inner join?
# If you are assuming that all records are complete and match on the field you desire, an inner join will do the same thing as an outer join. However, in the event you are wrong or a mistake is made, an outer join followed by a null-check will catch it. (Comparing before/after # of rows for inner join is equivalent, but requires keeping track of before/after row #'s. Outer join is easier.)

# In[ ]:


joined = join_df(train, stores, "store_nbr")
len(joined[joined.type.isnull()])


# In[ ]:


joined.head()


# In[ ]:


joined_test = join_df(test, stores, "store_nbr")
len(joined_test[joined_test.type.isnull()])


# In[ ]:


joined = join_df(joined, items, "item_nbr")
len(joined[joined.family.isnull()])


# In[ ]:


joined.head()


# In[ ]:


joined_test = join_df(joined_test, items, "item_nbr")
len(joined_test[joined_test.family.isnull()])


# In[ ]:


joined = join_df(joined, transactions, ["date", "store_nbr"] )
len(joined[joined.store_nbr.isnull()])


# In[ ]:


joined_test = join_df(joined_test, transactions, ["date", "store_nbr"] )
len(joined_test[joined_test.store_nbr.isnull()])


# #### **Note**: at this stage, we don't incorporate the Holidays (needs tuning for local vs national) or the Oil prices, this will also require external data sources on the Test set.
# 
# **TBD**

# In[ ]:





# In[ ]:


# we drop the duplicate columns ending with _y
for df in (joined, joined_test):
    for c in df.columns:
        if c.endswith('_y'):
            if c in df.columns: df.drop(c, inplace=True, axis=1)


# In[ ]:


joined.head()


# In[ ]:


joined.describe()


# In[ ]:


joined_test.head()


# Next we'll fill in missing values to avoid complications with NA's. NA (not available) is how Pandas indicates missing values; many models have problems when missing values are present, so it's always important to think about how to deal with them. In these cases, we are picking an arbitrary signal value that doesn't otherwise appear in the data.
# 
# ** Note**: as seen below, its seems there are no NANs !?!?!

# In[ ]:


# Check if any NANs
joined.isnull().values.any()


# In[ ]:


# Check if any NANs (slower, more complete)
joined.isnull().sum().sum()


# ## Durations : TBD !

# **NOTE: code from Rossmann has 25+ cells**
# 
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
        res.append(((d-last_date).astype('timedelta64[D]') / day1).astype(int))
    df[pre+fld] = res
# We'll be applying this to a subset of columns:

# columns = ["Date", "Store", "Promo", "StateHoliday", "SchoolHoliday"]

# TBD

# In[ ]:




%add_datepart
# ## Create features

# In[ ]:


# Look at all columns pivoted to rows
joined.head().T.head(40)


# In[ ]:


# dropping "Elasped" as it generates an error later, due to crazy 10 digits
joined.drop(['Elapsed'],axis = 1, inplace = True)


# In[ ]:


joined.head().T.head(40)


# Now that we've engineered all our features, we need to convert to input compatible with a neural network.
# 
# This includes converting categorical variables into contiguous integers or one-hot encodings, normalizing continuous features to standard normal, etc...

# In[ ]:


cat_vars = ['store_nbr', 'item_nbr', 'onpromotion', 'Year', 'Month', 'Week', 'Day', 'Dayofweek', 'Dayofyear',
            'city', 'state', 'type', 'cluster', 'family', 'class', 'perishable']


# In[ ]:


contin_vars = ['transactions']


# In[ ]:


n = len(joined); n


# In[ ]:


for v in cat_vars:
    joined[v] = joined[v].astype('category').cat.as_ordered()


# In[ ]:


for v in cat_vars:
    joined_test[v] = joined_test[v].astype('category').cat.as_ordered()

for v in contin_vars:
    joined[v] = joined[v].astype('float32')
    joined_test[v] = joined_test[v].astype('float32')
# In[ ]:


for v in contin_vars:
    joined[v] = joined[v].astype('float32')


# In[ ]:


dep = 'unit_sales'
joined = joined[cat_vars+contin_vars+[dep, 'date']].copy()


# In[ ]:


joined_test[dep] = 0
joined_test = joined_test[cat_vars+contin_vars+[dep, 'date', 'id']].copy()


# In[ ]:


joined.head().T.head(40)


# In[ ]:


joined_test.head().T.head(40)

# check this cell function ?
apply_cats(joined_test, joined)
# In[ ]:


idxs = get_cv_idxs(n)
joined_samp = joined.iloc[idxs].set_index("date")
samp_size = len(joined_samp)
samp_size


# In[ ]:


joined_samp = joined.set_index("date")


# In[ ]:


samp_size = len(joined_samp)
samp_size


# In[ ]:


joined_samp.head()


# In[ ]:


joined_samp.tail()

%proc_df
[x, y, nas, mapper(optional)]:

    x: x is the transformed version of df. x will not have the response variable
        and is entirely numeric.

    y: y is the response variable

    nas (handles missing values): returns a dictionary of which nas it created, and the associated median.

    mapper: A DataFrameMapper which stores the mean and standard deviation of the corresponding continous
    variables which is then used for scaling of during test-time.

# In[ ]:


df, y, nas, mapper = proc_df(joined_samp, 'unit_sales', do_scale=True)


# In[ ]:


yl = np.log(y)


# In[ ]:


# df is now a entirely numeric dataframe, without the "unit sales" columns
df.head()


# In[ ]:


# y contains the "unit sales" now
y


# In[ ]:


min(y)


# In[ ]:


yl


# In[ ]:


max(y)


# In[ ]:


np.isnan(y).any()


# In[ ]:


np.isnan(y).


# In[ ]:


joined_test = joined_test.set_index("date")


# In[ ]:


joined_test.head()


# In[ ]:


# joined_test.drop(['transactions'], axis = 1, inplace = True)


# In[ ]:


df_test, _, nas, mapper = proc_df(joined_test, 'unit_sales', do_scale=True, skip_flds=['transactions'],
                                  na_dict=nas)


# In time series data, cross-validation is not random. Instead, our holdout data is generally the most recent data, as it would be in real application. This issue is discussed in detail in [this post](http://www.fast.ai/2017/11/13/validation-sets/) on our web site.
# 
# One approach is to take the last 25% of rows (sorted by date) as our validation set.

# In[ ]:


#ratio of .754 is 16 days by 65 days, to be close to real test duration
train_ratio = 0.754
# train_ratio = 0.9
train_size = int(samp_size * train_ratio); train_size
val_idx = list(range(train_size, len(df)))


# An even better option for picking a validation set is using the exact same length of time period as the test set uses - this is implemented here:
val_idx = np.flatnonzero(
    (df.index<=datetime.datetime(2016,8,16)) & (df.index>=datetime.datetime(2016,8,31)))
# In[ ]:


len(val_idx)


# In[ ]:


samp_size


# In[ ]:


1 - (len(val_idx)/ samp_size)


# In[ ]:





# ## Deep Learning

# We're ready to put together our models.
# 
# Root-mean-squared percent error is the metric Kaggle used for this ROSSMANN competition.

# In[ ]:


#from Rossmann
def inv_y(a): return np.exp(a)

def exp_rmspe(y_pred, targ):
    targ = inv_y(targ)
    pct_var = (targ - inv_y(y_pred))/targ
    return math.sqrt((pct_var**2).mean())

max_log_y = np.max(y)
y_range = (0, max_log_y*1.2)

# Favorita
# Normalized Weighted Root Mean Squared Logarithmic Error (NWRMSLE)
# https://www.kaggle.com/tkm2261/eval-metric-and-evaluating-last-year-sales-bench
WEIGHTS = 
def NWRMSLE(y, pred):
    y = y.clip(0, y.max())
    pred = pred.clip(0, pred.max())
    score = np.nansum(WEIGHTS * ((np.log1p(pred) - np.log1p(y)) ** 2)) / WEIGHTS.sum()
    return np.sqrt(score)
# We can create a ModelData object directly from out data frame.

# In[ ]:


md = ColumnarModelData.from_data_frame(PATH, val_idx, df, y.astype(np.float32), cat_flds=cat_vars, bs=512,
                                       test_df=df_test)


# Some categorical variables have a lot more levels than others. Store, in particular, has over a thousand in the Rossmann competition.
# Let's see in Favorita.

# In[ ]:


cat_sz = [(c, len(joined_samp[c].cat.categories)+1) for c in cat_vars]


# In[ ]:


cat_sz


# We use the *cardinality* of each variable (that is, its number of unique values) to decide how large to make its *embeddings*. Each level will be associated with a vector with length defined as below.

# In[ ]:


emb_szs = [(c, min(50, (c+1)//2)) for _,c in cat_sz]


# In[ ]:


emb_szs


# In[ ]:


m = md.get_learner(emb_szs, len(df.columns)-len(cat_vars),
                   0.04, 1, [1000,500], [0.001,0.01], y_range=y_range)
lr = 1e-3


# In[ ]:


m.lr_find()


# In[ ]:


m.sched.plot(100)


# In[ ]:





# ### Sample
m = md.get_learner(emb_szs, len(df.columns)-len(cat_vars),
                   0.04, 1, [1000,500], [0.001,0.01], y_range=y_range)
lr = 1e-3
# In[ ]:


m.fit(lr, 3, metrics=[exp_rmspe])


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




