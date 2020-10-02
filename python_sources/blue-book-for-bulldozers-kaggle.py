#!/usr/bin/env python
# coding: utf-8

# In[ ]:


get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


from fastai.imports import *
from fastai.structured import *

from pandas_summary import DataFrameSummary
from sklearn.ensemble import RandomForestRegressor
from IPython.display import display

from sklearn.model_selection import train_test_split

from sklearn import metrics


# In[ ]:


PATH = "../input/TrainAndValid.csv"


# In[ ]:


df_raw = pd.read_csv(f"{PATH}", low_memory= False, parse_dates=['saledate'])


# In[ ]:


def display_all(df):
    with pd.option_context("display.max_rows", 1000):
        with pd.option_context("display.max_columns", 1000):
            display(df)


# In[ ]:


display_all(df_raw.tail().transpose())


# In[ ]:


df_raw.SalePrice


# In[ ]:


def add_datepart(df, fldname, drop=True):
    fld = df[fldname]
    if not np.issubdtype(fld.dtype, np.datetime64):
        df[fldname] = fld = pd.to_datetime(fld, 
                                     infer_datetime_format=True)
    targ_pre = re.sub('[Dd]ate$', '', fldname)
    for n in ('Year', 'Month', 'Week', 'Day', 'Dayofweek', 
            'Dayofyear', 'Is_month_end', 'Is_month_start', 
            'Is_quarter_end', 'Is_quarter_start', 'Is_year_end', 
            'Is_year_start'):
        df[targ_pre+n] = getattr(fld.dt,n.lower())
    df[targ_pre+'Elapsed'] = fld.astype(np.int64) // 10**9
    if drop: df.drop(fldname, axis=1, inplace=True)


# In[ ]:


add_datepart(df_raw, 'saledate')
df_raw.saleYear.head()


# In[ ]:


train_cats(df_raw)
df_raw.UsageBand.cat.categories


# In[ ]:


df_raw.UsageBand.cat.set_categories(['High', 'Medium', 'Low'], ordered=True, inplace=True)


# In[ ]:


df_raw.UsageBand.cat.categories


# In[ ]:


display_all((df_raw.isnull().sum().sort_index()/len(df_raw))*100)


# In[ ]:


df, y, nas = proc_df(df_raw, 'SalePrice')


# In[ ]:


display_all(df.tail().transpose())


# In[ ]:


m = RandomForestRegressor(n_jobs=-1, n_estimators=100)
m.fit(df, y)
m.score(df, y)


# In[ ]:


def split_vals(a,n): return a[:n].copy(), a[n:].copy()


# In[ ]:


#n_valid = 12000  # same as Kaggle's test set size
#n_trn = len(df)-n_valid
#raw_train, raw_valid = split_vals(df_raw, n_trn)
#X_train, X_valid = split_vals(df, n_trn)
#y_train, y_valid = split_vals(y, n_trn)

X_train, X_test, y_train, y_test = train_test_split(df, y, test_size=1200)


# In[ ]:


X_train.shape, y_train.shape, X_test.shape


# In[ ]:


def rmse(x,y): return math.sqrt(((x-y)**2).mean())


# In[ ]:


def print_score(m):
    res = [rmse(m.predict(X_train), y_train),
           rmse(m.predict(X_test), y_test),
           m.score(X_train, y_train), m.score(X_test, y_test)]
    if hasattr(m, 'oob_score_'): res.append(m.oob_score_)
    print(res)


# In[ ]:


m = RandomForestRegressor(n_jobs=-1, n_estimators=100)
get_ipython().run_line_magic('time', 'm.fit(X_train, y_train)')
print_score(m)
#[3060.685861023887, 8328.198733521502, 0.9822879462172822, 0.8820442561482913]

