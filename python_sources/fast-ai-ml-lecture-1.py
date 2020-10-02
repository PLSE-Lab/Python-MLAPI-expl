#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from fastai.imports import *
from fastai.structured import *

from pandas_summary import DataFrameSummary
from sklearn.ensemble import RandomForestRegressor
from IPython.display import display

from sklearn import metrics


# In[ ]:


PATH = '../input/'
get_ipython().system('ls {PATH}')


# In[ ]:


df_raw = pd.read_csv(f'{PATH}Train/Train.csv', low_memory=False, parse_dates=['saledate'])


# In[ ]:


def display_all(df):
    with pd.option_context('display.max_rows', 1000):
        with pd.option_context('display.max_columns', 1000):
            display(df)


# In[ ]:


display_all(df_raw.tail().transpose())


# In[ ]:


df_raw.SalePrice = np.log(df_raw.SalePrice)


# In[ ]:


add_datepart(df_raw, 'saledate')
df_raw.saleYear.head()


# In[ ]:


df_raw.columns


# In[ ]:


train_cats(df_raw)


# In[ ]:


display_all(df_raw.isnull().sum().sort_index() / len(df_raw))


# In[ ]:


os.makedirs('tmp', exist_ok=True)
df_raw.to_feather('tmp/raw')


# In[ ]:


df, y, *rest = proc_df(df_raw, 'SalePrice')


# In[ ]:


df.head().transpose()


# In[ ]:


m = RandomForestRegressor(n_jobs=-1)
m.fit(df, y)
m.score(df, y)


# In[ ]:


def split_vals(a, n): return a[:n].copy(), a[n:].copy()

n_valid = 12000
n_trn = len(df) - n_valid
raw_train, raw_valid = split_vals(df_raw, n_trn)
X_train, X_valid = split_vals(df, n_trn)
y_train, y_valid = split_vals(y, n_trn)

X_train.shape, y_train.shape, X_valid.shape


# In[ ]:


def rmse(x, y): return math.sqrt(((x-y)**2).mean())

def print_score(m):
    res = [rmse(m.predict(X_train), y_train),
          rmse(m.predict(X_valid), y_valid),
          m.score(X_train, y_train),
          m.score(X_valid, y_valid)]
    if hasattr(m, 'oob_score_') :
        res.append(m.oob_score_)
        
    print(res)


# In[ ]:


m = RandomForestRegressor(n_jobs=-1)
get_ipython().run_line_magic('time', 'm.fit(X_train, y_train)')
print_score(m)


# In[ ]:




