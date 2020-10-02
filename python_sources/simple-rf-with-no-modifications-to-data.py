#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from fastai.imports import *
from fastai.structured import *


# In[ ]:


get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')


# In[ ]:


from pandas_summary import DataFrameSummary


# In[ ]:


from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier


# In[ ]:


from IPython.display import display


# In[ ]:


from sklearn import metrics


# In[ ]:


PATH = "../input/"


# ### load_data

# In[ ]:


df_raw = pd.read_csv(f'{PATH}train.csv', low_memory = False)


# In[ ]:


df_test = pd.read_csv(f'{PATH}test.csv', low_memory = False)


# In[ ]:


def display_all(df):
    with pd.option_context("display.max_rows", 1000, "display.max_columns", 1000):
            display(df)


# In[ ]:


display_all(df_raw.head().T)


# In[ ]:


display_all(df_raw.tail().T)


# In[ ]:


for col in df_raw.columns.tolist(): 
    if df_raw[col].dtype =='object':
        print (col, df_raw[col].dtype)


# In[ ]:


df_raw.edjefa.value_counts()


# In[ ]:


df_raw.edjefe.value_counts()


# In[ ]:


df_raw.dependency.value_counts()


# In[ ]:


df_raw[df_raw['dependency'] == 'yes'].idhogar.value_counts()


# In[ ]:


display_all(df_raw[df_raw['idhogar'] == 'ae6cf0558'])


# In[ ]:


df_raw.Target.value_counts()


# In[ ]:


display_all(df_raw[df_raw['idhogar'] == 'fd8a6d014'][df_raw['parentesco1'] == 1])


# In[ ]:


df_raw['idhogar'].nunique()


# In[ ]:


df_raw['parentesco1'].sum()


# In[ ]:


df_raw['parentesco1'].unique()


# In[ ]:


df_raw.groupby(['idhogar'])[['parentesco1']].sum().sort_values(by = 'parentesco1')


# In[ ]:


display_all(df_raw[df_raw['idhogar'].isin(['1bc617b23','03c6bdf85','61c10e099','ad687ad89','1367ab31d','f2bfa75c4','6b1b2405f','896fe6d3e','c0c8a5013','b1f4d89d7','374ca5a19','bfd5067c2','a0812ef17','d363d9183','09b195e7a'])].sort_values('idhogar'))


# In[ ]:


df_raw.head()


# In[ ]:


df_raw.edjefe.unique()


# In[ ]:


df_raw.edjefa.unique()


# In[ ]:


df_raw.edjefe.head()


# ### modifications to dataset

# In[ ]:


df_raw.edjefe = np.where(df_raw.edjefe == 'yes', 1,
        np.where(df_raw.edjefe == 'no', 0,
                df_raw.edjefe))


# In[ ]:


df_raw.edjefa = np.where(df_raw.edjefa == 'yes', 1,
        np.where(df_raw.edjefa == 'no', 0,
                df_raw.edjefa))


# In[ ]:


df_raw.dependency = np.where(df_raw.dependency == 'yes', 1,
        np.where(df_raw.dependency == 'no', 0,
                df_raw.dependency))


# In[ ]:


df_test.edjefe = np.where(df_test.edjefe == 'yes', 1,
        np.where(df_test.edjefe == 'no', 0,
                df_test.edjefe))


# In[ ]:


df_test.edjefa = np.where(df_test.edjefa == 'yes', 1,
        np.where(df_test.edjefa == 'no', 0,
                df_test.edjefa))


# In[ ]:


df_test.dependency = np.where(df_test.dependency == 'yes', 1,
        np.where(df_test.dependency == 'no', 0,
                df_test.dependency))


# In[ ]:


df_raw = df_raw.sort_values(by = 'idhogar')


# In[ ]:


df_raw = df_raw.reset_index (drop = True)


# In[ ]:


train_cats(df_raw)


# In[ ]:


apply_cats(df_test, df_raw)


# In[ ]:


df_train, label_train, nas = proc_df(df_raw.drop(['Id', 'idhogar'], axis = 1), 'Target')


# In[ ]:


df_test, y_test, nas2 = proc_df(df_test.drop(['idhogar'], axis = 1), y_fld = None, na_dict = nas)


# In[ ]:


X_train = df_train[0:7645]
y_train = label_train[0:7645]
print (X_train.shape)
print (y_train.shape)


# In[ ]:


X_valid = df_train[7645:]
y_valid = label_train[7645:]
print (X_valid.shape)
print (y_valid.shape)


# ### splitting train and valid sets

# In[ ]:


def print_score(m):
    res = [m.score(X_train, y_train), m.score(X_valid, y_valid)]
    if hasattr(m, 'oob_score_'):res.append(m.oob_score_)
    print (res)


# In[ ]:


m = RandomForestClassifier(n_jobs = -1)
get_ipython().run_line_magic('time', 'm.fit(X_train, y_train)')
print_score(m)


# In[ ]:


m = RandomForestClassifier(n_estimators = 3, n_jobs = -1)
get_ipython().run_line_magic('time', 'm.fit(X_train, y_train)')
print_score(m)


# In[ ]:


m = RandomForestClassifier(n_estimators = 3, n_jobs = -1, min_samples_leaf=5)
get_ipython().run_line_magic('time', 'm.fit(X_train, y_train)')
print_score(m)


# In[ ]:


m = RandomForestClassifier(n_estimators = 3, n_jobs = -1, min_samples_leaf=5, oob_score=True)
get_ipython().run_line_magic('time', 'm.fit(X_train, y_train)')
print_score(m)


# In[ ]:


m = RandomForestClassifier(n_jobs = -1, n_estimators = 40, min_samples_leaf = 3, oob_score = True, max_features = 0.5)
get_ipython().run_line_magic('time', 'm.fit(X_train, y_train)')
print_score(m)


# In[ ]:


m = RandomForestClassifier(n_jobs = -1, n_estimators = 80, min_samples_leaf = 3, oob_score = True, max_features = 0.5)
get_ipython().run_line_magic('time', 'm.fit(X_train, y_train)')
print_score(m)


# In[ ]:


m = RandomForestClassifier(n_jobs = -1, n_estimators = 120, min_samples_leaf = 3, oob_score = True, max_features = 0.5)
get_ipython().run_line_magic('time', 'm.fit(X_train, y_train)')
print_score(m)


# In[ ]:


m = RandomForestClassifier(n_jobs = -1, n_estimators = 120, min_samples_leaf = 4, oob_score = True, max_features = 0.5)
get_ipython().run_line_magic('time', 'm.fit(X_train, y_train)')
print_score(m)


# ### predicting on test set

# In[ ]:


df_test['Target'] = m.predict(df_test.drop(['Id'], axis = 1))


# In[ ]:


df_test[['Id', 'Target']].to_csv('submission.csv', index=False)


# In[ ]:




