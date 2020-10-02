#!/usr/bin/env python
# coding: utf-8

# # Fastai based Mixed Input Model
# My Idea here was to use the approach that succeded in the Rossman Kaggle challenge [see the fastai notebook here](https://github.com/fastai/fastai/blob/master/courses/dl1/lesson3-rossman.ipynb)
# The data loading and initial feature engineering I borrowed heavily from [YouHan Lees Kernel](https://www.kaggle.com/youhanlee/hello-elo-ensemble-will-help-you) and [Hyun woo kim](https://www.kaggle.com/chocozzz/simple-data-exploration-with-python-lb-3-764/notebook)
# I think in this challenge the key thing is the feature engineering.
# Here I start with the model and use the borrowed feature engineering as a starting point.
# I might use these embedding layers to do the feature engineering on the new and historic transactions but as of Yet I'm not sure how to do it.
# Suggestions are always welcome
# 

# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')
get_ipython().run_line_magic('reload_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')
from fastai.structured import *
from fastai.column_data import *


# In[ ]:


path='../input/'


# In[ ]:


new_transactions = pd.read_csv(f'{path}new_merchant_transactions.csv')
new_transactions.head()


# In[ ]:


new_transactions['authorized_flag'] = new_transactions['authorized_flag']=='Y'


# In[ ]:


def aggregate_new_transactions(new_trans):    
    agg_func = {
        'authorized_flag': ['sum', 'mean'],
        'merchant_id': ['nunique'],
        'city_id': ['nunique'],
        'purchase_amount': ['sum', 'median', 'max', 'min', 'std'],
        'installments': ['sum', 'median', 'max', 'min', 'std'],
        'month_lag': ['min', 'max']
        }
    agg_new_trans = new_trans.groupby(['card_id']).agg(agg_func)
    agg_new_trans.columns = ['new_' + '_'.join(col).strip() 
                           for col in agg_new_trans.columns.values]
    agg_new_trans.reset_index(inplace=True)
    
    df = (new_trans.groupby('card_id')
          .size()
          .reset_index(name='new_transactions_count'))
    
    agg_new_trans = pd.merge(df, agg_new_trans, on='card_id', how='left')
    
    return agg_new_trans

new_trans = aggregate_new_transactions(new_transactions)
new_trans.head()


# In[ ]:


del new_transactions


# In[ ]:


historical_transactions = pd.read_csv('../input/historical_transactions.csv')
historical_transactions.head()


# In[ ]:


historical_transactions['authorized_flag'] = historical_transactions['authorized_flag']=='Y'


# In[ ]:


def aggregate_historical_transactions(history):
    
    history.loc[:, 'purchase_date'] = pd.DatetimeIndex(history['purchase_date']).                                      astype(np.int64) * 1e-9
    
    agg_func = {
        'authorized_flag': ['sum', 'mean'],
        'merchant_id': ['nunique'],
        'city_id': ['nunique'],
        'purchase_amount': ['sum', 'median', 'max', 'min', 'std'],
        'installments': ['sum', 'median', 'max', 'min', 'std'],
        'purchase_date': [np.ptp],
        'month_lag': ['min', 'max']
        }
    agg_history = history.groupby(['card_id']).agg(agg_func)
    agg_history.columns = ['hist_' + '_'.join(col).strip() 
                           for col in agg_history.columns.values]
    agg_history.reset_index(inplace=True)
    
    df = (history.groupby('card_id')
          .size()
          .reset_index(name='hist_transactions_count'))
    
    agg_history = pd.merge(df, agg_history, on='card_id', how='left')
    
    return agg_history

history = aggregate_historical_transactions(historical_transactions)
history.head()


# In[ ]:


del historical_transactions


# In[ ]:


def read_data(input_file):
    df = pd.read_csv(input_file)
    add_datepart(df,'first_active_month')
    return df
train = read_data('../input/train.csv')
test = read_data('../input/test.csv')

target='target'


# In[ ]:


train = pd.merge(train, history, on='card_id', how='left')
test = pd.merge(test, history, on='card_id', how='left')

train = pd.merge(train, new_trans, on='card_id', how='left')
test = pd.merge(test, new_trans, on='card_id', how='left')


# In[ ]:


train=train.set_index('card_id')
test=test.set_index('card_id')


# In[ ]:


cat_vars = [col  for col in train.columns if('feature' in col or 'first_active_month' in col)]
contin_vars = [col  for col in train.columns if ('feature' not in col and 'first_active_month' not in col) and target not in col]


# In[ ]:


train = train[cat_vars+contin_vars+[target]].copy()
n = len(train); n


# In[ ]:


test[target] = 0
test = test[cat_vars+contin_vars+[target]].copy()


# In[ ]:


for v in cat_vars: train[v] = train[v].astype('category').cat.as_ordered()
apply_cats(test, train)


# In[ ]:


for v in contin_vars:
    train[v] = train[v].fillna(0).astype('float32')
    test[v] = test[v].fillna(0).astype('float32')


# In[ ]:


df, y, nas, mapper = proc_df(train, target, do_scale=True)
n=len(df);n


# In[ ]:


df_test, _, nas, mapper = proc_df(test, target, do_scale=True, 
                                  mapper=mapper, na_dict=nas)


# In[ ]:


train_ratio = 0.75
# train_ratio = 0.9
train_size = int(n * train_ratio); train_size
val_idx = list(range(train_size, len(df)))


# In[ ]:


model_data = ColumnarModelData.from_data_frame('.', val_idx, df, y.astype(np.float32), cat_flds=cat_vars, bs=64,
                                       test_df=df_test)


# In[ ]:


cat_sz = [(c, len(train[c].cat.categories)+1) for c in cat_vars]
emb_szs = [(c, min(50, (c+1)//2)) for _,c in cat_sz]
y_range=(np.min(y),np.max(y))


# In[ ]:


def rmse(y_pred, targ):
    var = np.square(targ - y_pred)
    return math.sqrt(var.mean())


# In[ ]:


get_ipython().run_line_magic('pinfo2', 'model_data.get_learner')


# In[ ]:


learn = model_data.get_learner(emb_szs, len(df.columns)-len(cat_vars),
                   0.20, 1, [1000,500], [0.2,0.2], y_range=y_range,metrics=[rmse])


# In[ ]:


learn.lr_find()
learn.sched.plot()


# In[ ]:


learn.fit(1e-3, 1,cycle_len=5,use_clr_beta=(10,10,0.95,0.85))
learn.sched.plot_loss()


# In[ ]:


learn.sched.plot_lr()


# In[ ]:


x,y=learn.predict_with_targs()
rmse(x,y)


# In[ ]:


preds=learn.predict(is_test=True)


# In[ ]:


submission=pd.read_csv(f'{path}/sample_submission.csv',index_col='card_id')
submission.loc[df_test.index]=preds


# In[ ]:


submission.to_csv('submission.csv')


# In[ ]:


get_ipython().system('head submission.csv')


# In[ ]:




