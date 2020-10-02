#!/usr/bin/env python
# coding: utf-8

# In[ ]:


get_ipython().system(' pip install fastai2')


# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# ## Time series Analysis

# In[ ]:


from fastai2.tabular.all import *


# In[ ]:


train = pd.read_pickle('/kaggle/input/fastai-v3-rossman-data-clean/train_clean')
test = pd.read_pickle('/kaggle/input/fastai-v3-rossman-data-clean/test_clean')


# In[ ]:


train.head().T


# In[ ]:


train.columns


# In[ ]:




cat_vars = ['Store', 'DayOfWeek', 'Year', 'Month', 'Day', 'StateHoliday', 'CompetitionMonthsOpen',
    'Promo2Weeks', 'StoreType', 'Assortment', 'PromoInterval', 'CompetitionOpenSinceYear', 'Promo2SinceYear',
    'State', 'Week', 'Events', 'Promo_fw', 'Promo_bw', 'StateHoliday_fw', 'StateHoliday_bw',
    'SchoolHoliday_fw', 'SchoolHoliday_bw', 'Promo', 'SchoolHoliday']

cont_vars = ['CompetitionDistance', 'Max_TemperatureC', 'Mean_TemperatureC', 'Min_TemperatureC',
   'Max_Humidity', 'Mean_Humidity', 'Min_Humidity', 'Max_Wind_SpeedKm_h', 
   'Mean_Wind_SpeedKm_h', 'CloudCover', 'trend', 'trend_DE',
   'AfterStateHoliday', 'BeforeStateHoliday']
   
dep_var = 'Sales'


# In[ ]:


train['Sales'] = np.log(train[dep_var])


# In[ ]:


procs = [FillMissing, Normalize, Categorify]


# In[ ]:


len(train), len(test)


# In[ ]:


train['Date'].min(), train['Date'].max()


# In[ ]:


test['Date'].min(), test['Date'].max()


# In[ ]:


idx = train['Date'][train['Date'] == train['Date'][len(test)]].index.max()


# In[ ]:


idx


# In[ ]:


splits = (L(range(idx, len(train))), L(range(idx)))


# In[ ]:


splits


# In[ ]:




pd.options.mode.chained_assignment=None


# In[ ]:


to = TabularPandas(train, procs, cat_vars, cont_vars, dep_var, block_y=RegressionBlock(),                       splits=splits, inplace=True, reduce_memory=True)


# In[ ]:


dls = to.dataloaders(bs=512)


# In[ ]:


dls.show_batch()


# ## Modelling
# 
# - TabularLearner

# In[ ]:


max_log_y = np.max(train['Sales'])*1.2


# In[ ]:


max_log_y


# In[ ]:


y_range = torch.tensor([0, max_log_y]); y_range


# In[ ]:


learn = tabular_learner(dls, layers=[1000, 500], ps=[0.001, 0.01],
                       embed_p=0.04, y_range=y_range, metrics=exp_rmspe,
                       loss_func=MSELossFlat())


# In[ ]:


learn.summary()


# In[ ]:


learn.lr_find()


# In[ ]:


learn.fit_one_cycle(10, 5e-3)


# ## Inferencing on the models

# In[ ]:


learn.export('myModel')


# In[ ]:


del learn


# In[ ]:


learn = load_learner('myModel')
dl = learn.dls.test_dl(test)


# In[ ]:


raw_test_preds = learn.get_preds(dl=dl)
learn.validate(dl=dl)


# In[ ]:


test_preds = np.exp(raw_test_preds[0]).numpy().T[0]
test['Sales'] = test_preds
test[['Id', "Sales"]] = test[['Id', 'Sales']].astype('int')


# In[ ]:


test[['Id', 'Sales']].to_csv('submission.csv', index=False)


# [submission](submission.csv)

# ## Permutation Importance

# In[ ]:


class PermutationImportance():
  "Calculate and plot the permutation importance"
  def __init__(self, learn:Learner, df=None, bs=None):
    "Initialize with a test dataframe, a learner, and a metric"
    self.learn = learn
    self.df = df
    bs = bs if bs is not None else learn.dls.bs
    if self.df is not None:
      self.dl = learn.dls.test_dl(self.df, bs=bs)
    else:
      self.dl = learn.dls[1]
    self.x_names = learn.dls.x_names.filter(lambda x: '_na' not in x)
    self.na = learn.dls.x_names.filter(lambda x: '_na' in x)
    self.y = dls.y_names
    self.results = self.calc_feat_importance()
    self.plot_importance(self.ord_dic_to_df(self.results))

  def measure_col(self, name:str):
    "Measures change after column shuffle"
    col = [name]
    if f'{name}_na' in self.na: col.append(name)
    orig = self.dl.items[col].values
    perm = np.random.permutation(len(orig))
    self.dl.items[col] = self.dl.items[col].values[perm]
    metric = learn.validate(dl=self.dl)[1]
    self.dl.items[col] = orig
    return metric

  def calc_feat_importance(self):
    "Calculates permutation importance by shuffling a column on a percentage scale"
    print('Getting base error')
    base_error = self.learn.validate(dl=self.dl)[1]
    self.importance = {}
    pbar = progress_bar(self.x_names)
    print('Calculating Permutation Importance')
    for col in pbar:
      self.importance[col] = self.measure_col(col)
    for key, value in self.importance.items():
      self.importance[key] = (base_error-value)/base_error #this can be adjusted
    return OrderedDict(sorted(self.importance.items(), key=lambda kv: kv[1], reverse=True))

  def ord_dic_to_df(self, dict:OrderedDict):
    return pd.DataFrame([[k, v] for k, v in dict.items()], columns=['feature', 'importance'])

  def plot_importance(self, df:pd.DataFrame, limit=20, asc=False, **kwargs):
    "Plot importance with an optional limit to how many variables shown"
    df_copy = df.copy()
    df_copy['feature'] = df_copy['feature'].str.slice(0,25)
    df_copy = df_copy.sort_values(by='importance', ascending=asc)[:limit].sort_values(by='importance', ascending=not(asc))
    ax = df_copy.plot.barh(x='feature', y='importance', sort_columns=True, **kwargs)
    for p in ax.patches:
      ax.annotate(f'{p.get_width():.4f}', ((p.get_width() * 1.005), p.get_y()  * 1.005))


# In[ ]:


res = PermutationImportance(learn, tran.iloc[:1000], bs=64)


# In[ ]:


res.importance


# In[ ]:




