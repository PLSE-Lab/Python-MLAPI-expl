#!/usr/bin/env python
# coding: utf-8

# In[ ]:


get_ipython().system(' pip install fastai2 ')
get_ipython().system('pip install bayesian-optimization -q')


# In[ ]:


from fastai2.tabular.all import *
from bayes_opt import BayesianOptimization


# In[ ]:


path = untar_data(URLs.ADULT_SAMPLE)
df = pd.read_csv(path/'adult.csv')


# 
# Bayesian Optimization
# 
# When working with BayesianOpimization, everything needs to be in a fit_with function that accepts our tuned parameters, and does whatever we require of it:
# 

# In[ ]:


def fit_with(lr:float, wd:float, dp:float):
    learn = tabular_learner


# In[ ]:





# In[ ]:


def fit_with(lr:float, wd:float, dp:float, n_layers:float, layer_1:float, layer_2:float, layer_3:float):

  print(lr, wd, dp)
  if int(n_layers) == 2:
    layers = [int(layer_1), int(layer_2)]
  elif int(n_layers) == 3:
    layers = [int(layer_1), int(layer_2), int(layer_3)]
  else:
    layers = [int(layer_1)]

  learn = tabular_learner(dls, layers=layers, metrics=accuracy, embed_p=float(dp), wd=float(wd))

  with learn.no_bar() and learn.no_logging():
    learn.fit(5, lr=float(lr))

  acc = float(learn.validate()[1])

  return acc


# In[ ]:


cat_names = ['workclass', 'education', 'marital-status', 'occupation', 'relationship', 'race']
cont_names = ['age', 'fnlwgt', 'education-num']
procs = [Categorify, FillMissing, Normalize]
y_names = 'salary'
block_y = CategoryBlock()
splits = RandomSplitter()(range_of(df))


# In[ ]:




to = TabularPandas(df, procs=procs, cat_names=cat_names, cont_names=cont_names,
                   y_names=y_names, block_y=block_y, splits=splits)


# In[ ]:


dls = to.dataloaders(bs=512)


# In[ ]:


hps = {'lr': (1e-05, 1e-01),
      'wd': (4e-4, 0.4),
      'dp': (0.01, 0.5),
       'n_layers': (1,3),
       'layer_1': (50, 200),
       'layer_2': (100, 1000),
       'layer_3': (200, 2000)}


# In[ ]:


optim = BayesianOptimization(
    f = fit_with, # our fit function
    pbounds = hps, # our hyper parameters to tune
    verbose = 2, # 1 prints out when a maximum is observed, 0 for silent
    random_state=1
)


# In[ ]:


get_ipython().run_line_magic('time', 'optim.maximize(n_iter=10)')


# In[ ]:


print(optim.max)


# In[ ]:




