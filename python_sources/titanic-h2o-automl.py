#!/usr/bin/env python
# coding: utf-8

# ### References::  
# * Bojan Tunguz's https://www.kaggle.com/tunguz/elo-with-h2o-automl 
# * The H2O AutoML tutorial (https://github.com/h2oai/h2o-tutorials/blob/master/h2o-world-2017/automl/Python/automl_binary_classification_product_backorders.ipynb)

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[ ]:


import h2o
from h2o.automl import H2OAutoML

h2o.init(max_mem_size='4G')


# In[ ]:


get_ipython().run_cell_magic('time', '', 'train = h2o.import_file("../input/train.csv")\ntest = h2o.import_file("../input/test.csv")')


# In[ ]:


train.describe()


# In[ ]:


x = train.columns
y = "Survived"


# In[ ]:


# For binary classification, response should be a factor
train[y] = train[y].asfactor()
x.remove(y)


# In[ ]:


get_ipython().run_cell_magic('time', '', '# Run AutoML for 10 base models (limited to 1 hour max runtime by default)\naml = H2OAutoML(max_models=10, seed=42, max_runtime_secs=7200)\naml.train(x=x, y=y, training_frame=train)')


# In[ ]:


lb = aml.leaderboard
lb.head(rows=lb.nrows)  # Print all rows instead of default (10 rows)


# ### So the best model gets an AUC of 0.876 (and this is without any parameter tuning)

# In[ ]:


aml.leader # Best model


# In[ ]:


# If you need to generate predictions on a test set, you can make
# predictions directly on the `"H2OAutoML"` object, or on the leader
# model object directly

preds = aml.predict(test)


# In[ ]:


predictions = preds[0].as_data_frame().values.flatten()


# In[ ]:


sample_submission = pd.read_csv('../input/gender_submission.csv')
sample_submission['Survived'] = predictions
sample_submission.to_csv('h2O_titanic_1.csv', index=False)

