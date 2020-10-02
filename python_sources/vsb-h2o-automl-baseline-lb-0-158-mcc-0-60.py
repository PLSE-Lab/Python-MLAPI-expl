#!/usr/bin/env python
# coding: utf-8

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


# ### cf. https://www.kaggle.com/bluexleoxgreen/simple-feature-lightgbm-baseline

# In[ ]:


import h2o
from h2o.automl import H2OAutoML

h2o.init(max_mem_size='15G')


# In[ ]:


get_ipython().run_cell_magic('time', '', 'train = h2o.import_file("../input/vsb-train-test-1/train_new.csv")\ntest = h2o.import_file("../input/vsb-train-test-1/test_new.csv")')


# In[ ]:


train = train.drop(['id_measurement','phase'])
test = test.drop(['id_measurement','phase'])


# In[ ]:


train.describe()


# In[ ]:


x = train.columns
y = "target"


# In[ ]:


# For binary classification, response should be a factor
train[y] = train[y].asfactor()
x.remove(y)


# In[ ]:


get_ipython().run_cell_magic('time', '', '# Run AutoML for 50 base models (limited to 1 hour max runtime by default)\naml = H2OAutoML(max_models=50, seed=42, max_runtime_secs=7200)\naml.train(x=x, y=y, training_frame=train)')


# In[ ]:


lb = aml.leaderboard
lb.head(rows=lb.nrows) 


# In[ ]:


aml.leader # Best model


# In[ ]:


preds = aml.predict(test)


# In[ ]:


predictions = preds[0].as_data_frame().values.flatten()


# In[ ]:


x_filename = pd.read_csv('../input/vsb-train-test-1/test_new.csv')['signal_id']
sub_df = pd.DataFrame({"signal_id":x_filename})
sub_df["target"] = pd.Series(predictions).round()
sub_df['signal_id'] = sub_df['signal_id'].astype(np.int64)
sub_df['target'] = sub_df['target'].astype(np.int64)
sub_df.to_csv("submission.csv", index=False)
sub_df

