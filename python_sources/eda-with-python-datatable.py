#!/usr/bin/env python
# coding: utf-8

# One of the main issues in this competition is the size of the dataset. Pandas crashes when attempting to load the entire train and test datasets at once. [One of the kernels has been able to read the entire train dataset using dask](https://www.kaggle.com/ashishpatel26/how-to-handle-this-big-dataset-dask-vs-pandas). In this kernels we'll use [Python datatable package](https://github.com/h2oai/datatable) to load the entire train and test datasets, and do some simple EDA on them. Python datatable is still in early alpha stage and is under very active curent development. It is designed from ground up for big datasets and with efficiency and speed in mind. It is closely related to [R's data.table](https://github.com/Rdatatable/data.table) and attempts to mimic its core algorithams and API. 

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 
from datetime import datetime
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)


# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
import gc
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# Unfortunately datatable is not currently available in Kaggle Docker image. My attempts to install it via Kaggle kernel package installation API have failed, but I have been able to load it from the following pre-made wheel. (A huge shoutout to [Olivier](https://www.kaggle.com/ogrellier) for his help with this.)

# In[ ]:


get_ipython().system('pip install https://s3.amazonaws.com/artifacts.h2o.ai/releases/ai/h2o/pydatatable/0.7.0.dev490/x86_64-centos7/datatable-0.7.0.dev490-cp36-cp36m-linux_x86_64.whl')


# Now let's import datatable

# In[ ]:


from sklearn.metrics import log_loss, roc_auc_score
from datetime import datetime
import datatable as dt
from datatable.models import Ftrl


# Now let's load the train dataset:

# In[ ]:


get_ipython().run_cell_magic('time', '', "train = dt.fread('../input/train.csv')")


# And test:

# In[ ]:


get_ipython().run_cell_magic('time', '', "test = dt.fread('../input/test.csv')")


# We were able to load all of the train and test datasets, and pretty much exhausted all of kernel's 17.2 GB of RAM. But we did it!

# Let's take a look at the train:

# 

# In[ ]:


train.head()


# In[ ]:


train.shape


# And test:

# In[ ]:


test.head()


# In[ ]:


test.shape


# Look at the number of unique values in the two datasets:

# In[ ]:


train.nunique()


# In[ ]:


test.nunique()


# In[ ]:


train[:, 'EngineVersion'].nunique1()


# In[ ]:


train_unique = dt.unique(train[:, 'EngineVersion']).to_list()[0]
len(train_unique)


# In[ ]:


test_unique = dt.unique(test[:, 'EngineVersion']).to_list()[0]
len(test_unique)


# In[ ]:


intersection = list(set(train_unique) & set(test_unique))
len(intersection)


# We see there are only 66 values that overlap in the train and test for this feature.
# 
# Let's see what are the names of the features in the dataset:

# In[ ]:


train.names


# And their types:

# In[ ]:


train.ltypes


# Next, we are going to try to fit an Ftrl model on the train set. Here we will adopt [Olivier's great discussion topic](https://www.kaggle.com/c/microsoft-malware-prediction/discussion/75478). First, let's replace all the missing values.

# In[ ]:


'''%%time
for name in test.names:
    if test[:, name].ltypes[0] == dt.ltype.str:
        train.replace(None, '-1')
        test.replace(None, '-1')
    elif test[:, name].ltypes[0] == dt.ltype.int:
        train.replace(None, -1)
        test.replace(None, -1)
    elif test[:, name].ltypes[0] == dt.ltype.bool:
        train.replace(None, 0)
        test.replace(None, 0)
    elif test[:, name].ltypes[0] == dt.ltype.real:
        train.replace(None, -1.0)
        test.replace(None, -1.0)'''


# Next, we'll factorize all the string columns. Unfortunately, datatabel still doesn't handle this natively, so we'll have to use the Pandas crutch.

# In[ ]:


get_ipython().run_cell_magic('time', '', "for f in train.names:\n    if f not in ['MachineIdentifier', 'HasDetections']:\n        if train[:, f].ltypes[0] == dt.ltype.str:\n            print('factorizing %s' % f)\n            col_f = pd.concat([train[:, f].to_pandas(), test[:, f].to_pandas()], ignore_index=True)\n            encoding = col_f.groupby(f).size()\n            encoding = encoding/len(col_f)\n            column = col_f[f].map(encoding).values.flatten()\n            del col_f, encoding\n            gc.collect()\n            train[:, f] = dt.Frame(column[:8921483])\n            test[:, f] = dt.Frame(column[8921483:])\n            del column\n            gc.collect()")


# In[ ]:


train[:, f]


# In[ ]:


train.head()


# In[ ]:


test.head()


# Now, let's fit the model:

# In[ ]:


features = [f for f in train.names if f not in ['HasDetections']]
ftrl = Ftrl(nepochs=2, interactions=True)


# In[ ]:


get_ipython().run_cell_magic('time', '', "print('Start Fitting on   ', train.shape, ' @ ', datetime.now())\nftrl.fit(train[:, features], train[:, 'HasDetections'])\nprint('Fitted complete on ', train.shape, ' @ ', datetime.now())  \nprint('Current loss : %.6f' \n          % log_loss(np.array(train[:, 'HasDetections'])[:, 0],  \n                             np.array(ftrl.predict(train[:, features]))))")


# In[ ]:


print('Current AUC : %.6f' 
          % roc_auc_score(np.array(train[:, 'HasDetections'])[:, 0],  
                             np.array(ftrl.predict(train[:, features]))))


# In[ ]:


preds1 = np.array(ftrl.predict(test[:, features]))
preds1 = preds1.flatten()


# In[ ]:


ftrl = Ftrl(nepochs=20, interactions=False)
ftrl.fit(train[:, features], train[:, 'HasDetections'])
preds2 = np.array(ftrl.predict(test[:, features]))
preds2 = preds2.flatten()


# In[ ]:


np.save('preds1', preds1)
np.save('preds2', preds2)


# In[ ]:


sample_submission = pd.read_csv('../input/sample_submission.csv')


# In[ ]:


sample_submission['HasDetections'] = 0.6*preds1+0.4*preds2


# In[ ]:


sample_submission.to_csv('datatable_ftrl_submission.csv', index=False)


# To be continued ...
