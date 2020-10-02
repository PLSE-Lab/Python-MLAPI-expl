#!/usr/bin/env python
# coding: utf-8

# 

# [H2O](http://docs.h2o.ai/h2o/latest-stable/h2o-py/docs/intro.html) is an open source Java-based software for data modeling and general computing, created and maintained by [H2O.ai](https://www.h2o.ai/). The H2O software is many things, but the primary purpose of H2O is as a distributed (many machines), parallel (many CPUs), in memory (several hundred GBs Xmx) processing engine.
# 
# H2O supports [Deep Learning](http://docs.h2o.ai/h2o/latest-stable/h2o-docs/data-science/deep-learning.html). H2O's Deep Learning is based on a multi-layer feedforward artificial neural network that is trained with stochastic gradient descent using back-propagation. The network can contain a large number of hidden layers consisting of neurons with tanh, rectifier, and maxout activation functions. Advanced features such as adaptive learning rate, rate annealing, momentum training, dropout, L1 or L2 regularization, checkpointing, and grid search enable high predictive accuracy. Each compute node trains a copy of the global model parameters on its local data with multi-threading (asynchronously) and contributes periodically to the global model via model averaging across the network. 

# In[ ]:


## This Python 3 environment comes with many helpful analytics libraries installed
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


# In[ ]:


import h2o
print(h2o.__version__)
from h2o.estimators.deeplearning import H2ODeepLearningEstimator

h2o.init(max_mem_size='16G')


# In[ ]:


get_ipython().run_cell_magic('time', '', 'train = h2o.import_file("../input/digit-recognizer/train.csv")\ntest = h2o.import_file("../input/digit-recognizer/test.csv")')


# In[ ]:


train.head()


# In[ ]:


x = train.columns[1:]
y = 'label'
# For binary classification, response should be a factor
train[y] = train[y].asfactor()


# In[ ]:


dl = H2ODeepLearningEstimator(input_dropout_ratio = 0.2, nfolds=3)
dl.train(x=x, y=y, training_frame=train)


# In[ ]:


h2o.download_pojo(dl)  # production code in Java, more at http://docs.h2o.ai/h2o/latest-stable/h2o-docs/productionizing.html#about-pojos-and-mojos


# In[ ]:


dl.model_performance(xval=True)


# In[ ]:


preds = dl.predict(test)
preds['p1'].as_data_frame().values.shape


# In[ ]:


preds


# In[ ]:


sample_submission = pd.read_csv('../input/digit-recognizer/sample_submission.csv')
sample_submission.shape


# In[ ]:


sample_submission['Label'] = preds['predict'].as_data_frame().values
sample_submission.to_csv('h2o_dl_submission_1.csv', index=False)
sample_submission.head()

