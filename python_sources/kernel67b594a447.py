#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
import h2o
from h2o.automl import H2OAutoML
from h2o.estimators.deeplearning import H2ODeepLearningEstimator


# In[ ]:


h2o.init(max_mem_size='16G')


# 

# In[ ]:


train_path = '../input/train.csv'
test_path = '../input/test.csv'


# In[ ]:


col_types = {'target': 'enum'}


# In[ ]:


train = h2o.import_file(path=train_path, col_types=col_types)


# In[ ]:


test = h2o.import_file(path=test_path)


# In[ ]:


y = 'target'
X = [name for name in train.columns if name not in ['ID_code', y]]


# In[ ]:


train[y] = train[y].asfactor()


# In[ ]:


model = H2OAutoML(max_models=50,
                  max_runtime_secs =7200,
                seed=12345)
model.train(x=X, y=y, training_frame=train)


# In[ ]:


result = model.leader.predict(test)


# In[ ]:


sub = test.cbind(result[0])


# In[ ]:


sub = sub[['ID_code','predict']]
sub = sub.rename(columns={'predict':'target'})


# In[ ]:


sub = sub.as_data_frame()


# In[ ]:


sub.to_csv('submission.csv',index=False)


# In[ ]:


h2o.cluster().shutdown(prompt=True)


# In[ ]:


y

