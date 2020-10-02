#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# In[ ]:


import h2o
from h2o.automl import H2OAutoML
h2o.init(max_mem_size='16G')


# In[ ]:


train = h2o.import_file('../input/home-data-for-ml-course/train.csv')
test = h2o.import_file('../input/home-data-for-ml-course/test.csv')


# In[ ]:


train.tail(2)


# In[ ]:


test.head(2)


# In[ ]:


print("Train shape:", train.shape, '\nTest shape:', test.shape)


# In[ ]:


y = "SalePrice"
x = train.columns
x.remove('Id')
x.remove(y)
test = test.drop(['Id'])


# In[ ]:


aml = H2OAutoML()
aml.train(x=x, y=y, training_frame=train)


# In[ ]:


lb = aml.leaderboard
lb.head(rows=lb.nrows)


# In[ ]:


aml.leader


# In[ ]:


preds = aml.predict(test)


# In[ ]:


preds.head()


# In[ ]:


submission = pd.read_csv('../input/home-data-for-ml-course/sample_submission.csv')
submission.head()


# In[ ]:


submission[y] = preds.as_data_frame().values
submission


# In[ ]:


submission.to_csv('submission.csv', index=False)


# In[ ]:


pd.read_csv('submission.csv')

