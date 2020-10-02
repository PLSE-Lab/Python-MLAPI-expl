#!/usr/bin/env python
# coding: utf-8

# ### Import packages:

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


# ### Initialize H2O cluster:

# In[ ]:


h2o.init(max_mem_size='14G')


# ### Import files:

# In[ ]:


train = h2o.import_file('../input/learn-together/train.csv')
test = h2o.import_file('../input/learn-together/test.csv')


# In[ ]:


display(train.head(2))
print(f'Train shape: {train.shape}')

display(test.head(2))
print(f'Test shape: {test.shape}')


# ### Drop id col:

# In[ ]:


train = train.drop('Id')
display(train.head(1))
print(f"Train shape: {train.shape}")


# In[ ]:


df_id = test['Id']
display(test.head(1))
print(f'Test shape: {test.shape}')


# ### Set predictors as x and target as y (and asfactor() because it's an int):

# In[ ]:


test = test.drop('Id')
x = train.columns
x.remove('Cover_Type')
y = 'Cover_Type'

train[y] = train[y].asfactor()


# ### Create train/valid set ?

# In[ ]:


# df = train.split_frame(ratios=0.8, seed=8)
# df_train = df[0]
# df_valid = df[1]


# ### Create model and train:

# In[ ]:


aml = H2OAutoML(max_runtime_secs=22000, seed=8)
aml.train(x=x, y=y, training_frame=train)


# ### Display leaderboard:

# In[ ]:


lb = aml.leaderboard
lb.head(rows=lb.nrows)


# In[ ]:


aml.leader


# ### Predict on test set:

# In[ ]:


preds = aml.predict(test)


# In[ ]:


preds.head(2)


# ### Submission:
# 
# #### I tried to add my predictions as usual with the sample_submission.csv file and then the as_data_frame().values command but it didn't worked I had some errors when submitting the file, it was saying that I had 4 cols (?) while there were only two
# 
# so i borrowed this part of code from https://www.kaggle.com/cgurkan/classify-forest-type-using-h2o-automl
# thanks to him

# In[ ]:


submission = pd.DataFrame({'Id': df_id.as_data_frame().squeeze(),
                       'Cover_Type': preds['predict'].as_data_frame().squeeze()})

submission.to_csv('submission.csv', index=False)

