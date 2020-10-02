#!/usr/bin/env python
# coding: utf-8

# The time column was neglected while creating this kernel

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import time
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))


# Any results you write to the current directory are saved as output.


# In[ ]:


train_data=pd.read_csv("../input/train.csv")
test_data=pd.read_csv("../input/test.csv")


# In[ ]:


train_data.head()


# In[ ]:


test_data.head()


# In[ ]:


train_data['store_and_fwd_flag'].value_counts()


# In[ ]:


mapping = {'N':0, 'Y':1}
train_data = train_data.replace({'store_and_fwd_flag':mapping})


# In[ ]:


test_data=test_data.replace({'store_and_fwd_flag':mapping})


# In[ ]:


train_data=train_data.drop('pickup_datetime',1)


# In[ ]:


test_data=test_data.drop('pickup_datetime',1)
train_data=train_data.drop('dropoff_datetime',1)
train_data=train_data.drop('id',1)


# In[ ]:


sub=test_data['id']


# In[ ]:


test_data=test_data.drop('id',1)


# In[ ]:


train_data['vendor_id'].value_counts()


# In[ ]:


train_data.head()


# In[ ]:


test_data.head()


# In[ ]:


target=train_data['trip_duration']


# In[ ]:


train_data=train_data.drop('trip_duration',1)


# In[ ]:


from sklearn.ensemble import RandomForestRegressor


# In[ ]:


reg=RandomForestRegressor()


# In[ ]:


reg.fit(train_data,target)


# In[ ]:


predictions=reg.predict(test_data)


# In[ ]:


submission=pd.DataFrame({'id':sub,'trip_duration':predictions})


# In[ ]:


submission.head()


# In[ ]:


submission.to_csv('subRandomForsts.csv',index=False)


# In[ ]:




