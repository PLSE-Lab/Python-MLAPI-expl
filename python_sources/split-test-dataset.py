#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)


import os
print(os.listdir("../input"))


# In[ ]:


#Load data
train_df = pd.read_csv('../input/train.csv')
test_df = pd.read_csv('../input/test.csv')


# In[ ]:


train_features = train_df.drop(['target','ID_code'], axis = 1)
test_features = test_df.drop(['ID_code'],axis = 1)
train_target = train_df['target']


# In[ ]:


train_all = pd.concat([train_features,test_features], axis = 0)


# In[ ]:


for f in train_all.columns[0:200]:
    train_all[f+'duplicate'] = train_all.duplicated(f,False).astype(int)


# In[ ]:


train_features = train_all.iloc[:len(train_target)]
test_features = train_all.iloc[len(train_target):len(train_all)]


# In[ ]:


test_features['count_total_all']=test_features[test_features.columns[200:400]].sum(axis=1)
fake_data = test_features[test_features['count_total_all']==200]
real_data = test_features[test_features['count_total_all']!=200]


# In[ ]:


np.save('index_of_fake_data.npy',np.array(fake_data.index))
np.save('index_of_real_data.npy',np.array(real_data.index))


# In[ ]:




