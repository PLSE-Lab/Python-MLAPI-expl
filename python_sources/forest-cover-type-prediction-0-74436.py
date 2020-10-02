#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style="darkgrid")
get_ipython().run_line_magic('matplotlib', 'inline')
from sklearn.model_selection import GridSearchCV , train_test_split , cross_val_score , StratifiedKFold
from sklearn.metrics import roc_curve, auc
import os
import warnings
warnings.filterwarnings('ignore')

print(os.listdir("../input/"))


# In[3]:


train = pd.read_csv('../input/train.csv' , index_col='Id')
test = pd.read_csv('../input/test.csv'  , index_col='Id')
labels = train.Cover_Type
train.drop('Cover_Type' , axis = 1 , inplace =True)
train.head(3)


# In[4]:


### Cover type


# ### Cover type

# In[5]:


# names =  {
#     1 : 'Spruce', 
#     2 : 'Lodgepole',
#     3 : 'Ponderosa',
#     4 : 'Cottonwood',
#     5 : 'Aspen',
#     6 : 'Douglas',
#     7 : 'Krummholz' 
# }
# train.Cover_Type = train.Cover_Type.map(names)


# In[6]:


ax = sns.countplot(x = labels)


# All the classes in this dataset are balanced.

# In[7]:


train.columns


# In[8]:


train.isna().sum()


# In[9]:


test.isna().sum()


# In[10]:


from sklearn.preprocessing import RobustScaler
scaler = RobustScaler()
train = scaler.fit_transform(train)
test = scaler.transform(test)


# In[49]:


from sklearn.ensemble import GradientBoostingClassifier
clf = GradientBoostingClassifier(learning_rate=0.65 , n_estimators= 250 ,max_depth = 9  )
cross_val_score(clf , train , labels , cv = 3)


# In[ ]:


clf.fit(train, labels)


# In[ ]:


pre = clf.predict(test)


# In[ ]:


ansdf = pd.read_csv('../input/sampleSubmission.csv')
ansdf['Cover_Type'] = pre
ansdf.to_csv('submit.csv', index = False)


# In[ ]:


ansdf.head()


# In[ ]:




