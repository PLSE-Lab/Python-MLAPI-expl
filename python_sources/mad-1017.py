#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#matplotlib inline
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import os
from sklearn.preprocessing import LabelEncoder
from sklearn.cross_validation import KFold
from sklearn.metrics import log_loss

gatrain = pd.read_csv('../input/gender_age_train.csv')
gatest = pd.read_csv('../input/gender_age_test.csv')
gatrain.head(3)

gatrain.shape[0] - gatrain.device_id.nunique()
gatrain.group.value_counts().sort_index(ascending=False).plot(kind='barh')
c = gatrain.groupby(['age','gender']).size().unstack().reindex(index=np.arange(gatrain.age.min(), gatrain.age.max()+1)).fillna(0)
ax1, ax2 = c.plot(kind='bar',figsize=(12,6),subplots=True);
ax1.vlines(np.array([23,26,28,32,42])-0.5,0,1800,alpha=0.5,linewidth=1,color='r')
ax2.vlines(np.array([22,26,28,31,38])-0.5,0,3000,alpha=0.5,linewidth=1,color='r')
phone = pd.read_csv('../input/phone_brand_device_model.csv',encoding='utf-8')
phone.head(3)
dup = phone.groupby('device_id').size()
dup = dup[dup>1]
dup.shape


dup = phone.loc[phone.device_id.isin(dup.index)]
first = dup.groupby('device_id').first()
last = dup.groupby('device_id').last()
diff = (first != last).sum(axis=1).nonzero()
pd.concat((first.iloc[diff], last.iloc[diff]),axis=1)


# In[ ]:


At this scale, you can actually see the finite resolution of the lat/lon values.
trtrt
### Now we can try plotting the same for male/female, and show average age per grid


# At this scale, you can actually see the finite resolution of the lat/lon values.
# 
# ### AAAAAAA
