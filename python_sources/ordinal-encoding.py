#!/usr/bin/env python
# coding: utf-8

# Is the second character of ord_5 junk?

# In[ ]:


import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import roc_auc_score, log_loss
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


ls ../input


# In[ ]:


def OrdinalConverter(d):
    a1 = ord(d[:1])-65
    if(a1>26):
        a1-=6
    if(len(d)==1):
        return a1
    a2 = ord(d[1:2])-65
    if(a2>26):
        a2-=6
    return a1*52+a2


# In[ ]:


df_train = pd.read_csv('../input/cat-in-the-dat-ii/train.csv')


# In[ ]:


df_train.ord_5.unique()


# In[ ]:


df_train = df_train.set_index('id')
y_train = df_train.target
del df_train['target']


# In[ ]:


ord_0_mapping = {1 : 0, 2 : 1, 3 : 2}
ord_1_mapping = {'Novice' : 0, 'Contributor' : 1, 'Expert' : 2, 'Master': 3, 'Grandmaster': 4}
ord_2_mapping = { 'Freezing': 0, 'Cold': 1, 'Warm' : 2, 'Hot': 3, 'Boiling Hot' : 4, 'Lava Hot' : 5}
df_train['real_ord_0'] = df_train.loc[df_train.ord_0.notnull(), 'ord_0'].map(ord_0_mapping)
df_train['real_ord_1'] = df_train.loc[df_train.ord_1.notnull(), 'ord_1'].map(ord_1_mapping)
df_train['real_ord_2'] = df_train.loc[df_train.ord_2.notnull(), 'ord_2'].map(ord_2_mapping)
otherordinals = ['ord_3','ord_4','ord_5']

for c in otherordinals:
    print(c)
    df_train['real_'+c] = df_train[[c]].apply(lambda a: OrdinalConverter(a[c]) if not pd.isnull(a[c]) else np.nan,axis=1)



# In[ ]:


df_train['real_right_'+c] = df_train[[c]].apply(lambda a: OrdinalConverter(a[c][:1]) if not pd.isnull(a[c]) else np.nan,axis=1)
df_train['real_left_'+c] = df_train[[c]].apply(lambda a: OrdinalConverter(a[c][1:]) if not pd.isnull(a[c]) else np.nan,axis=1)


# In[ ]:


realordinals = [c for c in df_train.columns if 'real_' in c]
for c in realordinals:
    print(c)
    x = pd.DataFrame()
    x['target'] = y_train[~df_train[c].isnull()]
    x[c] = df_train[~df_train[c].isnull()][c]
    y = x.groupby(c).target.mean().reset_index(drop=False)

    plt.scatter(y[c],y.target)
    plt.show()

