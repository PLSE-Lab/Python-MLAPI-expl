#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
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


df_train = pd.read_csv("/kaggle/input/cat-in-the-dat/train.csv")
df_test = pd.read_csv("/kaggle/input/cat-in-the-dat/test.csv")


# In[ ]:


df_train.isnull().sum()


# In[ ]:


df_train.columns


# In[ ]:


df_test.columns


# In[ ]:


df_train.head(10)


# In[ ]:


from sklearn.preprocessing import LabelEncoder
lb = LabelEncoder()
df_train['bin_3'] = lb.fit_transform(df_train['bin_3'])
df_test['bin_3'] = lb.fit_transform(df_test['bin_3'])

df_train['bin_4'] = lb.fit_transform(df_train['bin_4'])
df_test['bin_4'] = lb.fit_transform(df_test['bin_4'])

df_train['nom_0'] = lb.fit_transform(df_train['nom_0'])
df_test['nom_0'] = lb.fit_transform(df_test['nom_0'])

df_train['nom_1'] = lb.fit_transform(df_train['nom_1'])
df_test['nom_1'] = lb.fit_transform(df_test['nom_1'])

df_train['nom_2'] = lb.fit_transform(df_train['nom_2'])
df_test['nom_2'] = lb.fit_transform(df_test['nom_2'])

df_train['nom_3'] = lb.fit_transform(df_train['nom_3'])
df_test['nom_3'] = lb.fit_transform(df_test['nom_3'])

df_train['nom_4'] = lb.fit_transform(df_train['nom_4'])
df_test['nom_4'] = lb.fit_transform(df_test['nom_4'])


# **Trying out Frequency Encoding for columns nom5 to nom9**

# In[ ]:


encoding = df_train.groupby("nom_5").size()
encoding = encoding/len(df_train)
df_train['nom_5'] = df_train.nom_5.map(encoding)


# In[ ]:


df_train['nom_5']


# In[ ]:


encoding = df_train.groupby("nom_6").size()
encoding = encoding/len(df_train)
df_train['nom_6'] = df_train.nom_6.map(encoding)

encoding = df_train.groupby("nom_7").size()
encoding = encoding/len(df_train)
df_train['nom_7'] = df_train.nom_7.map(encoding)

encoding = df_train.groupby("nom_8").size()
encoding = encoding/len(df_train)
df_train['nom_8'] = df_train.nom_8.map(encoding)

encoding = df_train.groupby("nom_9").size()
encoding = encoding/len(df_train)
df_train['nom_9'] = df_train.nom_9.map(encoding)

encoding = df_test.groupby("nom_5").size()
encoding = encoding/len(df_test)
df_test['nom_5'] = df_test.nom_5.map(encoding)

encoding = df_test.groupby("nom_6").size()
encoding = encoding/len(df_test)
df_test['nom_6'] = df_test.nom_6.map(encoding)

encoding = df_test.groupby("nom_7").size()
encoding = encoding/len(df_test)
df_test['nom_7'] = df_test.nom_7.map(encoding)

encoding = df_test.groupby("nom_8").size()
encoding = encoding/len(df_test)
df_test['nom_8'] = df_test.nom_8.map(encoding)

encoding = df_test.groupby("nom_9").size()
encoding = encoding/len(df_test)
df_test['nom_9'] = df_test.nom_9.map(encoding)


# In[ ]:


df_train.head()


# In[ ]:


df_train['nom_5'].value_counts()


# In[ ]:


df_train['ord_4'].nunique()


# In[ ]:


df_train['ord_0'] = lb.fit_transform(df_train['ord_0'])
df_test['ord_0'] = lb.fit_transform(df_test['ord_0'])


# In[ ]:


df_train['ord_1'] = lb.fit_transform(df_train['ord_1'])
df_test['ord_1'] = lb.fit_transform(df_test['ord_1'])


# In[ ]:


df_train['ord_2'] = lb.fit_transform(df_train['ord_2'])
df_test['ord_2'] = lb.fit_transform(df_test['ord_2'])


# In[ ]:


df_train['ord_3'] = lb.fit_transform(df_train['ord_3'])
df_test['ord_3'] = lb.fit_transform(df_test['ord_3'])


# In[ ]:


encoding = df_train.groupby("ord_4").size()
encoding = encoding/len(df_train)
df_train['ord_4'] = df_train.ord_4.map(encoding)


# In[ ]:


encoding = df_train.groupby("ord_5").size()
encoding = encoding/len(df_train)
df_train['ord_5'] = df_train.ord_5.map(encoding)


# In[ ]:


encoding = df_test.groupby("ord_4").size()
encoding = encoding/len(df_test)
df_test['ord_4'] = df_test.ord_4.map(encoding)


# In[ ]:


encoding = df_test.groupby("ord_5").size()
encoding = encoding/len(df_test)
df_test['ord_5'] = df_test.ord_5.map(encoding)


# **Label encoding on day and month**

# In[ ]:


df_train['day'] = lb.fit_transform(df_train['day'])
df_test['day'] = lb.fit_transform(df_test['day'])


# In[ ]:


df_train['month'] = lb.fit_transform(df_train['month'])
df_test['month'] = lb.fit_transform(df_test['month'])


# In[ ]:


X = df_train


# In[ ]:


X


# In[ ]:


y = X.target


# In[ ]:


X.drop("target",axis = 1, inplace = True)


# In[ ]:


from sklearn.ensemble import RandomForestClassifier
gbc = RandomForestClassifier()
gbc.fit(X,y)


# In[ ]:


X_test = df_test


# In[ ]:


X_test


# In[ ]:


pred = dt.predict(X_test)


# In[ ]:


pred


# In[ ]:


pred.shape


# In[ ]:


sample_submission = pd.read_csv("/kaggle/input/cat-in-the-dat/sample_submission.csv")


# In[ ]:


sample_submission.drop("target",axis = 1, inplace = True)


# In[ ]:


sample_submission['target'] = pd.DataFrame(pred)


# In[ ]:


sample_submission


# In[ ]:


sample_submission.head(50)


# In[ ]:


sample_submission['target'].value_counts()


# In[ ]:


sample_submission.to_csv("submission_saturday_XG.csv",index = False)


# In[ ]:


import os
os.getcwd()


# In[ ]:


sample_submission.info()


# In[ ]:




