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


# importing libraries

get_ipython().run_line_magic('matplotlib', 'inline')
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.ensemble import IsolationForest
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


# In[ ]:


# defining the column names

# specifing the data types prior which reduce the memory - by default allocated at the time of creation of data frame
dtypes = {
"duration": np.int8,
"protocol_type": np.object,
"service": np.object,
"flag": np.object,
"src_bytes":  np.int8,
"dst_bytes":  np.int8,
"land": np.int8,
"wrong_fragment":  np.int8,
"urgent": np.int8,
"hot": np.int8,
"m_failed_logins":  np.int8,
"logged_in":  np.int8,
"num_compromised":  np.int8,
"root_shell":  np.int8,
"su_attempted":  np.int8,
"num_root": np.int8,
"num_file_creations":  np.int8,
"num_shells":  np.int8,
"num_access_files":  np.int8,
"num_outbound_cmds":  np.int8,
"is_host_login":  np.int8,
"is_guest_login":  np.int8,
"count": np.int8,
"srv_count":  np.int8,
"serror_rate": np.float16,
"srv_serror_rate": np.float16,
"rerror_rate": np.float16,
"srv_rerror_rate": np.float16,
"same_srv_rate": np.float16,
"diff_srv_rate": np.float16,
"srv_diff_host_rate": np.float16,
"dst_host_count":  np.int8,
"dst_host_srv_count":  np.int8,
"dst_host_same_srv_rate": np.float16,
"dst_host_diff_srv_rate": np.float16,
"dst_host_same_src_port_rate": np.float16,
"dst_host_srv_diff_host_rate": np.float16,
"dst_host_serror_rate": np.float16,
"dst_host_srv_serror_rate": np.float16,
"dst_host_rerror_rate": np.float16,
"dst_host_srv_rerror_rate": np.float16,
"label": np.object
}

columns = ["duration","protocol_type","service","flag","src_bytes","dst_bytes","land","wrong_fragment","urgent","hot","m_failed_logins",
"logged_in", "num_compromised","root_shell","su_attempted","num_root","num_file_creations","num_shells","num_access_files",
"num_outbound_cmds","is_host_login","is_guest_login","count","srv_count","serror_rate","srv_serror_rate","rerror_rate","srv_rerror_rate",
"same_srv_rate","diff_srv_rate","srv_diff_host_rate","dst_host_count","dst_host_srv_count","dst_host_same_srv_rate","dst_host_diff_srv_rate",
"dst_host_same_src_port_rate", "dst_host_srv_diff_host_rate","dst_host_serror_rate","dst_host_srv_serror_rate","dst_host_rerror_rate",
"dst_host_srv_rerror_rate","label"]

df = pd.read_csv("/kaggle/input/kdd-cup-1999-data/kddcup.data.corrected", sep=",", names=columns, dtype=dtypes, index_col=None)


# In[ ]:


df.head()


# In[ ]:


df.describe()


# In[ ]:


# metadata information about the data

df.info()


# In[ ]:


# shape of the data

df.shape


# In[ ]:


# plotting the service and checking it's distribution

plt.figure(figsize=(18, 6))
df['service'].value_counts().plot(kind="bar")


# In[ ]:


# plotting the label and checking it's distribution

plt.figure(figsize=(10, 6))
df['label'].value_counts().plot(kind="bar")


# In[ ]:


# displaying the count values

df['label'].value_counts()


# In[ ]:


# no.f missing values

df.isnull().sum(axis=0)


# In[ ]:


# percentage of missing values

round((df.isnull().sum(axis=0) / len(df)), 2)


# In[ ]:


# values count of the protocol type

df['protocol_type'].value_counts()


# In[ ]:


# values count of the property flag

df['flag'].value_counts()


# In[ ]:


# because of size of the data we are keeping only http attacks

df = df[df['service'] == "http"]


# In[ ]:


df.info()


# In[ ]:


# dropping the service now

df = df.drop("service", axis=1)


# In[ ]:


# removing the service from the columns

columns.remove("service")


# In[ ]:


# value count on the new reduced dataset

df['label'].value_counts()


# In[ ]:


# using the sklearn LabelEncoder to convert the categorical value into numeric type

for column in df.columns:
    if df[column].dtype == np.object:
        encoded = LabelEncoder()
        encoded.fit(df[column])
        df[column] = encoded.transform(df[column])


# In[ ]:


# after converting all the categorical variables into numerica values

df.head()


# In[ ]:


df.info()


# In[ ]:


# checking correlation

corr = df.corr()


# In[ ]:


plt.figure(figsize=(20, 12))
sns.heatmap(corr, annot=True)


# In[ ]:




