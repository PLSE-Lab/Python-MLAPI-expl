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


df = pd.read_csv('/kaggle/input/e-commerce-purchase-dataset/purchase_data_exe.csv')


# In[ ]:


df.head()


# In[ ]:


df.drop(['Unnamed: 7'], axis=1, inplace=True)


# In[ ]:


df.head()


# In[ ]:


df['payment_method'] = df['payment_method'].astype('category').cat.codes
#Now 0 represents credit and 1 represents paypal


# In[ ]:


df['payment_method'] = df['payment_method'].astype('int')


# In[ ]:


df.dtypes


# In[ ]:


from datetime import datetime

#df['DateTime'] = pd.to_datetime(df['date'])
df['Year']=[d.split('/')[2] for d in df.date]
df['Month']=[d.split('/')[1] for d in df.date]
df['Day']=[d.split('/')[0] for d in df.date]
df.Year = df.Year.astype('int')
df.Month = df.Month.astype('int')
df.Day = df.Day.astype('int')
df.drop(['date'], axis = 1, inplace= True)


# In[ ]:


df.head()


# In[ ]:


df.dtypes


# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# In[ ]:


plt.hist(df.product_category, bins = 20)
plt.xlabel("Product Category")
plt.ylabel("Frequency")
plt.show()


# In[ ]:


plt.hist(df.payment_method, bins = 2)
plt.xlabel("Payment Method")
plt.ylabel("Frequency")
plt.show()


# In[ ]:


plt.scatter(x = df["clicks_in_site"], y = df["value [USD]"])
plt.xlabel("Clicks in site")
plt.ylabel("Value in USD")
plt.show()


# In[ ]:


plt.scatter(x = df["time_on_site [Minutes]"], y = df["value [USD]"])
plt.xlabel("Time on Site in Minutes")
plt.ylabel("Value in USD")
plt.show()


# In[ ]:


plt.scatter(x = df["payment_method"], y = df["value [USD]"])
plt.xlabel("Payment Method")
plt.ylabel("Value in USD")
plt.show()


# In[ ]:


new=df.groupby("payment_method")["value [USD]"]
new.mean()


# In[ ]:


plt.scatter(x = df["product_category"], y = df["value [USD]"])
plt.xlabel("Product Category")
plt.ylabel("Value in USD")
plt.show()


# In[ ]:


new=df.groupby("product_category")["value [USD]"]
new.mean()


# In[ ]:


new=df.groupby("Day")["value [USD]"]
new.sum()


# In[ ]:


df.corr()


# In[ ]:





# In[ ]:




