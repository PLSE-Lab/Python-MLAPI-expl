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


# ## Problem Description

# **Predict if the Merchant is Fraudster or not for an e-commerce clint**

# **Description:**

# 'Suppose "RRR" is a large e-commerce company with its operations in several countries. As the online giants grows, so has the number of fraudster merchants are. They deliver counterfeits or, in some cases, nothing at all. schemes leave customers duped, and place both ligitimate merchants and the company itself in a constant battle to rid the marketplace of scammers. Determing this also important in budgetting for fraud investigation. It's a well known problem both to the company and to the merchants, which they say hasn't effectively addressed the issue. They are serious about it and want to protect themselves from these fraudulent merchants using technology.

# *We're expected to create an analytical and modelling framework to predict the Merchant Fraudulency(yes/no) based on the quantitative and qualitative features provided in the datset*

# ##  Training Data- Pre Processing Steps

# In[ ]:


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# !pip install ipaddress
import ipaddress

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC


# In[ ]:


import missingno as msno


# In[ ]:


order = pd.read_csv("../input/train_order_data.csv") #order information
merch = pd.read_csv("../input/train_merchant_data.csv") #Merchant information
target = pd.read_csv("../input/train.csv") #Target is not available as it is to be predicted
ip_country = pd.read_csv("../input/ip_boundaries_countries.csv") #Ip address boundries of each country


# In[ ]:


# Sanity check for Train Dataset(Orders)
order.head()
# order.count() #54213


# In[ ]:


order.count() #54213


# In[ ]:


order.tail()


# In[ ]:


import matplotlib.pyplot as plt
msno.bar(order)
plt.show()


# In[ ]:


order['Order_ID'].nunique() #54213-- No duplicates found in order_Id


# In[ ]:


#sanity check for Train Dataset(merchant)
merch.head()


# In[ ]:


merch.tail()


# In[ ]:



msno.bar(merch)
plt.show()


# In[ ]:


merch.count() #54213


# In[ ]:


merch["Merchant_ID"].nunique() #54213---No duplicates found in Merchant_ID


# In[ ]:


#Customer_id and Order ID are almost varying by record. Can be dropped

order["Customer_ID"].nunique() #34881


# In[ ]:


order['Order_ID'].nunique() #54213- No predictive power


# In[ ]:


order.head()


# **join order and merchants data on merchant ID**

# In[ ]:


merged_data = pd.merge(order, merch, how='inner', on = "Merchant_ID")
merged_data.head()


# In[ ]:


target.head()
target['Merchant_ID'].nunique() #54231 -- No duplicates on Merchant ID. Ok to join


# In[ ]:


# target['Merchant_ID'].nunique() #54231 -- No duplicates on Merchant ID. Ok to join


# **Join train data with y_train data to grt it all together**

# In[ ]:


merged_data = pd.merge(merged_data, target, how = 'inner', on = 'Merchant_ID')


# In[ ]:


merged_data.head()


# In[ ]:


merged_data.tail()


# In[ ]:


print(merged_data.count()) #54213 --Row count not affected, joins worked fine


# del(target)

# ##   SPLITTING TRAIN & VALIDATION DATASET

# In[ ]:


X = merged_data.copy().drop("Fraudster",axis=1)
y = merged_data["Fraudster"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20,stratify=y) 


# In[ ]:


print("X_train",X_train.shape)
print("X_test",X_test.shape)
print("y_train",y_train.shape)
print("y_test",y_test.shape)


# In[ ]:


y_train.head()


# In[ ]:


X_train.head()


# **Building a dataframe at a device ID level.**

# **The assumption here being for a given device ID, there should be a single merchant found. In scenarios where there are multiples, it could be suspected as fraudlent activity
# **

# In[ ]:


device_id = X_train.groupby(["Registered_Device_ID"]).agg({"Merchant_ID":"nunique"}).reset_index()
device_id = device_id[(device_id["Merchant_ID"] > 1)].reset_index(drop=True)
device_id.head()


# In[ ]:


device_id.count()


# In[ ]:


device_id.count().nunique()


# ## Creating a new feature which will indicate if a given device type ID has multiple regreesions tagged against it

# In[ ]:


device_id["Multiple_Merchants"] = np.nan

for i in range(0, device_id.shape[0]):
    if device_id.loc[i, 'Merchant_ID'] > 1:
        device_id.loc[i,'Multiple_Merchants'] =1
    else:
        device_id.loc[i,'Multiple_Merchants'] = 0


# In[ ]:


device_id["Muliple_Merchants"] = device_id["Multiple_Merchants"].astype('category')
device_id.head()


# In[ ]:


device_id.drop(['Merchant_ID'],axis=1,inplace=True)


# In[ ]:


print(device_id.dtypes)
device_id.head()


# In[ ]:


device_id.count()


# **Add the new feature on to the train data**

# In[ ]:


X_train = pd.merge(X_train, device_id, how='left', on = 'Registered_Device_ID') #get the device-id level flags on train data


# In[ ]:


X_train.count()


# In[ ]:


print(X_train.dtypes)
X_train.head()


# In[ ]:


X_train["target"] = y_train


# In[ ]:


ip_address = X_train.groupby(['IP_Address']).agg({'target':"sum",'Merchant_ID':'nunique'}).reset_index()
ip_address[ip_address['target']>1].head()

