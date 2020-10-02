#!/usr/bin/env python
# coding: utf-8

# # Can we find unique clients in data?
# Transactions in data looks like are independent of each other. Perhaps the organizers made it for better data anonymization. But what if we find transactions belonging to the same user? May be it will help someone in this competition.
# 
# I accidentally saw some magic in the feature V307

# In[ ]:



import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import hashlib
import matplotlib.pyplot as plt
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[ ]:


pd.options.display.max_rows = 500
pd.options.display.max_columns = 100


# In[ ]:


train = pd.read_csv('/kaggle/input/ieee-fraud-detection/train_transaction.csv')
train_ind = pd.read_csv('/kaggle/input/ieee-fraud-detection/train_identity.csv')


# In[ ]:


train = train.merge(train_ind, how = 'left', on ='TransactionID' )
del train_ind


# In[ ]:


train['card1'] = train['card1'].fillna(0)
train['card2'] = train['card2'].fillna(0)
train['card3'] = train['card3'].fillna(0)
train['card5'] = train['card5'].fillna(0)
train['card4'] = train['card4'].fillna('nan')
train['card6'] = train['card6'].fillna('nan')


# ### Functions that will help us find unique devices and unique cards. (I think it is clear for you why we use hash functions)

# In[ ]:


def card_info_hash(x):
    s = (str(int(x['card1']))+
         str(int(x['card2']))+
         str(int(x['card3']))+
         str(x['card4'])+
         str(int(x['card5']))+
         str(x['card6']))
    h = hashlib.sha256(s.encode('utf-8')).hexdigest()[0:15]
    return h


# In[ ]:


def device_hash(x):
    s =  str(x['id_30'])+str(x['id_31'])+str(x['id_32'])+str(x['id_33'])+str( x['DeviceType'])+ str(x['DeviceInfo'])
    h = hashlib.sha256(s.encode('utf-8')).hexdigest()[0:15]
    return h


# In[ ]:


train['card_hash'] = train.apply(lambda x: card_info_hash(x), axis=1   )
train['device_hash'] = train.apply(lambda x: device_hash(x), axis=1   )


# In[ ]:


def get_data_by_card_hash( data, card_hash):
    mask = data['card_hash']==card_hash
    return data.loc[mask,:].copy()


def get_data_by_device_hash( data, device_hash):
    mask = data['device_hash']==device_hash
    return data.loc[mask,:].copy()


def get_data_by_card_and_device_hash( data, card_hash, device_hash):
    mask = (data['card_hash']==card_hash) &(data['device_hash']==device_hash)
    return data.loc[mask,:].copy()


# In[ ]:


s = train.groupby(['card_hash' , 'device_hash'])['isFraud'].agg(['mean', 'count'])


# In[ ]:


s[(s['mean']==1) & (s['count']>15) ].head(500)


# In[ ]:





# In[ ]:


very_strange_thing = get_data_by_card_and_device_hash(train, '751777fefa9891d', '669f9256b04fb02')


# In[ ]:


very_strange_thing[[ 'TransactionID',
 'isFraud',
 'TransactionDT',
 'TransactionAmt',
 'ProductCD',
 'device_hash','card_hash', 'V307']]


# # Don't see anything?

# In[ ]:


#magic
very_strange_thing['V307_diff'] = very_strange_thing['V307'].diff().shift(-1)


# In[ ]:


very_strange_thing['difference'] = very_strange_thing['V307_diff'] - very_strange_thing['TransactionAmt']


# In[ ]:


very_strange_thing[[ 'TransactionID',
 'isFraud',
 'TransactionDT',
 'TransactionAmt',
 'ProductCD',
 'device_hash','card_hash', 'V307', 'V307_diff', 'difference']]


# In[ ]:


len(very_strange_thing)


# > # Coincidence? i don't think so 

# ![](https://www.meme-arsenal.com/memes/06308418f56ad0d8d674c247e5ccba49.jpg )

# ### Highly likely V307 is a cumulative sum of transactions for a certain period for one unique client
# 
# This group of transactions has the same card hash and device hash, and 41 consecutively increasing numbers. Also all this transactions are fraud. 
# 
# May be this knowledge will help you to significantly improve your models. 
# 
# It doesn't work for all pairs of card and devices hashes, so you can improve alghoritm for searching unique clients.

# # Bonus: You can try to decipher the values of some features

# In[ ]:


def dt_features(data):
    data = data.copy()
    
    start_dt = data['TransactionDT'].min()
    data['TransactionDT_norm']  = (data['TransactionDT'] - start_dt)/3600
    data['TransactionDT_norm_days'] = data['TransactionDT_norm']/24
    data['TransactionDT_diff'] = data['TransactionDT_norm'].diff().fillna(0)
    data['TransactionDT_diff_days'] = data['TransactionDT_diff']/24
    return data


# In[ ]:


very_strange_thing = dt_features(very_strange_thing)


# In[ ]:


list_of_interesting_features = [ 'TransactionID',
 'isFraud',
 'TransactionDT',
 'TransactionAmt',
 'ProductCD',
 'device_hash','card_hash', 'V307', 'V307_diff', 'difference', 
                                'TransactionDT_norm_days', 'TransactionDT_diff', 'TransactionDT_diff_days', 
                               
                               ] + ['D{}'.format(i) for i in range(1,16)]


# In[ ]:


very_strange_thing[list_of_interesting_features]

# look at  TransactionDT_norm_days and  D8	D9
# D1 looks like  TransactionDT_norm_days rounded to int


# In[ ]:


plt.figure(figsize=(14,6))
plt.plot( very_strange_thing['TransactionDT_norm_days'], very_strange_thing['V202'] , label = 'V202' )
plt.plot( very_strange_thing['TransactionDT_norm_days'], very_strange_thing['V203'] , label = 'V203')
plt.plot( very_strange_thing['TransactionDT_norm_days'], very_strange_thing['V204'] , label = 'V204')

plt.plot( very_strange_thing['TransactionDT_norm_days'], very_strange_thing['V306'] , label = 'V306')
plt.plot( very_strange_thing['TransactionDT_norm_days'], very_strange_thing['V307'] , label = 'V307')
plt.plot( very_strange_thing['TransactionDT_norm_days'], very_strange_thing['V308'] , label = 'V308')

plt.plot( very_strange_thing['TransactionDT_norm_days'], very_strange_thing['V316'] , label = 'V316')
plt.plot( very_strange_thing['TransactionDT_norm_days'], very_strange_thing['V317'] , label = 'V317')
plt.plot( very_strange_thing['TransactionDT_norm_days'], very_strange_thing['V318'] , label = 'V318')

plt.xlabel('Datetime')

plt.legend()
plt.show()


# In[ ]:


list_of_interesting_features = [ 'TransactionID',
 'isFraud',
 'TransactionDT',
 'TransactionAmt',
 'ProductCD',
 'device_hash','card_hash', 'V307', 'V307_diff', 'difference', 
                                'TransactionDT_norm_days', 'TransactionDT_diff', 'TransactionDT_diff_days', 
                               
                               ] + ['C{}'.format(i) for i in range(1,15)]


# In[ ]:


very_strange_thing[list_of_interesting_features]
# c2


# In[ ]:





# ![](https://www.dictionary.com/e/wp-content/uploads/2018/04/another-one.jpg )

# In[ ]:


s = get_data_by_card_and_device_hash(train, 'b4f15ed9e7c1e0a', 'b62aa9813bf1ea8')[[ 'TransactionID',
'isFraud',
'TransactionDT',
'TransactionAmt',
'ProductCD',
'device_hash','card_hash', 'V307']]


# In[ ]:


#magic
s['V307_diff'] = s['V307'].diff().shift(-1)
s['difference'] = s['V307_diff'] - s['TransactionAmt']


# In[ ]:


s

