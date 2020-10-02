#!/usr/bin/env python
# coding: utf-8

# # Find unique client by using D1 and Card information
# 
# I joined this competition two weeka ago, and try to [find the good teammate](https://www.kaggle.com/c/ieee-fraud-detection/discussion/109873#latest-632441), Finally team up with @[Nanashi](https://www.kaggle.com/jesucristo), very great team work experience with him.
# 
# ## EDA on first week
# Because we don't have feature, I foucs on EDA, base on Konstantin's feature on this great kernel,and try some FE
# * https://www.kaggle.com/kyakovlev/ieee-simple-lgbm
# * Generate some new feature, single LGBM model, LB 0.9485 to 0.9504
# 
# ## Below is the leak that we find, implemnt it on submit file, boost ~ 0.002 score 
# ### [Reference kernel](https://www.kaggle.com/alexanderzv/find-unique-clients)

# In[ ]:


import numpy as np # linear algebra 
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from datetime import timedelta
import os, sys, gc, warnings, random, datetime
import hashlib
import matplotlib.pyplot as plt
import os
import gc
"""
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
"""
# Any results you write to the current directory are saved as output.


# In[ ]:


pd.options.display.max_rows = 500
pd.options.display.max_columns = 100


# ### Read Data

# In[ ]:


train = pd.read_csv('/kaggle/input/ieee-fraud-detection/train_transaction.csv')
train_ind = pd.read_csv('/kaggle/input/ieee-fraud-detection/train_identity.csv')
test = pd.read_csv('/kaggle/input/ieee-fraud-detection/test_transaction.csv')
test_ind = pd.read_csv('/kaggle/input/ieee-fraud-detection/test_identity.csv')


# In[ ]:


#train = pd.read_csv('/kaggle/input/ieee-fraud-detection/train_transaction.csv')
train_len = len(train)
#del train
#gc.collect()


# ### Merge ID to original training data

# In[ ]:


train = train.merge(train_ind, how = 'left', on ='TransactionID' )
test = test.merge(test_ind, how = 'left', on ='TransactionID' )
del train_ind,test_ind


# In[ ]:


all_data = pd.concat([train, test])
del train,test
gc.collect()


# ### D1 meaning (days from the first transaction of each client)
# * https://www.kaggle.com/akasyanama13/eda-what-s-behind-d-features
# 
# As my experimence of time series base data analysis, time correlate feature is the important information, I read this great kernel and I realize the D1 meaning, using D1 and try to find client information.

# In[ ]:


START_DATE = datetime.datetime.strptime('2017-11-30', '%Y-%m-%d')
all_data['DT_time'] = all_data['TransactionDT'].apply(lambda x: (START_DATE + datetime.timedelta(seconds = x)))
all_data['count'] = 1
all_data['diff_days_from_first_transaction'] = all_data['D1'].fillna(0).apply(lambda x: (datetime.timedelta(days = x)))
all_data['client_firstdate'] = all_data['DT_time']- all_data['diff_days_from_first_transaction']
all_data['client_firstdate_days'] = (all_data['DT_time']- all_data['diff_days_from_first_transaction']).apply(lambda x:str(x.date()))


# In[ ]:


all_data[['DT_time','diff_days_from_first_transaction','client_firstdate_days','D1']].head()


# ### Remove no need feature

# In[ ]:


anaysis_fea = [ 'TransactionID',
 'isFraud',
 'TransactionDT',
 'TransactionAmt',
 'ProductCD',
 'device_hash','card_hash', 'V307','id_30','id_31','id_32','id_33','DeviceType','DeviceInfo',
 'card1','card2','card3','card4','card5','card6','client_firstdate_days','dist1','dist2','P_emaildomain','addr1','addr2','train_or_test','count','DT_time','diff_days_from_first_transaction','client_firstdate']
#anaysis_fea+['M'+str(i+1) for i in range(9)]
drop_fea = [col for col in all_data.columns if col not in anaysis_fea]


# In[ ]:


all_data = all_data.drop(drop_fea,axis=1)


# In[ ]:


gc.collect()


# ### Functions that will help us find unique cards.

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


all_data['card1'] = all_data['card1'].fillna(0)
all_data['card2'] = all_data['card2'].fillna(0)
all_data['card3'] = all_data['card3'].fillna(0)
all_data['card5'] = all_data['card5'].fillna(0)
all_data['card4'] = all_data['card4'].fillna('nan')
all_data['card6'] = all_data['card6'].fillna('nan')


# In[ ]:


all_data['card_hash'] = all_data.apply(lambda x: card_info_hash(x), axis=1)


# In[ ]:


def get_data_by_card_hash( data, card_hash):
    mask = data['card_hash']==card_hash
    return data.loc[mask,:].copy()


def get_data_by_device_hash( data, device_hash):
    mask = data['device_hash']==device_hash
    return data.loc[mask,:].copy()


def get_data_by_card_and_startdate( data, card_hash, device_hash):
    mask = (data['client_firstdate_days']==card_hash) &(data['card_hash']==device_hash)
    return data.loc[mask,:].copy()


# ## Groupby start_date and unique card to find the unique client

# In[ ]:


all_data['count']=1
grp = all_data.iloc[:train_len].groupby(['client_firstdate_days','card_hash'])['count'].agg('sum')
#Let us display the count >10 client 
display_group = get_data_by_card_and_startdate(all_data,grp[grp>10].index[0][0],grp[grp>10].index[0][1])
display_group[['DT_time','TransactionAmt','V307','id_30','id_31','id_32','id_33','DeviceType','DeviceInfo','dist1','dist2','P_emaildomain']]


# ### What we found 
# * You can see the V307 means the "cumulative sum of amounts", I don't think it's coincidence.
# * So I think that "Same start date" and "same Card information" would be the unique client 
# * Then we can use this information to featch more things and create more features.

# ### Find the client that all Fraud
# * Base on AirmH's [discussion](https://www.kaggle.com/c/ieee-fraud-detection/discussion/109455) here.
# * Once we can find the unique client, and if the client alwasy fraud, and the client also in the test set, then we can just set them to isFraus=1

# In[ ]:


s = all_data.iloc[:train_len].groupby(['client_firstdate_days','card_hash'])['isFraud'].agg(['mean', 'count'])


# In[ ]:


s.head()


# ### Get those client in Test set, and record the Transaction ID

# In[ ]:


from tqdm import tqdm
Test_ID=[]
for ind in tqdm(s[(s['mean']==1)].index):
    very_strange_thing = get_data_by_card_and_startdate(all_data, ind[0],ind[1])
    Test_ID.extend(very_strange_thing[very_strange_thing['isFraud'].isna()]['TransactionID'].tolist())


# In[ ]:


Test_ID[:20]


# In[ ]:


np.save("test_IDs", Test_ID)


# ### Set those client as Fraud client

# In[ ]:


submit = pd.read_csv("../input/ieeecis-fraud-detection-results/stack_gmean_09_16th_2019_LB0pt9530.csv")


# In[ ]:


mask = submit['TransactionID'].isin(Test_ID)
submit.loc[mask,'isFraud'] =1


# In[ ]:


submit.loc[mask,:]


# In[ ]:


submit.to_csv('submit_try.csv',index = False)


# ## Conclusion
# * I found this at last 4 hours, and boost our ensemble model to silver zone, it's the key that we can get the silver.
# * I wish I have more time to use this information to create mroe feature and imporve model.
# * It's really fun and interest competition, I am very regret that I didn't join this competition early.
# * Congrats to all winners! Happy kagglers!
