#!/usr/bin/env python
# coding: utf-8

# # Finding users
# 
# This is the second notebook about identifiers, coming after the one where I try to find unique cards. I'm going to use the results of the first one also as some of the methodology, so make sure you read it: 
# https://www.kaggle.com/tuttifrutti/isolating-a-cardid
# 
# ### In many discussions the Card ID is called User ID. But a User can use multiple cards! and especially a frauding one!
# 
# 

# The assumption is that a fraudster - if he is a professional in the field- uses multiple cards to perform his art. He has a set of cards, and will try to pay with each one, expecting one or all to work.
# 
# In addition, recreating the users identificators enables to create variables about User behaviour, and help the models to spot the differences between regular users and fraudsters (such as, using multiple cards, multiples devices, multiples IP adresses ect..)
# 
# Let's analyze a frauding sequence, and find a way to recreate it.
# 
# In the following image, we identified a cluster of cards. The transactions were made in a short time period, the amount was exactly the same, and so are id_19 and id_20 (which I interpret as a being some sort of machine or connection address). 
# I have colored the card groups identified by the previous kernel (1 card with 3 transactions, 1 with 2, and 2 with 1. The aim of this one, is to group these cards under the same user. Here the grouping is quite straightforward right? same amount, same "IP", same device, same browser. <p>
# 
# <img src="https://i.imgur.com/RvNZSo5.png" alt="ex1">
# 
# 
# Let's have a look at a much more interesting example. Below, we can see that the cardID algorithms identified 1 card with 6 transactions. And we have in the middle two transactions from different card.
# We can assume that all these transactions have been made by the same user.
# Grouping by device AND IP AND Amount, you'll group the 3 last transactions. The fact that in these group of 3, one of the transactions is connected to another group (the cardID group) enables you to create a group for the User.
# 
# The last transaction, acts like a **bridge** between the group of cardID, and the group of Device x Amount x IP x Browser
# 
# 
# <img src="https://i.imgur.com/RPXTko9.png" alt="ex1">
# 
# 
# 
# Actually here, you have to have a close look at the data and make your assumptions, my grouping keys are the following for product C:
# - **Key1**: group by Amount, id_19 and id_20 (this is pretty restrictive as the amounts in group C are very specific, so is the "IP" in a short timeframe)
# - **Key2**: group by id_19, id_20, id_31, Deviceinfo (this is also restrictive on a short timeframe
# - **Key3**: group by cardID (already done)
# - perform the grouping on 0.1 day (with the day truncated, and then rounded, to have a pseudo sliding window of size 0.1)
# 
# **Then find transactions that act like BRIDGES between those groups** (group by intersection of groups) <br>
# At the end of the day, all these transaction will be found in the same User group (see column 'newgroup'):
# 
# <img src="https://i.ibb.co/P1bkNNL/Capture-d-cran-de-2019-10-04-12-50-34.png" alt="ex1">
# 
# 
# 
# Now, it's time to compute it.
# 
# <img src="https://www.incimages.com/uploaded_files/image/970x450/dirty-hands-1725x810_21760.jpg" alt="handdirt" width=500>
# 

# ### 1. Load, merge, and subset, keeping the columns that we need
# 
# Make sure you download Train AND Test to create the User ID, so we can find ovelapping users.
# 
# In this example I'm running run the code only on train, as it will be quicker to run. Do not do it at home!

# In[ ]:


import pandas as pd
import numpy as np

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
train_transaction = pd.read_csv('../input/ieee-fraud-detection/train_transaction.csv')        
train_identity = pd.read_csv('../input/ieee-fraud-detection/train_identity.csv')
test_transaction = pd.read_csv('../input/ieee-fraud-detection/test_transaction.csv')        
test_identity = pd.read_csv('../input/ieee-fraud-detection/test_identity.csv')

train_transaction = train_transaction[['TransactionID','TransactionDT','TransactionAmt','ProductCD']]
train_identity = train_identity[['TransactionID','id_19','id_20','id_31','DeviceInfo']]
test_transaction = test_transaction[['TransactionID','TransactionDT','TransactionAmt','ProductCD']]
test_identity = test_identity[['TransactionID','id_19','id_20','id_31','DeviceInfo']]

train_transaction = train_transaction.merge(train_identity, how='left', left_on='TransactionID', right_on='TransactionID')
test_transaction = test_transaction.merge(test_identity, how='left', left_on='TransactionID', right_on='TransactionID')

total = train_transaction.copy()
total = total[total.ProductCD=='C']
del train_transaction
del test_transaction


# In[ ]:


#same functions as in my previous kernel about CardID
import itertools
import math
import networkx as nx

#function to create keys based on multiple columns
def create_key(df, cols, name_new_col):
    '''
    df: pandas dataframe
    cols: list of columns composing the key
    name_new_col: name given to the new column
    '''
    df.loc[:,name_new_col] = ''
    for col in cols:
        df.loc[:,name_new_col] = df.loc[:,name_new_col] + df.loc[:,col].astype(str)
    return df  

def truncate(f, n):
    return math.floor(f * 10 ** n) / 10 ** n  

def merge(list1, list2): 
    merged_list = [[p1, p2] for idx1, p1 in enumerate(list1)  
    for idx2, p2 in enumerate(list2) if idx1 == idx2] 
    return merged_list  

def find_groups(df, groupingcriteria):   
    a=[]
    liste_sameamount = df.groupby(groupingcriteria)['TransactionID'].apply(list).tolist()
    res = [list(map(a.append, map(list,zip(i, i[1:] + i[:1])))) for i in liste_sameamount]
    return a


# ### 2. Load the cards ID from the previous kernel

# In[ ]:


groups = pd.read_csv('../input/cardid/groups.csv')
groups = groups.set_index('TransactionID')
dictgroups = groups['groups'].to_dict()
total['cardID'] = total['TransactionID'].map(dictgroups)


# ### 3. Create the pseudo time window.
# 
# After some research you might find that a time window of 0.1 day is well suited (or not, you have to make your own assumptions) <br>
# So i choosed to perform two loops:
# - One on the truncated day, so i have ranges (time windows) like [1.1-1.199] [2.4-2.499]
# - One on rounded day, time windows: [1.75001-1.84999] [34.2400001-34.349999]
# 
# This way, if 2 transactions of the same group appear on 1.999 and 2.0001, they will be together in the "rounded" window. However, if the group is in 1.30 and 1.39, the truncated window will catch it.

# In[ ]:


total['day'] = total['TransactionDT']/(3600*24)
total['daytrunc'] = total['day'].apply(lambda x: truncate(x,1))
total['dayround'] = total['day'].apply(lambda x: round(x,1))
total['TransactionAmtround'] = total['TransactionAmt'].apply(lambda x: round(x,3))


# ### 4. Grouping by keys (heart of the notebook)
# 
# Remember, my grouping keys are the following:
# - **Key1**: group by Amount, id_19 and id_20 (this is pretty restrictive as the amounts in group C are very specific, so is the "IP" in a short timeframe)
# - **Key2**: group by id_19, id_20, id_31, Deviceinfo (this is also restrictive on a short timeframe
# - **Key3**: group by cardID (already done)
# 
# 
# #### Key1: Amount, id_19, id_20

# a. Create the key

# In[ ]:


total1 = total[['TransactionID','TransactionAmt','TransactionAmtround','id_19','id_20','daytrunc','dayround','day']].copy()
total1 = create_key(total1, ['TransactionAmtround','id_19','id_20'],'firstgroupcriteriaC')


# b. Get the matching couples of TransactionIDs based on the first key, on the **"rounded"** time window

# In[ ]:


import gc
timeframe = total1.dayround.unique().tolist()
group_list_C_criteria1 = []

for frame in timeframe:
    if frame%50==0:
        print('day',frame)
        gc.collect()
    
    subset = total1[total1['dayround']==frame].copy()
    if len(subset)==1:
        group_list_C_criteria1.append(subset['TransactionID'].tolist())
    else:
        group_list_C_criteria1.extend(find_groups(subset, 'firstgroupcriteriaC'))


# b bis. Get the matching couples of TransactionIDs based on the first key, on the **"truncated"** time window

# In[ ]:


timeframe = total1.daytrunc.unique().tolist()

for frame in timeframe:
    if frame%50==0:
        print('day',frame)
        gc.collect()
    
    subset = total1[total1['daytrunc']==frame].copy()
    if len(subset)==1:
        group_list_C_criteria1.append(subset['TransactionID'].tolist())
    else:
        group_list_C_criteria1.extend(find_groups(subset, 'firstgroupcriteriaC'))


# c. Drop duplicates, as the rounded and the truncated windows overlap

# In[ ]:


print(len(group_list_C_criteria1))
group_list_C_criteria1 = [list(tupl) for tupl in {tuple(item) for item in group_list_C_criteria1 }]
print(len(group_list_C_criteria1))


# We get all couples of TransactionIDs that match based on the previous criterias

# In[ ]:


group_list_C_criteria1[:10]


# d. Group the groups thanks to the **bridges** (transactionIDs present in multiple groups) and save the result

# In[ ]:


L = group_list_C_criteria1
G = nx.Graph()

G.add_nodes_from(sum(L, []))
q = [[(s[i],s[i+1]) for i in range(len(s)-1)] for s in L]
for i in q:
          G.add_edges_from(i)

group_list = [list(i) for i in nx.connected_components(G)]

myDict = {}

for i in range(0,len(group_list)):
    for element in group_list[i]:
        name='group'+str(i)
        myDict[element] = name
    
groupsCAmtid1920 = pd.DataFrame.from_dict(myDict, orient='index').reset_index()
groupsCAmtid1920.columns=['TransactionID','groupsCAmtid1920']


# In[ ]:


groupsCAmtid1920.head(5)


# This is the resulting dataframe. Groups by the first key.

# #### Key2 group by id_19, id_20, id_31, Deviceinfo
# 
# Perform the same algo as for Key 1 on another key

# In[ ]:


total1 = total[['TransactionID','id_19','id_20','id_31','DeviceInfo','daytrunc','dayround','day']].copy()
total1 = create_key(total1, ['id_19','id_20','id_31','DeviceInfo'],'secondgroupcriteriaC')
total1 = total1[(total1['id_20'].isna()==False) & (total1['id_19'].isna()==False) & (total1['id_31'].isna()==False)]
# this key is too indulgent if we don't get rid of missing id_19 and id_20 as many are missing, but try your experiments to find the best combination


# In[ ]:


import gc
timeframe = total1.dayround.unique().tolist()
group_list_C_criteria1 = []

for frame in timeframe:
    if frame%50==0:
        print('day',frame)
        gc.collect()
    
    subset = total1[total1['dayround']==frame].copy()
    if len(subset)==1:
        group_list_C_criteria1.append(subset['TransactionID'].tolist())
    else:
        group_list_C_criteria1.extend(find_groups(subset, 'secondgroupcriteriaC'))
    
## Second on Truncated data
timeframe = total1.daytrunc.unique().tolist()

for frame in timeframe:
    if frame%50==0:
        print('day',frame)
        gc.collect()
    
    subset = total1[total1['daytrunc']==frame].copy()
    if len(subset)==1:
        group_list_C_criteria1.append(subset['TransactionID'].tolist())
    else:
        group_list_C_criteria1.extend(find_groups(subset, 'secondgroupcriteriaC'))       


# In[ ]:


print(len(group_list_C_criteria1))
group_list_C_criteria1 = [list(tupl) for tupl in {tuple(item) for item in group_list_C_criteria1 }]
print(len(group_list_C_criteria1)) 


# In[ ]:


L = group_list_C_criteria1
G = nx.Graph()

G.add_nodes_from(sum(L, []))
q = [[(s[i],s[i+1]) for i in range(len(s)-1)] for s in L]
for i in q:
    G.add_edges_from(i)

group_list = [list(i) for i in nx.connected_components(G)]

groupsCid192031Device = pd.DataFrame.from_dict(myDict, orient='index').reset_index()
groupsCid192031Device.columns=['TransactionID','groupsCid192031Device']


# #### Key3: CardID, we already have it computed from the previous kernel.

# ### 5. Group the groups (deep heart of the kernel)
# 
# a. Create a dataframe with all the IDs, and the 3 different groupings by columns <br>
# b. imputation of different numbers for the group that does not cover all the IDs (where id_19 and id_20 were missing)

# In[ ]:


total1 = total[['TransactionID','day','cardID']].copy()

groupC1 = groupsCAmtid1920.copy()
groupC2 = groupsCid192031Device.copy()

total1 = total1.merge(groupC1, how='left',left_on='TransactionID',right_on='TransactionID')
total1 = total1.merge(groupC2, how='left',left_on='TransactionID',right_on='TransactionID')

#imputation
total1['imputecol'] = [i for i in range(0,len(total1))]
total1.loc[total1.groupsCid192031Device.isna(), 'groupsCid192031Device'] = total1.loc[total1.groupsCid192031Device.isna(), 'imputecol']


# c. Create all the couples of ID from the same groups, and drop duplicates

# In[ ]:


groups_C_final = []
groups_C_final.extend(find_groups(total1, 'cardID'))
print('group1done')

groups_C_final.extend(find_groups(total1, 'groupsCAmtid1920'))
print('group2done')

groups_C_final.extend(find_groups(total1, 'groupsCid192031Device'))
print('group3done')

print(len(groups_C_final))
groups_C_final = [list(tupl) for tupl in {tuple(item) for item in groups_C_final }]
print(len(groups_C_final))


# d. Create all the groups based on the 3 keys

# In[ ]:


L = groups_C_final
G = nx.Graph()

G.add_nodes_from(sum(L, []))
q = [[(s[i],s[i+1]) for i in range(len(s)-1)] for s in L]
for i in q:
    G.add_edges_from(i)

group_list = [list(i) for i in nx.connected_components(G)]

myDict = {}

for i in range(0,len(group_list)):
    for element in group_list[i]:
        name='group'+str(i)
        myDict[element] = name
    
groupsCuser = pd.DataFrame.from_dict(myDict, orient='index').reset_index()
groupsCuser.columns=['TransactionID','groupsCuser']
groupsCuser.to_csv('groupsCuser.csv',index=False)


# ### 6. Check the groups, and try again with other keys if not satisfied

# a. Load data

# In[ ]:


train_transaction = pd.read_csv('../input/ieee-fraud-detection/train_transaction.csv')        
train_identity = pd.read_csv('../input/ieee-fraud-detection/train_identity.csv')

train_transaction = train_transaction[['TransactionID','TransactionDT','TransactionAmt','ProductCD','isFraud']]
train_identity = train_identity[['TransactionID','id_19','id_20','id_31','DeviceInfo']]

total = train_transaction.merge(train_identity, how='left', left_on='TransactionID', right_on='TransactionID')

total = total[total.ProductCD=='C']
del train_transaction


# b. Add the groups

# In[ ]:


#cardID
groups = pd.read_csv('../input/cardid/groups.csv')
groups = groups.set_index('TransactionID')
dictgroups = groups['groups'].to_dict()
total['cardID'] = total['TransactionID'].map(dictgroups)

#New User group
total = total.merge(groupsCuser, how='left',left_on='TransactionID',right_on='TransactionID')


# c. Create your indicators

# In[ ]:


total['CardIDcount'] = total['cardID'].map(total.cardID.value_counts())
total['CardID_fraud_sum'] = total.groupby('cardID')['isFraud'].sum()

total['UserIDcount'] = total['groupsCuser'].map(total.groupsCuser.value_counts())
total['UserID_fraud_sum'] = total.groupby('groupsCuser')['isFraud'].sum()


# In[ ]:


total[total.isFraud==1].to_csv('checkgroups.csv',index=False)


# Let's check in our results in excel.
# 
# **1st example**
# On this first screenshot we can see:
# - Card1 14276, was group by CardID with another card, we cannot see the second on the screenshot, but the variable cardID_count indicates that this group contains 2 transactions (and 2 of them are fraudulent (cardID_sumFraud))
# - Same for Card1 8755.
# - These two cards were used by the same User. And are grouped with other cards. (12 transactions in total by the User, with 12 frauds)
# 
# <img src="https://i.ibb.co/WH1Y7NR/Capture-d-cran-de-2019-10-04-14-54-32.png" alt="ex1">
# 

# **2nd example**
# 
# Here we can see that a fraudulent card had 10 transactions (incl. 10 fraudulent), two others had one. <br>
# The User grouping succeeded in grouping these cards together along with other that don't appear on this screenshot. 
# 
# However we can see, that this user has 2 non fraudulent transaction out of 30 (error of aggregation?)
# 
# <img src="https://i.ibb.co/BjWJzxb/Capture-d-cran-de-2019-10-04-14-54-10.png" alt="ex1">
# 

# **3rd example**
# 
# This one is also nice, we can see that a card with 9 transactions, has been grouped with a single transaction card. It will be obviously easier to spot single transaction frauds thanks to this grouping!
# <img src="https://i.ibb.co/LPX36rc/Capture-d-cran-de-2019-10-04-14-54-18.png" alt="ex1">

# **4th and most spectacular example**
# 
# This user tried at least 231 transactions (the rest might be errors of grouping).
# 
# According to the grouping, the User made 275 transactions, out of which 231 were fraudulent.
# 
# We can see here, how this grouping by user ID can help!
# 
# <img src="https://i.ibb.co/9NNLxzg/Capture-d-cran-de-2019-10-04-14-54-14.png" alt="ex1">

# **5th example: predicting manually**
# 
# The UserID that was more represented on the test set, and very fraudulent on the train, had 144 lines on the test.
# 
# This is the training data of a very big fraudulent group. You can see that some are misclassified (look at colum isFraud), tou can easily spot them following the pattern on the C1/C2 columns.
# 
# <img src="https://i.ibb.co/wBB6jsm/Capture-d-cran-de-2019-10-01-00-40-15.png" alt="ex1">
# 
# This user had around 150 transactions in the test set. Here there is no column isFraud, but i guess you can easily spot the not-necessarily frauds looking at the C columns
# 
# <img src="https://i.ibb.co/K0DTfn4/Capture-d-cran-de-2019-10-01-00-41-23.png" alt="ex1">
# 

# **You can imagine that with this kind of feature (user having hundred of lines), you have to be very carefull about overfitting. Any unique feature for the group might become an ID**

# ### 7. Create user ID for the other products
# 
# The data available differs by product. Here my keys for the others:
# - R: 2 keys
#     - Amount x id_19 x id_20 x id_31 x id_20 x id_33 x DeviceInfo 
#     - CardID
# - H: 2 keys
#     - id_19 x id_20 x id_31 x id_30 x id_33 x DeviceInfo
#     - CardID
# - S: 2 keys
#     - Amt x id_19 x id_20 x id_30 x id_31 x id_33 x DeviceInfod
#     - CardID

# ### 8. Once your happy with the groups. Create Features
# 
# The User groups enables you to create features that make sense for the problem.
# - Number of card used by User
# - How hard the fraudster try not to be detected (changing IP, browser, device)
# 
# We have tried isolation forest based on the User variables, to detect outstanding behaviours. We used the results as a feature, and it turned out to bring information to the models.

# ## Takeaways:
# - Explore the data widely (Don't be ashamed of using excel, this does not make you less of a Data Scientist)
# - Run the algo on all the dataset, train+test, you'll find overlapping IDs
# - Iterate: Always check your result, and adjust your keys
# - Create your own KPIs to test your ideas (like the counts and sumFraud by group, or feed straight away your models with the outputs...)

# In[ ]:




