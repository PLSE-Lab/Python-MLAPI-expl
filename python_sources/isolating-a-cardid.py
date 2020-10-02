#!/usr/bin/env python
# coding: utf-8

# # Isolate a cardID
# Unique card identifier was often called "magic feature", it indeed gives a great boost to the models if identified, but reconstructing it is quite straightforward.
# 
# This is the first of a two parts kernel series, in the second one we are going to identify a user.

# ### Let's begin
# <img src="https://images.tailorstore.com/YToyOntzOjU6IndpZHRoIjtzOjQ6IjEwMDAiO3M6NjoiaGVpZ2h0IjtzOjA6IiI7fQ%3D%3D/images/cms/TS-Sleeves-4.jpg" alt="remontetesmanchesfrangin" width="500">

# Have a look at this kernel, which is a starting point to the present one:
# https://www.kaggle.com/tuttifrutti/creating-features-from-d-columns-guessing-userid
# 
# Basicaly, it says that D1 is the number of days elapsed since the first transaction of a card. So that in order to identify a unique card, we could aggregate card1 to card6 (which we assume is stable by card), and D1 minus day.

# In[ ]:


import pandas as pd
import numpy as np

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
train_transaction = pd.read_csv('../input/ieee-fraud-detection/train_transaction.csv')        
train_identity = pd.read_csv('../input/ieee-fraud-detection/train_identity.csv')
train_transaction = train_transaction.merge(train_identity, how='left', left_on='TransactionID',right_on='TransactionID')
del train_identity


# In[ ]:


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


# On the cell below, I create the cardID resulting of the starting point notebook.

# In[ ]:


train_transaction['day'] = train_transaction['TransactionDT']/(3600*24)
train_transaction['D1minusday'] = (train_transaction['D1']-train_transaction['day']).replace(np.nan, -9999).map(int)
colsID = ['card1','card2','card3','card4','card5','card6','D1minusday','ProductCD']
train_transaction = create_key(train_transaction, colsID, 'cardID_D1')


# In[ ]:


train_transaction['cardID_D1'].value_counts()


# Ok, here we go. We obviously do not identify unique cards with this combination. We assume that a unique card cannot process 1414 transactions in a period of 6 months, neither 480. **We need to find something more robust.**
# 
# In the identity data, we have some informations about the device used, the browser, its version ect... 
# We can assume that a sequence of fraudulent transactions in a short timeframe, from the same card group we have created, made from the same Device has great chances to come from the same user right? <br>
# Let's have a look then at this kind of sequences and try to derive information from other variables.

# In[ ]:


len(train_transaction[(train_transaction.isFraud==1) & (train_transaction.ProductCD=="C")])


# In[ ]:


train_transaction['V307'] = train_transaction['V307'].fillna(0)
train_transaction['V307plus'] = train_transaction['V307']+train_transaction['TransactionAmt']


# As there are not that many, let's export all the frauds from product C in excel and see what we can find. Excel is so underrated, and I find it to be much easier to explore data with this tool, when the circumstances permit it (it is the case most of the time).
# 
# So after looking at the cluster in short time frame of devices, same amounts, or other variables that might indicate a unique user, we find that V307 represents the cumsum of the transactions of the same card groups.
# 
# So we could join the V307 to reconstruct the card.
# 
# I've created a column, V307plus, which is the transaction Amt plus V307 of the same line. The result is the value expected for V307 on the next transaction of the same card. <br>
# Here I have colored the groups respecting the sequence for a same card group created previously (try to match V307plus, with the next V307, and you'll find the sequences)
# 
# 
# <img src="https://i.ibb.co/NKCHH7M/Capture-d-cran-de-2019-09-27-21-44-34.png" alt="V307V307plus">
# 
# By the way, it seems that the first card was used on multiple device, this could be great feature.
# 
# Back to our notebook. Let's create the V307plus feature and try to recreate the sequences of cards.

# Basically we have to find the V307 that matches V307plus. Like the V307 value of ID 3030465 with V307plus of 3026025

# In[ ]:


train_transaction.loc[(train_transaction.TransactionID==3030465),'V307'].values[0] == train_transaction.loc[(train_transaction.TransactionID==3026025),'V307plus'].values[0]


# Ho wait! it does not match. 
# 
# The reason is that TransactionAmt suffered from some rounding or truncating.
# 
# Let's create some variants of V307 and TransactionAmt and see if we can reconstruct the cards (the parameters of the roundings and truncatings in the following cell were found after some trial and errors)

# In[ ]:


train_transaction['V307rtrunc'] = train_transaction['V307'].apply(lambda x: truncate(x,3))
train_transaction['V307round'] = train_transaction['V307'].apply(lambda x: round(x,3))
train_transaction['V307plusround'] = train_transaction['V307plus'].apply(lambda x: round(x,4))
train_transaction['V307plusroundtrunc'] = train_transaction['V307plusround'].apply(lambda x: truncate(x,3))
train_transaction['V307plusround'] = train_transaction['V307plus'].apply(lambda x: round(x,3))
train_transaction['V307trunc2'] = train_transaction['V307'].apply(lambda x: truncate(x,2))
train_transaction['V307plustrunc2'] = train_transaction['V307plus'].apply(lambda x: truncate(x,2))
train_transaction['TransactionAmttrunq'] = train_transaction['TransactionAmt'].apply(lambda x: round(x,3))


# And try to find the couples of TransactionID that have matching V307 and V307plus based on a transformation of V307 and V307plus (truncated for the first one, and round then truncated for the second)

# In[ ]:


#the card group of interest for this example
card_group = train_transaction[train_transaction.cardID_D1=='16136204.0185.0visa138.0debit108C']

list1 = card_group['V307plusroundtrunc'].tolist()
list2 = card_group['V307rtrunc'].tolist()
kv = []
res = [[list(filter(lambda z: list1[z]==x, range(len(list1)))),list(filter(lambda z: list2[z]==x, range(len(list2))))] for x in list1 if x in list2] #find the pairs
res= [list(map(kv.append,map(list,(itertools.product(*sublist))))) for sublist in res] #drop duplicates from list of list
res = list(map(list, set(map(lambda i: tuple(i), kv)))) #create list of couple indexes
list1 = card_group.iloc[[i[0] for i in res]]['TransactionID'].tolist()
list2 = card_group.iloc[[i[1] for i in res]]['TransactionID'].tolist()
liste_existstrun = merge(list1, list2)


# In[ ]:


liste_existstrun


# We get the following couples of TransactionID that match based on these transformations of V307 and V307plus. Let's group them

# In[ ]:


L = liste_existstrun
G = nx.Graph()

G.add_nodes_from(sum(L, []))
q = [[(s[i],s[i+1]) for i in range(len(s)-1)] for s in L]
for i in q:
    G.add_edges_from(i)
group_list = [list(i) for i in nx.connected_components(G)]
group_list


# It seems that we were successful, in drawing 6 groups with these transformations of V307 features.
# But we had 5 groups in this card group (based on D1 minus day and card1-6).
# 
# After many iterations, trial and errors, i ended-up with the following function, grouping based on:
# - matching V307plus rounded (4 decimals) then truncated (3dec), and V307 rounded (3dec)
# - V307plus rounded and V307 rounded (3dec)
# - V307plus and V307 truncated  (2dec)
# - V307 rounded and TransactionAmt truncated (sometimes a Transaction is not counted in V307, and repeat a previous Amount value)
# 
# You can see that here you have to make your assumptions, and that the result will depend strongly on them.

# In[ ]:


def find_groups(aa):
    group_list = []
    
    #get the couples by existstrun
    list1 = aa['V307plusroundtrunc'].tolist()
    list2 = aa['V307rtrunc'].tolist()
    kv = []
    res = [[list(filter(lambda z: list1[z]==x, range(len(list1)))),list(filter(lambda z: list2[z]==x, range(len(list2))))] for x in list1 if x in list2] #find the pairs
    res= [list(map(kv.append,map(list,(itertools.product(*sublist))))) for sublist in res] #drop duplicates from list of list
    res = list(map(list, set(map(lambda i: tuple(i), kv)))) #create list of couple indexes
    list1 = aa.iloc[[i[0] for i in res]]['TransactionID'].tolist()
    list2 = aa.iloc[[i[1] for i in res]]['TransactionID'].tolist()
    liste_existstrun = merge(list1, list2)


    #get the couples by existsroundtrunc
    list1 = aa['V307plusroundtrunc'].tolist()
    list2 = aa['V307round'].tolist()
    kv = []
    res = [[list(filter(lambda z: list1[z]==x, range(len(list1)))),list(filter(lambda z: list2[z]==x, range(len(list2))))] for x in list1 if x in list2] #find the pairs
    res= [list(map(kv.append,map(list,(itertools.product(*sublist))))) for sublist in res] #drop duplicates from list of list
    res = list(map(list, set(map(lambda i: tuple(i), kv)))) #create list of couple indexes
    list1 = aa.iloc[[i[0] for i in res]]['TransactionID'].tolist()
    list2 = aa.iloc[[i[1] for i in res]]['TransactionID'].tolist()
    liste_existsroundtrunc = merge(list1, list2)

    #get the couples by existsroundtrunc
    list1 = aa['V307plusround'].tolist()
    list2 = aa['V307round'].tolist()
    kv = []
    res = [[list(filter(lambda z: list1[z]==x, range(len(list1)))),list(filter(lambda z: list2[z]==x, range(len(list2))))] for x in list1 if x in list2] #find the pairs
    res= [list(map(kv.append,map(list,(itertools.product(*sublist))))) for sublist in res] #drop duplicates from list of list
    res = list(map(list, set(map(lambda i: tuple(i), kv)))) #create list of couple indexes
    list1 = aa.iloc[[i[0] for i in res]]['TransactionID'].tolist()
    list2 = aa.iloc[[i[1] for i in res]]['TransactionID'].tolist()
    liste_existsroundround = merge(list1, list2)


    #get the couples by existsroundtrunc
    list1 = aa['V307trunc2'].tolist()
    list2 = aa['V307plustrunc2'].tolist()
    kv = []
    res = [[list(filter(lambda z: list1[z]==x, range(len(list1)))),list(filter(lambda z: list2[z]==x, range(len(list2))))] for x in list1 if x in list2] #find the pairs
    res= [list(map(kv.append,map(list,(itertools.product(*sublist))))) for sublist in res] #drop duplicates from list of list
    res = list(map(list, set(map(lambda i: tuple(i), kv)))) #create list of couple indexes
    list1 = aa.iloc[[i[0] for i in res]]['TransactionID'].tolist()
    list2 = aa.iloc[[i[1] for i in res]]['TransactionID'].tolist()
    liste_existstrunc2 = merge(list1, list2)


    #get the couples by existsamount
    list1 = aa['TransactionAmttrunq'].tolist()
    list2 = aa['V307round'].tolist()
    kv = []
    res = [[list(filter(lambda z: list1[z]==x, range(len(list1)))),list(filter(lambda z: list2[z]==x, range(len(list2))))] for x in list1 if x in list2] #find the pairs
    res= [list(map(kv.append,map(list,(itertools.product(*sublist))))) for sublist in res] #drop duplicates from list of list
    res = list(map(list, set(map(lambda i: tuple(i), kv)))) #create list of couple indexes
    list1 = aa.iloc[[i[0] for i in res]]['TransactionID'].tolist()
    list2 = aa.iloc[[i[1] for i in res]]['TransactionID'].tolist()
    liste_existsamount = merge(list1, list2)

    #get by exact same amount
    a=[]
    liste_sameamount = aa.groupby('TransactionAmt')['TransactionID'].apply(list).tolist()
    res = [list(map(a.append, map(list,zip(i, i[1:] + i[:1])))) for i in liste_sameamount]

    group_list.extend(liste_existstrun)
    group_list.extend(liste_existsroundtrunc)
    group_list.extend(liste_existsamount)
    group_list.extend(liste_existsroundround)
    group_list.extend(liste_existstrunc2)

    group_list.extend(a)

    L = group_list
    G = nx.Graph()
    G.add_nodes_from(sum(L, []))
    q = [[(s[i],s[i+1]) for i in range(len(s)-1)] for s in L]
    for i in q:
        G.add_edges_from(i)
    group_list = [list(i) for i in nx.connected_components(G)]
    return group_list


# In[ ]:


groups_found = find_groups(card_group)
groups_found


# And here we go, we found the 5 groups that we identified in our Excel!
# 
# <img src="https://i.ibb.co/NKCHH7M/Capture-d-cran-de-2019-09-27-21-44-34.png" alt="V307V307plus">
