#!/usr/bin/env python
# coding: utf-8

# # Overview: 
# 
# This comes from the discussion about whether valiadtion schema. Should we use Holdout or CV? I argue that CV is a valid move if the split is applied based on unique card-ID. 
# 
# This kernel is not only about giving a unique ID for transaction made by the same card, but also gives a chance to apply CV in the right way withtout data leak. I will discuss this in another kernel. 
# 
# 
# # So How to identify unique card ID? 
# 
# 
# ### The idea behind : 
# 
# The card ID is made using the 'card_k' features. Logically speaking, a debit/credit card number is around 16 digits. Think about!
# 
# I found that a combination of card_k features : `['card1', 'card2', 'card3', 'card5',]` is giving me almost 14-16 digits. I used this as unique ID and found this pretty consistant. 
# 
# Let's see !
# 

# In[ ]:


import numpy as np  
import pandas as pd  
import os
print(os.listdir("../input"))
WDR="../input/"

import time
# import datetime
from contextlib import contextmanager
@contextmanager
def timer(title):
    print(title)
    t0 = time.time()
    yield
    print("{} - done in {:.0f}s".format(title, time.time() - t0))
    


# In[ ]:


with timer('reading the data...'):
    train_transaction = pd.read_csv(WDR+'train_transaction.csv')
#     train_identity = pd.read_csv(WDR+'train_identity.csv')
#      I found out we dont need train_identity for this. read below to understand
#     train = train_transaction.merge(train_identity, on='TransactionID', how='left', left_index=True, right_index=True)
    train = train_transaction.copy()


# In[ ]:


import datetime 
def corret_card_id(x): 
    x=x.replace('.0','')
    x=x.replace('-999','nan')
    return x

def definie_indexes(df):
    
    # create date column
    START_DATE = '2017-12-01'
    startdate = datetime.datetime.strptime(START_DATE, '%Y-%m-%d')
    df['date'] = df['TransactionDT'].apply(lambda x: (startdate + datetime.timedelta(seconds = x)))
   
    
    # create card ID 
    cards_cols= ['card1', 'card2', 'card3', 'card5']
    for card in cards_cols: 
        if '1' in card: 
            df['Card_ID']= df[card].map(str)
        else : 
            df['Card_ID']+= ' '+df[card].map(str)
    
    # sort train data by Card_ID and then by transaction date 
    df= df.sort_values(['Card_ID', 'date'], ascending=[True, True])
    
    # small correction of the Card_ID
    df['Card_ID']=df['Card_ID'].apply(corret_card_id)
    
    # set indexes 
    df= df.set_index(['Card_ID', 'date'])
    return df

with timer('define real IDs...'):
    train = definie_indexes(train)


# # Now we can see the data much much better than before
# 
# Now we have the data indexed by `Cards_ID` and `date` . Let's have a look on the train data: 

# In[ ]:


# I want to select only some columns 
cols= ['isFraud','TransactionDT','TransactionAmt','ProductCD', 'P_emaildomain', 'R_emaildomain']
cards_cols= ['card1', 'card2', 'card3','card4','card5', 'card6']
train[cols].head(30)


# # Findings: 

# 
# If I am right about the `Card_ID` there is something really weird in the data. But first I want to say this : 
# 
# 1. Bascially, if we detect a fraudulant card we should flag the card as fraud. Thus, every transaction made later by this card should be flagged as Fraud. 
# 
# 2. It's normal that for the same card, we have different (P_emaildomain,R_emaildomain) per transaction. This doesn't hurt the consistancy of my definition of `Card_ID`
# 
# 3. I think identity features has more to do with the device used to make a transaction. This is even more interesting because we can try to make a transaction from different devices but we are still using the same card. The Card_ID is more solid to create features IMHO.
# 
# 4. Use `Card_ID` instead of `TransactionID`. You will be able to create more features and more importantly, you will be able to link future transactions with past actions and get the fraudulant pattern. Plenty of features to be created based on that.
# 
# **Well, for the first point, this is not the case in the train data**
# 
# 
# 
# # An example : 
# Let's see an example here where a card was flagged as fraud but still allowed to make transactions later. This is really weird. It's either something wrong with the defintion of Fraud or there must be something that should be changed about target annotations. 
# 

# In[ ]:


card_ID = '10069 436 150 162'
df_sample=train[train.index.get_level_values('Card_ID')==card_ID][cols]
df_sample


# ### You see ?! isn't that weird thing. A card that was used many time and flagged multiple times as Fraud was still accepted after that like nothing happened !!!

# # How to get list of fraudulant cards ?

# In[ ]:


cols = ['Card_ID', 'isFraud']

# this gives cards_ID with their annotations (fraud or not fraud)
fraud_cards_map=train.reset_index()[cols].groupby('Card_ID').max().to_dict()['isFraud']

# get list of fraud cards
list_of_fraudualant_cards = [k for k in fraud_cards_map.keys() if fraud_cards_map[k]==1]

list_of_fraudualant_cards[:10]


# # Conclusion: 

# I beleive my way to identify unique card is somewhat correct. However, if you want to use it, you have to change the annotations. Which mean that once a card is flagged as Fraud at some point/transaction, all the following point/transactions should be annotated as Fraud. 
