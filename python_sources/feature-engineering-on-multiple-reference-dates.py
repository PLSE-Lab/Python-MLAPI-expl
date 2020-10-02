#!/usr/bin/env python
# coding: utf-8

# In many of the discussions and kernels for this competition the reference date was mentioned as end Feb 2018 or 1st March 2018.
# This was based on the fact the the last transaction date in historical transactions is 28th feb 2018. 
# It's also assumed by many that the customer loyalty score is calculated as on this date. 
# **This  is not true . There are multiple reference dates  as explained below ** .
# 

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import time
import sys
import datetime
import gc
import matplotlib.pyplot as plt
import seaborn as sns
from textwrap import wrap
import warnings
warnings.filterwarnings("ignore")
import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[ ]:


train = pd.read_csv("../input/train.csv")
test = pd.read_csv("../input/test.csv")
history = pd.read_csv("../input/historical_transactions.csv",parse_dates=['purchase_date'])
new =pd.read_csv("../input/new_merchant_transactions.csv",parse_dates=['purchase_date'])
history = history.loc[history.authorized_flag =="Y",]


# As explained the historical_transactions.csv and new_merchant_transactions.csv files contain information about each card's transactions. historical_transactions.csv contains up to 3 months' worth of transactions for every card at any of the provided merchant_ids. new_merchant_transactions.csv contains the transactions at new merchants (merchant_ids that this particular card_id has not yet visited) over a period of two months.
# 

# In[ ]:


print(history.purchase_date.min(),history.purchase_date.max())
print(new.purchase_date.min(),new.purchase_date.max())


# From the above result it is clear that seggragation of transactions into new and historical starts from March 2017 .

# In[ ]:


print(new.purchase_date[new.month_lag==1].min(),new.purchase_date[new.month_lag==1].max())
print(new.purchase_date[new.month_lag==2].min(),new.purchase_date[new.month_lag==2].max())


# Month lag is defined in Data Dictionary .xls as the month lag to reference date. From the above results of the earliest purchase dates for month lag of 1 & 2 for new transactions  it looks like there are multiple reference dates.
# As per the definition of month lag month lag of 0 would mean the last month of transactions in  historical transaction data for each card_id.
# Let's look at the earliest and the latest purchase dates for month lag of 0 in historical transactions.

# In[ ]:


print(history.loc[history.month_lag==0,'purchase_date'].min())
print(history.loc[history.month_lag==0,'purchase_date'].max())


# **This confrms that the refernce month could vary from February 2017 to February 2018**. 
#  
# Let's create a dataframe which captures the last purchase date for the month lag ==0 . Column ' refernce_ date' is the first date of the next month based on the month of the last purchase date for the month lag ==0

# In[ ]:


cardreferencedate = history.loc[history.month_lag==0,].groupby('card_id').agg({'purchase_date' : 'max'})
cardreferencedate.reset_index(inplace=True)
cardreferencedate['reference_date'] = cardreferencedate.purchase_date.apply(lambda x :x+ pd.offsets.MonthBegin())
cardreferencedate['reference_date']=cardreferencedate['reference_date'].apply(lambda x: x.replace(hour=0, minute=0, second=0))
cardreferencedate.head()


# In[ ]:


cardreferencedate.loc[:,'reference_date'].value_counts().sort_index().plot(kind='bar')


# About 70% of the card id's have March 1st 2018 as the reference date . Let's probe new transactions for those card id's having a refernce date of  March 1st 2017.

# In[ ]:


new.loc[new.card_id.isin(cardreferencedate.card_id[cardreferencedate.reference_date=='2017-03-01 00:00:00']),]


# In[ ]:


print(new.loc[new.card_id.isin(cardreferencedate.card_id[cardreferencedate.reference_date=='2017-03-01 00:00:00']),'purchase_date'].min())
print(new.loc[new.card_id.isin(cardreferencedate.card_id[cardreferencedate.reference_date=='2017-03-01 00:00:00']),'purchase_date'].max())


# For those card id's with reference date 01- March-2017 the new merchant transactions happen in March & April 2017. It looks like the refernce date is a cut offf date for seggregating historical and new transactions.
# Let's confirm these by further exploratory analysis

# In[ ]:


print("Number of cards with reference date", len(cardreferencedate.index))
print("Number of cards in train",len(train.index))
print("Number of cards in test",len(test.index))
print("Number of unique cards in history",len(history.card_id.unique()))


# About 32,368 cards are missing from the cardreferncedate we created (325,540 unique cards in history - 293,172).
# 
# This implies that these cards don't have any transaction records with month lag==0 .  These card ids's are stored in the array 'Nozeromonthlag'

# In[ ]:


Nozeromonthlag = history.loc[~history.card_id.isin(cardreferencedate.card_id),'card_id'].unique()
len(Nozeromonthlag)


# This finding is confirmed by checking the transactions of card_id 'C_ID_21117571cf'.
# There are no records for monthlag==0

# In[ ]:


history.loc[history.card_id=='C_ID_21117571cf','month_lag'].value_counts()


# The historical transactions for card_id=='C_ID_21117571cf' is below. Last transaction for this acrd is in Dec 2017 and the month lag is -2 which implies the reference date is Mar 01 ,2018

# In[ ]:


history.loc[history.card_id=='C_ID_21117571cf',]


# The new merchant transactions for the card is below. The transactions appear 2 month after the  refernce date. Hence we can assume that this card had no transactions during Jan & Feb 2018 , but had some new merchant transactions in April 2018. 
# 
# **This confirms our assumption that refernce date is a  cut off date for seggregating  historical and new transactions for a specific card.**
# 

# In[ ]:


new.loc[new.card_id=='C_ID_21117571cf',]


# Code below creates a  a data frame ' Nozeromonthlagrefdate' for those card_id's without any  0 values for Month_lag  based on the closest month_lag value to 0.

# In[ ]:


Nozeromonthlagrefdate = history.loc[~history.card_id.isin(cardreferencedate.card_id),].groupby('card_id').agg({'month_lag' : 'max','purchase_date':'max'})
Nozeromonthlagrefdate.reset_index(inplace=True)
Nozeromonthlagrefdate['month_add'] = Nozeromonthlagrefdate.month_lag.apply(lambda x : abs(x))
Nozeromonthlagrefdate['reference_date'] = Nozeromonthlagrefdate.apply(lambda x: x['purchase_date'] + pd.DateOffset(months = x['month_add']), axis=1)
Nozeromonthlagrefdate['reference_date'] = Nozeromonthlagrefdate.reference_date.apply(lambda x :x+ pd.offsets.MonthBegin())
Nozeromonthlagrefdate['reference_date'] = Nozeromonthlagrefdate['reference_date'].apply(lambda x: x.replace(hour=0, minute=0, second=0))


# Let's check whether all card id's with non zero month lag appear in new merchant transaction. As shown below only 26,849 out of 32,638 appear in new list.

# In[ ]:


sum(Nozeromonthlagrefdate.card_id.isin(new.card_id))


# Let's examine the balance 5,789 cards reference dates 

# In[ ]:


Nozeromonthlagrefdate.loc[~Nozeromonthlagrefdate.card_id.isin(new.card_id),'reference_date'].value_counts().plot(kind='bar')


# **It's surprising that these cards have different reference date even though they don't have any new merchant transactions. This merits further probing as this goes against the assumption that  reference date is the cut off date for seggregating  historical and new transactions. .**

# In[ ]:


Nozeromonthlag_nonewtransaction_card_id =Nozeromonthlagrefdate.card_id[~Nozeromonthlagrefdate.card_id.isin(new.card_id)]


# In[ ]:


Nozeromonthlag_nonewtransaction_card_agg=  history.loc[history.card_id.isin(Nozeromonthlag_nonewtransaction_card_id),].groupby(['card_id']).agg({'card_id': 'count','month_lag': ['min','max'],'purchase_date': ['min','max'] })


# In[ ]:


Nozeromonthlag_nonewtransaction_card_agg.head(50)


# From the above results it's not clear why the reference date for these cards are not 1st day of the month succeding their last transaction month . Since they don't have any new merchant transactions  further probing is not possible. But the mean customerloyalty scores of these cards is 0.53 which is significantly higher than -0.39 for  the target mean.

# In[ ]:


print(train.target.mean())
print(train.target[train.card_id.isin(Nozeromonthlag_nonewtransaction_card_id)].mean())


# In[ ]:


sns.boxplot(y=train.target[train.card_id.isin(Nozeromonthlag_nonewtransaction_card_id)])


# The data frames are combined to form a data frame with reference_date value for each card_id in the historical transaction. We will also categorize the cards into four categories.
# * 0- Cards with Reference date based on **Month lag==0**  in historical transactions(cards in the cardsreferencedate dataframe) and having new merchant transactions 
# * 1-  Cards with Reference date based on **Month lag==0**  in historical transactions(cards in the cardsreferencedate dataframe) and  **not** having new merchant transactions .
# * 2-  Cards with Reference date based on **Non zero month lag** in historical transactions(cards in the cardsreferencedate dataframe) and having new merchant transactions.
# * 3-  Cards with Reference date based on **Non zero month lag** in historical transactions(cards in the cardsreferencedate dataframe) and **not** having  new merchant transactions.

# In[ ]:


zeromonthlagmissinginnew= cardreferencedate.card_id[~cardreferencedate.card_id.isin(new.card_id)]
len(zeromonthlagmissinginnew)


# In[ ]:


cardreferencedate.drop(columns='purchase_date',inplace=True)
cardreferencedate['category_month_lag'] =np.where(cardreferencedate.card_id.isin(zeromonthlagmissinginnew),1,0)
Nozeromonthlagrefdate['category_month_lag']= np.where(Nozeromonthlagrefdate.card_id.isin(Nozeromonthlag_nonewtransaction_card_id),3,2)
Nozeromonthlagrefdate.drop(columns=['month_lag','purchase_date','month_add'],inplace=True)
cardreferencedate= pd.concat([cardreferencedate,Nozeromonthlagrefdate])
cardreferencedate.to_csv("Cardreferencedate.csv",index=False)
len(cardreferencedate.index)


# In[ ]:


cardreferencedate = pd.merge(cardreferencedate,train.loc[:,['card_id','target']],on='card_id',how='left')


# In[ ]:


cardreferencedate.head()


# In[ ]:


sns.set(rc={'figure.figsize':(24,12)})
p1= sns.boxplot(x=cardreferencedate.reference_date,y=cardreferencedate.target)
labels = [item.get_text() for item in p1.get_xticklabels()]
labels =[ '\n'.join(wrap(l, 10)) for l in labels ]
p1= p1.set_xticklabels(labels, rotation=90)


# Box plot shows there are no outliers  for cards with 1st March 2017 and 1st April 2017 reference dates. For other refernce dates  there is no major difference.
# 

# In[ ]:


sns.set(rc={'figure.figsize':(24,12)})
p1= sns.boxplot(x=cardreferencedate.category_month_lag,y=cardreferencedate.target)
# labels = [item.get_text() for item in p1.get_xticklabels()]
# labels =[ '\n'.join(wrap(l, 10)) for l in labels ]
# p1= p1.set_xticklabels(labels, rotation=90)


# In[ ]:


sns.set(rc={'figure.figsize':(24,6)})  
plt.subplot(1,2,1)
p1=cardreferencedate.loc[cardreferencedate.card_id.isin(train.card_id),'reference_date'].value_counts().sort_index().plot(kind='bar')
p1.set_title("Credit cardsreference date - Train")
plt.subplot(1,2,2)
p2=cardreferencedate.loc[cardreferencedate.card_id.isin(test.card_id),'reference_date'].value_counts().sort_index().plot(kind='bar')
p2.set_title("Credit cardsreference date - Test")


# In[ ]:


testmissingnew = test.card_id[~test.card_id.isin(new.card_id)]
len(testmissingnew)


# The boxplot of target for card_id's based on the newly created category shows some difference in the mean values. But is this significant enough?
# 
# ***Multiple reference dates throws many questions about the dataset*.**
# 
# * Does those card id's with an earlier reference date  have no transactions with their historical merchants subsequently ? or are they not included. Most of them have  transactions with new merchant id  for upto two months as is evident from the new merchant data.
# * Is the reference date just  a cut off  date to segregate historical transactions and new merchant transactions for every card id.
# * Will including the reference_date and new category  as a feature improve the RMSE ?
# 
# Please  share your thoughts on this.
