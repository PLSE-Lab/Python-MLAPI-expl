#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
get_ipython().run_line_magic('matplotlib', 'inline')
import os
import seaborn as sns
import matplotlib.pyplot as plt
from numpy import percentile
from datetime import datetime
from scipy.stats import pearsonr
print(os.listdir("../input"))


# In[ ]:


print(os.listdir("../input/elo-cards-basic-eda-exposes-data-skewness"))


# In[ ]:


# Read historical transactions data
hist_df = pd.read_csv("../input/elo-merchant-category-recommendation/historical_transactions.csv")
hist_df.head()


# In[ ]:


print(len(hist_df))


# In[ ]:


hist_df['pur_date']=pd.to_datetime(hist_df['purchase_date'])


# In[ ]:


hist_df['pur_date'].max()


# In[ ]:


hist_df['pur_date'].min()


# In[ ]:


#Lets take last 3 months historical data

date1 = datetime.strptime('2017-12-01', '%Y-%m-%d')
date2 = datetime.strptime('2018-02-28', '%Y-%m-%d')


# In[ ]:


hist_df_reduced = hist_df.loc[(hist_df['pur_date']>date1) & (hist_df['pur_date']<date2)]


# In[ ]:


hist_df_reduced.to_csv('hist_DecFeb18.csv')


# In[ ]:


print(len(hist_df_reduced))


# In[ ]:


date1 = datetime.strptime('2017-09-01', '%Y-%m-%d')
date2 = datetime.strptime('2017-11-30', '%Y-%m-%d')
hist_df_reduced_SepNov = hist_df.loc[(hist_df['pur_date']>date1) & (hist_df['pur_date']<date2)]
hist_df_reduced_SepNov.to_csv('hist_SepNov.csv')


# In[ ]:


date1 = datetime.strptime('2017-06-01', '%Y-%m-%d')
date2 = datetime.strptime('2017-08-31', '%Y-%m-%d')
hist_df_reduced_JunAug = hist_df.loc[(hist_df['pur_date']>date1) & (hist_df['pur_date']<date2)]
hist_df_reduced_JunAug.to_csv('hist_JunAug.csv')


# In[ ]:


date1 = datetime.strptime('2017-03-01', '%Y-%m-%d')
date2 = datetime.strptime('2017-05-31', '%Y-%m-%d')
hist_df_reduced_MarMay = hist_df.loc[(hist_df['pur_date']>date1) & (hist_df['pur_date']<date2)]
hist_df_reduced_MarMay.to_csv('hist_MarMay.csv')


# In[ ]:


hist_df_reduced_MarMay.head()


# In[ ]:


date1 = datetime.strptime('2017-01-01', '%Y-%m-%d')
date2 = datetime.strptime('2017-02-28', '%Y-%m-%d')
hist_df_reduced_JanFeb = hist_df.loc[(hist_df['pur_date']>date1) & (hist_df['pur_date']<date2)]
hist_df_reduced_JanFeb.to_csv('hist_JanFeb.csv')


# In[ ]:


hist_df_reduced['month']=hist_df_reduced['pur_date'].dt.to_period('M')


# In[ ]:


#Lets see the trans vol by month and week
#hist_df_red_by_month=hist_df_reduced.groupby(['month']).size()


# In[ ]:


#hist_df_red_by_month.head()


# In[ ]:


#hist_df_reduced['week']=hist_df_reduced['pur_date'].dt.to_period('W')


# In[ ]:


#hist_df_red_by_wk=hist_df_reduced.groupby(['week']).size()


# In[ ]:


#ax = hist_df_red_by_month.plot(kind='bar',figsize=(14,8),fontsize=14)
#ax.set_title("Historical Card Trans Volume By Month",fontsize=28,fontweight='bold')
#ax.set_xlabel("Month",fontsize=14,fontweight='bold')
#ax.set_ylabel("Volume of Trans", fontsize="14",fontweight="bold")
#ax.tick_params(axis='both',which='major',labelsize=18)
#ax.tick_params(axis='both',which='minor',labelsize=18)


# In[ ]:


#Lets look at the transactions by Merchant Category
Trans_by_MerchantCat = pd.DataFrame(hist_df_reduced.groupby(['merchant_category_id']).size())


# In[ ]:


Trans_by_MerchantCat = Trans_by_MerchantCat.rename(columns={0: 'TransCt'})


# In[ ]:


print("Max of Count of Trans by Merch Cat ID:"+str(Trans_by_MerchantCat['TransCt'].max().round(2)))
print("Median of Count of Trans by Merch Cat ID:"+str(round(Trans_by_MerchantCat['TransCt'].median(),2)))
print("Mean of Count of Trans by Merch Cat ID:"+str(round(Trans_by_MerchantCat['TransCt'].mean(),2)))
print("Min of Count of Trans by Merch Cat ID:"+str(round(Trans_by_MerchantCat['TransCt'].min(),2)))


# **We see this is highly skewed and is a wide variation between the vol of Trans for diff Merch Categories, there are Merchant category with 1.5M transactions in last 3 months, as well as some Merchant category with 1 trans. lets explore this a bit further**

# In[ ]:


Trans_by_MerchantCat.sort_values(by='TransCt',ascending=False).head()


# In[ ]:


axarr = Trans_by_MerchantCat['TransCt'].sort_values().plot(kind='bar',figsize=(60,40))
axarr.set_title("Bar chart of count of Trans by Merch Cat Id",fontsize=72,fontweight='bold')
axarr.set_xlabel("Merch Cat Id",fontsize=36)
#axarr[0][0].set_xlim((0,100))
axarr.set_ylabel("Frequency",fontsize=36)
axarr.tick_params(axis='both',which='major',labelsize=36)
axarr.tick_params(axis='both',which='minor',labelsize=36)


# In[ ]:


TransAmt_by_MerchantCat = pd.DataFrame(hist_df_reduced.groupby(['merchant_category_id'])['purchase_amount'].sum())


# In[ ]:


TransAmt_by_MerchantCat.head()


# In[ ]:


print(round(TransAmt_by_MerchantCat['purchase_amount'].max(),2))
print(round(TransAmt_by_MerchantCat['purchase_amount'].min(),2))
print(round(TransAmt_by_MerchantCat['purchase_amount'].median(),2))


# **The data shows highly spread out dist by Merch Cat, median is -185 USD, min is -763206 USD**

# In[ ]:


axarr = TransAmt_by_MerchantCat['purchase_amount'].sort_values().plot(kind='bar',figsize=(40,40))
axarr.set_title("Bar chart of Amount of Trans by Merch Cat Id",fontsize=72,fontweight='bold')
axarr.set_xlabel("Merch Cat Id",fontsize=36)
#axarr[0][0].set_xlim((0,100))
axarr.set_ylabel("Frequency",fontsize=36)
axarr.tick_params(axis='both',which='major',labelsize=36)
axarr.tick_params(axis='both',which='minor',labelsize=36)


# **The above shows that the Trans Amt is Negative for half of the Merchant categories. What does this mean, does it mean for these Merchant categories, the Users made payment to the Card, but never purchased !! Need to be explored from merchants.csv**

# **Lets group the transactions by card_id and sum the purchase_amounts**

# In[ ]:


TransAmt_by_Card = pd.DataFrame(hist_df_reduced.groupby(['card_id'])['purchase_amount'].sum())


# In[ ]:


ax=plt.hist(TransAmt_by_Card['purchase_amount'],bins=5)
plt.xlim(-2000,150000)
plt.show()


# In[ ]:


print("Max amt by Card:"+str(TransAmt_by_Card.max().round(2)))
print("Min amt by Card:"+str(TransAmt_by_Card.min().round(2)))
print("Median amt by Card:"+str(TransAmt_by_Card.median().round(2)))


# In[ ]:


data = TransAmt_by_Card['purchase_amount']
percentiles = percentile(data,[10,25,30,50,75,90, 95, 98, 99, 99.95])
print(percentiles.round(2))


# **This is a strange result, the 10 percentile value is -46 and 90 percentile value is -1, but max is 134755.**

# In[ ]:


print(len(TransAmt_by_Card))


# In[ ]:


TransAmt_by_Card.sort_values(by='purchase_amount',ascending=False)[:50]


# In[ ]:


TransAmt_by_Card.sort_values(by='purchase_amount',ascending=False)[27000:27100]


# **Hard to believe but true,  99 percentile of the data is 11 USD or below, above 99.5 percentile mark we get values like 3500 USD going all the way upto max 134755 USD**

# In[ ]:


TransAmt_by_Card_red = TransAmt_by_Card.loc[(TransAmt_by_Card['purchase_amount']>=100)]


# In[ ]:


print(len(TransAmt_by_Card_red))


# **This means only 405 cards have made total transactions worth USD 100 or more in last 3 months!!**

# In[ ]:


ax=plt.hist(TransAmt_by_Card_red['purchase_amount'],bins=100)
plt.xlim(-2000,25000)
plt.show()


# **Lets explore Trans amount by Merchant Id for a Card id**

# In[ ]:


TransAmt_by_CardbyMerch = pd.DataFrame(hist_df_reduced.groupby(['card_id','merchant_id'])['purchase_amount'].sum())


# In[ ]:


data = TransAmt_by_CardbyMerch['purchase_amount']
percentiles = percentile(data,[10,25,30,50,75,90, 95, 98, 99, 99.95])
print(percentiles.round(2))


# **This is a strange result, 99 percentile of Card-Merch combination is having purch value <= USD 1.77**

# **Lets look at number of transactions by Card id and Merchant id**

# In[ ]:


TransVol_by_CardbyMerch = pd.DataFrame(hist_df_reduced.groupby(['card_id','merchant_id']).size())


# In[ ]:


data = TransVol_by_CardbyMerch
percentiles = percentile(data,[10,25,30,50,75,90, 95, 98, 99, 99.95])
print(percentiles)


# **The above shows a strange result. For 90 percentile of the card-Merch combination have vol <=4, and 75 percentile of card-Merch has 2 or less transactions**

# In[ ]:


print(len(TransVol_by_CardbyMerch))


# In[ ]:


data = Trans_by_MerchantCat['TransCt']
percentiles = percentile(data,[10,25,30,50,75,90])
print(percentiles.round())


# In[ ]:


Trans_by_MerchantCat_gt25k=Trans_by_MerchantCat['TransCt'].loc[(Trans_by_MerchantCat['TransCt'] > 25000)]


# In[ ]:


len(Trans_by_MerchantCat)


# In[ ]:


len(Trans_by_MerchantCat_gt25k)


# In[ ]:


axarr = Trans_by_MerchantCat_gt25k.sort_values().plot(kind='bar',figsize=(60,40))
axarr.set_title("Bar chart of count of Trans by Merch Cat Id (>25k)",fontsize=72,fontweight='bold')
axarr.set_xlabel("Merch Cat Id",fontsize=36)
#axarr[0][0].set_xlim((0,100))
axarr.set_ylabel("Frequency",fontsize=36)
axarr.tick_params(axis='both',which='major',labelsize=36)
axarr.tick_params(axis='both',which='minor',labelsize=36)


# In[ ]:


HighVolMerchCat = Trans_by_MerchantCat_gt25k.index


# In[ ]:


HighVolMerchCat


# **Read train.csv to be combined with Historical and New Merch transactions**

# In[ ]:


#read train df
train_df=pd.read_csv('../input/elo-merchant-category-recommendation/train.csv')


# In[ ]:


train_df.head()


# In[ ]:


print(len(train_df))


# In[ ]:


train_df['first_active_month']=pd.to_datetime(train_df['first_active_month'])


# In[ ]:


train_df['first_active_month'].min()


# In[ ]:


train_df['first_active_month'].max()


# In[ ]:


print(len(train_df['card_id'].unique()))


# **Since the length of unique card ids = length of dataset, we conclude train_df has one row per card id, and hence the loyalty score is per card id**

# In[ ]:


train_df['month']=train_df['first_active_month'].dt.to_period('M')


# In[ ]:


train_df_by_month=train_df.groupby(['month']).size()


# In[ ]:


ax = train_df_by_month.plot(kind='bar',figsize=(14,8),fontsize=14)
ax.set_title("Training dataset - Card Trans Volume By First Active Month",fontsize=28,fontweight='bold')
ax.set_xlabel("First Active Month",fontsize=14,fontweight='bold')
ax.set_ylabel("Volume of Trans", fontsize="14",fontweight="bold")
ax.tick_params(axis='both',which='major',labelsize=18)
ax.tick_params(axis='both',which='minor',labelsize=18)


# In[ ]:


print(len(train_df['month'].unique()))


# In[ ]:


#train_df_by_month_mod = train_df_by_month.loc[train_df_by_month > 250]


# In[ ]:


#first_act_mth_list = train_df_by_month_mod.index


# In[ ]:


#first_act_mth_list


# In[ ]:


#train_df_red = train_df.loc[train_df.month.isin(first_act_mth_list)]


# In[ ]:


#print(len(train_df_red))


# In[ ]:


#train_df_red.head()


# **Lets read New Merchant Transactions and do similar study as in case of Historical Transactions**

# In[ ]:


new_merch_df = pd.read_csv("../input/elo-merchant-category-recommendation/new_merchant_transactions.csv")
new_merch_df.head()


# In[ ]:


new_merch_df['pur_date']=pd.to_datetime(new_merch_df['purchase_date'])


# In[ ]:


new_merch_df['pur_date'].max()


# In[ ]:


new_merch_df['pur_date'].min()


# In[ ]:


new_merch_df['month']=new_merch_df['pur_date'].dt.to_period('M')


# In[ ]:


#new_merch_df['wk']=new_merch_df['pur_date'].dt.to_period('W')


# In[ ]:


new_merch_df_by_month = new_merch_df.groupby(['month']).size()


# In[ ]:


#new_merch_df_by_wk = new_merch_df.groupby(['wk']).size()


# In[ ]:


ax = new_merch_df_by_month.plot(kind='bar',figsize=(14,8),fontsize=14)
ax.set_title("New Merchant Card Trans Volume By Month",fontsize=28,fontweight='bold')
ax.set_xlabel("Month",fontsize=14,fontweight='bold')
ax.set_ylabel("Volume of Trans", fontsize="14",fontweight="bold")
ax.tick_params(axis='both',which='major',labelsize=18)
ax.tick_params(axis='both',which='minor',labelsize=18)


# In[ ]:


#Group by Merchant Category Id
Trans_by_MerchantCat_new = pd.DataFrame(new_merch_df.groupby(['merchant_category_id']).size())


# In[ ]:


Trans_by_MerchantCat_new = Trans_by_MerchantCat_new.rename(columns={0: 'TransCt'})


# In[ ]:


axarr = Trans_by_MerchantCat_new['TransCt'].sort_values().plot(kind='bar',figsize=(60,40))
axarr.set_title("New Merchant Count of Trans by Merch Cat Id",fontsize=72,fontweight='bold')
axarr.set_xlabel("Merch Cat Id",fontsize=36)
#axarr[0][0].set_xlim((0,100))
axarr.set_ylabel("Frequency",fontsize=36)
axarr.tick_params(axis='both',which='major',labelsize=36)
axarr.tick_params(axis='both',which='minor',labelsize=36)


# In[ ]:


data = Trans_by_MerchantCat_new['TransCt']
percentiles = percentile(data,[25,30,50,75,90])
print(percentiles)


# In[ ]:


Trans_by_MerchantCat_new_gt5k=Trans_by_MerchantCat_new['TransCt'].loc[(Trans_by_MerchantCat_new['TransCt'] > 5000)]


# In[ ]:


HighVolMerchCat_new = Trans_by_MerchantCat_new_gt5k.index


# In[ ]:


HighVolMerchCat_new


# In[ ]:


c = set(HighVolMerchCat) & set(HighVolMerchCat_new)  #  & calculates the intersection.
print(len(c))


# In[ ]:


HighVolMerchComb = set(HighVolMerchCat) | set(HighVolMerchCat_new)


# In[ ]:


print(len(HighVolMerchComb))


# In[ ]:


#new_merch_df_red = new_merch_df.loc[(new_merch_df.merchant_category_id.isin(HighVolMerchCat_new))]


# In[ ]:


#print(len(new_merch_df_red))


# In[ ]:


print(len(new_merch_df['card_id'].unique()))


# In[ ]:


TransAmt_by_CardnewMerch = pd.DataFrame(new_merch_df.groupby(['card_id','merchant_id'])['purchase_amount'].sum())


# In[ ]:


#TransAmt_by_CardnewMerch.hist()
ax=plt.hist(TransAmt_by_CardnewMerch['purchase_amount'],bins=50)
plt.xlim(-5,25)
plt.show()


# In[ ]:


data = TransAmt_by_CardnewMerch['purchase_amount']
percentiles = percentile(data,[10,25,30,50,75,90, 95, 98, 99, 99.95])
print(TransAmt_by_CardnewMerch['purchase_amount'].min())
print(TransAmt_by_CardnewMerch['purchase_amount'].max())
print(percentiles)


# **Lets create Trans Vol grouped by Card Id and Merchant Id, ie for each card, the number of transactions done at same Merchant**

# In[ ]:


TransVol_by_CardnewMerch = pd.DataFrame(new_merch_df.groupby(['card_id','merchant_id']).size())


# In[ ]:


TransVol_by_CardnewMerch = TransVol_by_CardnewMerch.rename(columns={0: 'TransCt'})


# In[ ]:


TransVol_by_CardnewMerch.sort_values(by='TransCt',ascending=False).head()


# In[ ]:


TransVol_by_CardnewMerch = TransVol_by_CardnewMerch.reset_index()


# In[ ]:


TransVol_by_CardnewMerch.head()


# In[ ]:


data = TransVol_by_CardnewMerch['TransCt']
percentiles = percentile(data,[10,25,30,50,75,90, 95, 98, 99, 99.95,99.98])
print(TransVol_by_CardnewMerch['TransCt'][0].min())
print(TransVol_by_CardnewMerch['TransCt'][0].max())
print(percentiles)


# In[ ]:


merchant_df = pd.read_csv('../input/elo-merchant-category-recommendation/merchants.csv')


# In[ ]:


len(merchant_df)


# In[ ]:


len(merchant_df['merchant_id'].unique())


# **This is strange, merchant csv is supposed to be by merchant_id, but there are few duplicates, lets remove the duplicates and take unique values since diff is small**

# In[ ]:


merchantList=merchant_df['merchant_id'].unique()


# In[ ]:


merchant_df=merchant_df.drop_duplicates(subset=['merchant_id'])


# In[ ]:


len(merchant_df)


# In[ ]:


len(merchant_df['merchant_id'].unique())


# In[ ]:


merchant_df= merchant_df.reset_index(drop=True)


# In[ ]:


TransVol_by_CardnewMerch = pd.merge(TransVol_by_CardnewMerch, merchant_df, how = 'inner', left_on = ['merchant_id'], right_on=['merchant_id'])


# In[ ]:


TransVol_by_CardnewMerch.head()


# **Lets find out how many count of Transactions are done at High Vol Merch Categories**

# In[ ]:


TransVol_by_CardnewMerchHighVol = TransVol_by_CardnewMerch.loc[(TransVol_by_CardnewMerch.merchant_category_id.isin(HighVolMerchCat_new))]


# In[ ]:


len(TransVol_by_CardnewMerch)


# In[ ]:


len(TransVol_by_CardnewMerchHighVol)


# In[ ]:


TransVol_by_CardnewHighVolMerch=TransVol_by_CardnewMerchHighVol.groupby(['card_id']).size()


# In[ ]:


TransVol_by_CardnewHighVolMerch.head()


# In[ ]:


TransVol_by_CardnewHighVolMerch=pd.DataFrame(TransVol_by_CardnewHighVolMerch)
TransVol_by_CardnewHighVolMerch = TransVol_by_CardnewHighVolMerch.reset_index()
TransVol_by_CardnewHighVolMerch.head()


# In[ ]:


TransVol_by_CardnewHighVolMerch = TransVol_by_CardnewHighVolMerch.rename(columns={0: 'TransCt'})


# In[ ]:


TransVol_by_CardnewHighVolMerch.sort_values(by='TransCt',ascending=False).head()


# In[ ]:


TransVol_by_CardnewHighVolMerch = TransVol_by_CardnewHighVolMerch.rename(columns={'TransCt': 'NewHighVolMerchTransCt'})


# In[ ]:


TransVol_by_CardnewHighVolMerch.sort_values(by='NewHighVolMerchTransCt',ascending=False).head()


# **Lets group Trans Volume by Card for New Merchants with number of Merchants at which transaction done for each Card**

# In[ ]:


TransVol_by_CardnewMerchCt=TransVol_by_CardnewMerch.groupby(['card_id']).size()


# In[ ]:


TransVol_by_CardnewMerchCt.sort_values(ascending=False).head()


# In[ ]:


print(TransVol_by_CardnewMerchCt.min())
print(TransVol_by_CardnewMerchCt.median())
print(TransVol_by_CardnewMerchCt.max())


# In[ ]:


TransVol_by_CardnewMerchCt = TransVol_by_CardnewMerchCt.reset_index()


# In[ ]:


TransVol_by_CardnewMerchCt = TransVol_by_CardnewMerchCt.rename(columns={0: 'NewMerchTransCt'})


# In[ ]:


TransVol_by_CardnewMerchCt.head()


# In[ ]:


#hist_df_reduced_merch = hist_df_reduced.loc[(hist_df_reduced.merchant_category_id.isin(HighVolMerchCat))]


# In[ ]:


#print(len(hist_df_reduced_merch))


# In[ ]:


#print(len(hist_df_reduced_merch['card_id'].unique()))


# In[ ]:


#hist_df_reduced_merch.head()


# **Lets explore volume of Transactions by City**

# In[ ]:


print(len(hist_df_reduced['city_id'].unique()))


# In[ ]:


#Lets group transactions by city
hist_df_by_city = hist_df_reduced.groupby(['city_id']).size()


# In[ ]:


print(hist_df_by_city.max())
print(hist_df_by_city.min())
print(hist_df_by_city.median())


# **Again its skewed data with some city having 1.6M transactions in past 3 months, whereas some city having only 20 transactions**

# In[ ]:


axarr = hist_df_by_city.sort_values().plot(kind='bar',figsize=(60,40))
axarr.set_title("Bar chart of count of Trans by City Id",fontsize=72,fontweight='bold')
axarr.set_xlabel("City Id",fontsize=36)
#axarr[0][0].set_xlim((0,100))
axarr.set_ylabel("Frequency",fontsize=36)
axarr.tick_params(axis='both',which='major',labelsize=36)
axarr.tick_params(axis='both',which='minor',labelsize=36)


# In[ ]:


data = hist_df_by_city
percentiles = percentile(data, [10,25,30,40, 50, 60, 75, 90])


# In[ ]:


print(percentiles)


# In[ ]:


#hist_df_by_city=hist_df_by_city.loc[(hist_df_by_city>15000)]


# In[ ]:


#city_list = hist_df_by_city.index


# In[ ]:


#hist_df_reduced_merch_city = hist_df_reduced_merch.loc[hist_df_reduced_merch.city_id.isin(city_list)]


# In[ ]:


#print(len(hist_df_reduced_merch_city))


# In[ ]:


new_merch_df_city = new_merch_df.groupby(['city_id']).size()


# In[ ]:


data = new_merch_df_city
percentiles = percentile(data, [10,25,30,40, 50, 60, 75, 90])


# In[ ]:


print(percentiles)


# In[ ]:


axarr = new_merch_df_city.sort_values().plot(kind='bar',figsize=(60,40))
axarr.set_title("Bar chart of count of Trans by City Id, New Merch",fontsize=72,fontweight='bold')
axarr.set_xlabel("City Id",fontsize=36)
#axarr[0][0].set_xlim((0,100))
axarr.set_ylabel("Frequency",fontsize=36)
axarr.tick_params(axis='both',which='major',labelsize=36)
axarr.tick_params(axis='both',which='minor',labelsize=36)


# In[ ]:


#new_merch_df_city=new_merch_df_city.loc[(new_merch_df_city>5000)]


# In[ ]:


#city_list_new = new_merch_df_city.index


# In[ ]:


#new_merch_df_city = new_merch_df_red.loc[new_merch_df_red.city_id.isin(city_list_new)]


# In[ ]:


#new_merch_df_city.head()


# In[ ]:


#en(new_merch_df_city)


# **Lets group the transactions by Card summing up the purchase amounts, and getting max, median, std dev also, however we will not use all this in correlation analysis as in prev study we found correlation only with New Merch Max Trans Amt by Card**

# In[ ]:


new_merch_trans_sum_bycard = new_merch_df.groupby(['card_id'])['purchase_amount'].sum()
new_merch_trans_max_bycard = new_merch_df.groupby(['card_id'])['purchase_amount'].max()
new_merch_trans_median_bycard = new_merch_df.groupby(['card_id'])['purchase_amount'].median()
new_merch_trans_std_bycard = new_merch_df.groupby(['card_id'])['purchase_amount'].std()


# In[ ]:


new_merch_trans_sum_bycard.sort_values(ascending=False).head()


# In[ ]:


hist_trans_sum_bycard = hist_df_reduced.groupby(['card_id'])['purchase_amount'].sum()
hist_trans_max_bycard = hist_df_reduced.groupby(['card_id'])['purchase_amount'].max()
hist_trans_median_bycard = hist_df_reduced.groupby(['card_id'])['purchase_amount'].median()
hist_trans_std_bycard = hist_df_reduced.groupby(['card_id'])['purchase_amount'].std()


# In[ ]:


hist_trans_sum_bycard.sort_values(ascending=False).head()


# In[ ]:


Card_union = set(new_merch_trans_sum_bycard.index) | set(hist_trans_sum_bycard.index)


# In[ ]:


len(Card_union)


# In[ ]:


Card_intersection = set(new_merch_trans_sum_bycard.index) & set(hist_trans_sum_bycard.index)


# In[ ]:


len(Card_intersection)


# In[ ]:


#new_merch_df_city.to_csv('new_merch_red.csv')


# In[ ]:


#hist_df_reduced_merch_city.to_csv('hist_trans_red.csv')


# In[ ]:


#train_df_red.to_csv('train_red.csv')


# In[ ]:


hist_df_reduced.head()


# In[ ]:


hist_df_auth_ct_bycard=hist_df_reduced.groupby('card_id')['authorized_flag'].apply(lambda x: (x=='Y').sum()).reset_index(name='Auth_count')


# In[ ]:


hist_df_auth_ct_bycard.head()


# In[ ]:


hist_df_ref_ct_bycard=hist_df_reduced.groupby('card_id')['authorized_flag'].apply(lambda x: (x=='N').sum()).reset_index(name='Ref_count')


# In[ ]:


hist_df_ref_ct_bycard.head()


# In[ ]:


print(len(hist_trans_sum_bycard.index))


# In[ ]:


print(len(new_merch_trans_sum_bycard.index))


# **We ignore the payment amounts now, as correlation analysis in previous iteration didnt give any result**

# In[ ]:


#df_card_all = pd.concat([new_merch_trans_sum_bycard, hist_trans_sum_bycard])\
#       .groupby('card_id').sum().reset_index()


# In[ ]:


#df_card_all.head()


# In[ ]:


#print(len(df_card_all))


# ****The above is a verification that the operation is ok since the count matches the set union value above****

# In[ ]:


new_merch_trans_max_bycard = pd.DataFrame(new_merch_trans_max_bycard)
new_merch_trans_median_bycard = pd.DataFrame(new_merch_trans_median_bycard)
new_merch_trans_std_bycard = pd.DataFrame(new_merch_trans_std_bycard)
#new_merch_trans_max_bycard.head()


# In[ ]:


hist_trans_max_bycard = pd.DataFrame(hist_trans_max_bycard)
hist_trans_median_bycard = pd.DataFrame(hist_trans_median_bycard)
hist_trans_std_bycard = pd.DataFrame(hist_trans_std_bycard)


# In[ ]:


new_merch_trans_max_bycard = new_merch_trans_max_bycard.rename(columns={'purchase_amount': 'NewMerch_Max_Amt'})
new_merch_trans_median_bycard = new_merch_trans_median_bycard.rename(columns={'purchase_amount': 'NewMerch_Median_Amt'})
new_merch_trans_std_bycard = new_merch_trans_std_bycard.rename(columns={'purchase_amount': 'NewMerch_StdDev_Amt'})


# In[ ]:


hist_trans_max_bycard = hist_trans_max_bycard.rename(columns={'purchase_amount': 'HistMerch_Max_Amt'})
hist_trans_median_bycard = hist_trans_median_bycard.rename(columns={'purchase_amount': 'HistMerch_Median_Amt'})
hist_trans_std_bycard = hist_trans_std_bycard.rename(columns={'purchase_amount': 'HistMerch_StdDev_Amt'})


# ****We ignore sum of installments by card as prev iteration showed no correlation**

# In[ ]:


#hist_instalment_by_card = hist_df_reduced.groupby(['card_id'])['installments'].sum()


# In[ ]:


#new_instalment_by_card = new_merch_df.groupby(['card_id'])['installments'].sum()


# In[ ]:


#df_card_all_instalment = pd.concat([hist_instalment_by_card, new_instalment_by_card])\
#       .groupby('card_id').sum().reset_index()


# In[ ]:


#df_card_all_instalment.head()


# **We merge the train dataset with the historical and new merchant transactions data to add purchase amount and installments**

# In[ ]:


#train_df_merged = pd.merge(train_df,df_card_all, how = 'inner', left_on = ['card_id'], right_on=['card_id'])


# In[ ]:


#train_df_merged = pd.merge(train_df_merged,df_card_all_instalment, how = 'inner', left_on = ['card_id'], right_on=['card_id'])


# In[ ]:


train_df_merged = pd.merge(train_df,new_merch_trans_max_bycard, how = 'inner', left_on = ['card_id'], right_on = ['card_id'])


# In[ ]:


train_df_merged = pd.merge(train_df_merged,TransVol_by_CardnewHighVolMerch, how = 'inner', left_on = ['card_id'], right_on = ['card_id'])


# In[ ]:


train_df_merged = pd.merge(train_df_merged,TransVol_by_CardnewMerchCt, how = 'inner', left_on = ['card_id'], right_on = ['card_id'])


# In[ ]:


#train_df_merged = pd.merge(train_df_merged,new_merch_trans_median_bycard, how = 'inner', left_on = ['card_id'], right_on = ['card_id'])
#train_df_merged = pd.merge(train_df_merged,new_merch_trans_std_bycard, how = 'inner', left_on = ['card_id'], right_on = ['card_id'])


# In[ ]:


train_df_merged.head()


# In[ ]:


#train_df_merged = pd.merge(train_df_merged,hist_trans_max_bycard, how = 'inner', left_on = ['card_id'], right_on = ['card_id'])
#train_df_merged = pd.merge(train_df_merged,hist_trans_median_bycard, how = 'inner', left_on = ['card_id'], right_on = ['card_id'])
#train_df_merged = pd.merge(train_df_merged,hist_trans_std_bycard, how = 'inner', left_on = ['card_id'], right_on = ['card_id'])


# In[ ]:


train_df_merged.head()


# In[ ]:


#train_df_merged = pd.merge(train_df_merged,TransVol_by_CardMerchm, how = 'inner', left_on = ['card_id'], right_on = ['card_id'])


# In[ ]:


train_df_merged = pd.merge(train_df_merged,hist_df_auth_ct_bycard, how = 'inner', left_on = ['card_id'], right_on = ['card_id'])


# In[ ]:


train_df_merged = pd.merge(train_df_merged,hist_df_ref_ct_bycard, how = 'inner', left_on = ['card_id'], right_on = ['card_id'])


# In[ ]:


train_df_merged.head()


# In[ ]:


#train_df_merged.to_csv('train_df_merged.csv')


# In[ ]:


#do some data cleansing to make sure the correlation analysis runs without any error
#m=train_df_merged["NewMerch_StdDev_Amt"].isnull().any()
#print(m[m])
#m=train_df_merged["HistMerch_StdDev_Amt"].isnull().any()
#print(m[m])


# In[ ]:


#train_df_merged["NewMerch_StdDev_Amt"].fillna(train_df_merged["NewMerch_StdDev_Amt"].mean())
#train_df_merged["NewMerch_StdDev_Amt"] = train_df_merged["NewMerch_StdDev_Amt"].apply(lambda x: x if not pd.isnull(x) else train_df_merged["NewMerch_StdDev_Amt"].mean())
#m=train_df_merged["NewMerch_StdDev_Amt"].isnull().any()
#print(m[m])
#train_df_merged["HistMerch_StdDev_Amt"].fillna(train_df_merged["HistMerch_StdDev_Amt"].mean())
#train_df_merged["HistMerch_StdDev_Amt"] = train_df_merged["HistMerch_StdDev_Amt"].apply(lambda x: x if not pd.isnull(x) else train_df_merged["NewMerch_StdDev_Amt"].mean())
#m=train_df_merged["HistMerch_StdDev_Amt"].isnull().any()
#print(m[m])


# In[ ]:


#
#corr, _ = pearsonr(train_df_merged['target'],train_df_merged['installments'])
#print('Pearsons correlation betweeen target and installments: %.3f' % corr)


# In[ ]:


#corr, _ = pearsonr(train_df_merged['target'],train_df_merged['purchase_amount'])
#print('Pearsons correlation betweeen target and sum purchase amount by card id: %.3f' % corr)


# In[ ]:


corr, _ = pearsonr(train_df_merged['target'],train_df_merged['NewMerch_Max_Amt'])
print('Pearsons correlation betweeen target and New Merch Max puch amt: %.3f' % corr)


# In[ ]:


corr, _ = pearsonr(train_df_merged['target'],train_df_merged['NewMerchTransCt'])
print('Pearsons correlation betweeen target and New Merch Count: %.3f' % corr)


# In[ ]:


corr, _ = pearsonr(train_df_merged['target'],train_df_merged['NewHighVolMerchTransCt'])
print('Pearsons correlation betweeen target and New Merch Count: %.3f' % corr)


# In[ ]:


corr, _ = pearsonr(train_df_merged['target'],train_df_merged['Auth_count'])
print('Pearsons correlation betweeen target and Authorised Trans Count: %.3f' % corr)


# In[ ]:


corr, _ = pearsonr(train_df_merged['target'],train_df_merged['Ref_count'])
print('Pearsons correlation betweeen target and Refused Trans Count: %.3f' % corr)


# **We explore the feature 1 , feature 2 and feature 3 values using bar charts**

# In[ ]:


train_df_merged['feature_1'].value_counts().sort_values().plot(kind='bar')


# In[ ]:


train_df_merged['feature_2'].value_counts().sort_values().plot(kind='bar')


# In[ ]:


train_df_merged['feature_3'].value_counts().sort_values().plot(kind='bar')


# **We notice these are categorical variables. We now explore correlation between target and the purchase amount and installments**

# In[ ]:


train_df_corr = train_df_merged[['NewMerch_Max_Amt','NewMerchTransCt','NewHighVolMerchTransCt','Auth_count','Ref_count','target']]


# In[ ]:


corr=train_df_corr.corr()


# In[ ]:


corr


# In[ ]:


# Generate a mask for the upper triangle
mask = np.zeros_like(corr, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True
# Set up the matplotlib figure
f, ax = plt.subplots(figsize=(11, 9))
# Generate a custom diverging colormap
cmap = sns.diverging_palette(220, 10, as_cmap=True)
# Draw the heatmap with the mask and correct aspect ratio
sns.heatmap(corr, mask=mask, cmap=cmap, vmax=1, center=0,
            square=True, linewidths=.5, cbar_kws={"shrink": .5})
plt.show()


# **Added NewMerchTransCt and NewHighVolMerchTransCt as two features**

# In[ ]:


import seaborn as sns
sns.set(style="ticks")
sns.pairplot(train_df_merged[['target','feature_1']])


# **Shows there are few outliers in target (-30), but these occur for all values of feature_1**

# In[ ]:


import seaborn as sns
sns.set(style="ticks")
sns.pairplot(train_df_merged[['target','feature_2']])


# **Feature 2 does not show any correlation with target, the value 3 starts slightly higher, but this can be ignored as there is a huge spread in target value for all three values**

# In[ ]:


import seaborn as sns
sns.set(style="ticks")
sns.pairplot(train_df_merged[['target','feature_3']])


# **Feature 2 also does not show much correlation with target**

# In[ ]:


train_df_merged['target'].hist()


# **The above shows Outliers in target value, few values at -30, should be ignored**

# In[ ]:


ax=plt.hist(train_df_merged['target'],bins=100)
plt.xlim(-7.5,7.5)
plt.show()


# In[ ]:


print(len(train_df_merged.loc[(train_df_merged['target'] < -7)]))


# In[ ]:


print(len(train_df_merged.loc[(train_df_merged['target'] > 7)]))


# In[ ]:


print(len(train_df_merged))


# **Lets remove the extreme Outliers from target value, keeping it between -7 and +7 and explore the Correlation matrix again**

# In[ ]:


train_df_merged = train_df_merged.loc[(train_df_merged['target'] >= -7)]


# In[ ]:


print(len(train_df_merged))


# In[ ]:


train_df_merged = train_df_merged.loc[(train_df_merged['target'] <= 7)]


# In[ ]:


train_df_merged.to_csv('train_df_merged.csv')


# In[ ]:


print(len(train_df_merged))


# In[ ]:


train_df_corr = train_df_merged[['NewMerch_Max_Amt','NewMerchTransCt','NewHighVolMerchTransCt','Auth_count','Ref_count','target']]


# In[ ]:


corr=train_df_corr.corr()
corr


# In[ ]:


# Generate a mask for the upper triangle
mask = np.zeros_like(corr, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True
# Set up the matplotlib figure
f, ax = plt.subplots(figsize=(11, 9))
# Generate a custom diverging colormap
cmap = sns.diverging_palette(220, 10, as_cmap=True)
# Draw the heatmap with the mask and correct aspect ratio
sns.heatmap(corr, mask=mask, cmap=cmap, vmax=1, center=0,
            square=True, linewidths=.5, cbar_kws={"shrink": .5})
plt.show()


# **Shows target has a slight negative correlation with NewMerch_Max_ Amt and NewMerchTransCt**

# In[ ]:


#merchant_df = pd.read_csv('../input/merchants.csv')


# In[ ]:


merchant_df.head()


# In[ ]:


print(len(merchant_df['merchant_id'].unique()))


# In[ ]:


print(len(merchant_df['merchant_group_id'].unique()))


# In[ ]:


print(len(merchant_df['merchant_category_id'].unique()))


# In[ ]:


merchant_df['subsector_id'].value_counts()


# In[ ]:


merchant_df['category_1'].value_counts()


# In[ ]:


merchant_df['most_recent_purchases_range'].value_counts()


# In[ ]:


merchant_df['most_recent_sales_range'].value_counts()

