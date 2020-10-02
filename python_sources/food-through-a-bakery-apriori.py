#!/usr/bin/env python
# coding: utf-8

# From  many days I have been thinking if I were to start my own business what would be it?.My Answer is Bakery.I love cooking and I am a foodie.So I will try to get some insights from this Dataset.This is a work in process and I will be updating the Kernel in coming days.If you like my work please do vote.
# 
# The data belongs to a bakery called "The Bread Basket", located in the historic center of Edinburgh. This bakery presents a refreshing offer of Argentine and Spanish products.

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# ## Importing the modules needed 

# In[ ]:


import pandas as pd 
import numpy as np 
import seaborn as sns
import matplotlib.pyplot as plt


# ## Importing the data

# In[ ]:


Bak=pd.read_csv('../input/BreadBasket_DMS.csv')


# ## Displaying the data

# In[ ]:


Bak.head()


# We can see that the dataset has data of Data,Time,Transaction and the item sold at the bakery.

# ## Inspecting the data

# In[ ]:


Bak.loc[Bak['Item']=='NONE',:].head()


# In[ ]:


Bak.loc[Bak['Item']=='NONE',:].count()


# We can see that there are 786 instances in the data set where NONE is mentioned against the item.We have to remove this from our data set.

# ## Dropping none values from the dataset

# In[ ]:


Bak=Bak.drop(Bak.loc[Bak['Item']=='NONE'].index)


# In[ ]:


Bak.loc[Bak['Item']=='NONE',:].count()


# Now we can see that there are no Items with NONE in the dataset

# In[ ]:


Bak['Year'] = Bak.Date.apply(lambda x:x.split('-')[0])
Bak['Month'] = Bak.Date.apply(lambda x:x.split('-')[1])
Bak['Day'] = Bak.Date.apply(lambda x:x.split('-')[2])
Bak['Hour'] =Bak.Time.apply(lambda x:int(x.split(':')[0]))
#df = df.drop(columns='Time')
Bak.head()


# ## Finding out the items that are sold at the Bakery

# In[ ]:


print('Total number of Items sold at the bakery is:',Bak['Item'].nunique())


# In[ ]:


print('List of Items sold at the bakery:')
Bak['Item'].unique()


# In[ ]:


print('List of Items sold at the Bakery:\n')
for item in set(Bak['Item']):
    print(item)


# In[ ]:


print('Ten Most Sold Items At The Bakery')
Bak['Item'].value_counts().head(10)


# In[ ]:


fig, ax=plt.subplots(figsize=(16,7))
Bak['Item'].value_counts().sort_values(ascending=False).head(20).plot.bar(width=0.5,edgecolor='k',align='center',linewidth=2)
plt.xlabel('Food Item',fontsize=20)
plt.ylabel('Number of transactions',fontsize=20)
ax.tick_params(labelsize=20)
plt.title('20 Most Sold Items at the Bakery',fontsize=25)
plt.grid()
plt.ioff()


# Coffee,Bread,Tea,Cake and Pastry are the most sold item at the bakery.Just like in any business we can see that out of 94 items 3 items contribute to close to 50% of the sales.This is Pareto's principle at play in the Bakery Business.

# ## Business during different times of the day

# In[ ]:


Bak.loc[Bak['Time']<'12:00:00','Daytime']='Morning'
Bak.loc[(Bak['Time']>='12:00:00')&(Bak['Time']<'17:00:00'),'Daytime']='Afternoon'
Bak.loc[(Bak['Time']>='17:00:00')&(Bak['Time']<'20:00:00'),'Daytime']='Evening'
Bak.loc[(Bak['Time']>='20:00:00')&(Bak['Time']<'23:50:00'),'Daytime']='Night'


# In[ ]:


fig, ax=plt.subplots(figsize=(16,7))
Bak['Daytime'].value_counts().plot.bar(width=0.5,edgecolor='k',align='center',linewidth=2)
plt.xlabel('Time of the Day',fontsize=20)
plt.ylabel('Number of transactions',fontsize=20)
ax.tick_params(labelsize=20)
plt.title('Sales During Different Period of the Day',fontsize=25)
plt.grid()
plt.ioff()


# Morning upto 12 pm ,Afternoon 12-5 pm,Evening 5-8 pm and Night 8-11.30 pm.We can see from the above plot that the maximum sales happen in the afternoon followed by morning.Evening and Night sales is very less.I am wondering why sales are less in evening and night?.This could be because in cold countries people are reluctant to step out of house in evening and night.

# ## Sales on different days of the week

# In[ ]:


Bak1 = Bak.groupby(['Date']).size().reset_index(name='counts')
Bak1['Day'] = pd.to_datetime(Bak1['Date']).dt.day_name()
#Bak1


# In[ ]:


plt.figure(figsize=(20,15))
plt.subplot(2,2,1)
ax=sns.boxplot(x='Day',y='counts',data=Bak1,width=0.8,linewidth=2)
plt.xlabel('Day of the Week',fontsize=15)
plt.ylabel('Total Sales',fontsize=15)
plt.title('Sales on Different Days of Week',fontsize=20)
ax.tick_params(labelsize=10)
plt.grid()
plt.ioff()


# We can see that maximum sales take place on saturday.This could be beacuse more tourist visit the bakery on weekends.

# ## Sales on different Months of the year

# In[ ]:


Bak['Year'] = Bak.Date.apply(lambda x:x.split('-')[0])
Bak['Month'] = Bak.Date.apply(lambda x:x.split('-')[1])
Bak['Day'] = Bak.Date.apply(lambda x:x.split('-')[2])
Bak['Hour'] =Bak.Time.apply(lambda x:int(x.split(':')[0]))
#df = df.drop(columns='Time')
Bak.head()


# In[ ]:


Bak.loc[Bak.Month == '10', 'Monthly'] = 'Oct'  
Bak.loc[Bak.Month == '11', 'Monthly'] = 'Nov' 
Bak.loc[Bak.Month == '12', 'Monthly'] = 'Dec' 
Bak.loc[Bak.Month == '01', 'Monthly'] = 'Jan' 
Bak.loc[Bak.Month == '02', 'Monthly'] = 'Feb' 
Bak.loc[Bak.Month == '03', 'Monthly'] = 'Mar' 
Bak.loc[Bak.Month == '04', 'Monthly'] = 'Apr' 
#Bak.loc[Bak.Month == '05', 'Monthly'] = 'May' 
#df.loc[df.First_name != 'Bill', 'name_match'] = 'Mis-Match' 
Bak.tail()


# In[ ]:


fig, ax=plt.subplots(figsize=(16,7))
ax=Bak.groupby('Monthly')['Item'].count().sort_values().plot.bar(width=0.5,edgecolor='k',align='center',linewidth=2)
plt.xlabel('Month',fontsize=20)
plt.ylabel('Number of transactions',fontsize=20)
ax.tick_params(labelsize=20)
plt.title('Sales During Different Months of a Year',fontsize=25)
plt.grid()
plt.ioff()


# We see that highest sale was in month of Nov.

# ## Association Rule Mining via Apriori Algorithm 
# Apriori algorithm was proposed by Agrawal and Srikant in 1994. Apriori is designed to operate on databases containing transactions (for example, collections of items bought by customers, or details of a website frequentation or IP addresses).This algorithm is used to find out the association between the items brought at the bakery.Generally transactions have a pattern and finding out the pattern helps us to increase business.For example mothers with babies generally buy milk and diapers together.So at the mall or supermarket if we keep milk and diapers together this can increase our sales.Support,Confidence and Lift are the three important terms used in Apriori Algorithm.
# 
# For example in a mall we have total of 1000 transactions.Out  of this in 150 transcations beer is purchased and 100 times chips is purchased.Beer and Chips are brough together 50 times.We can calculate support,confidence and lift for this case as follows 
# 
# **Support**
# 
# Support (Chips))=Transactions containing(Chips)/Total transcations
# 
# Support(Beer)=100/1000
#                          =10%
#                          
# **Confidence**
# 
# Confidence(Beer-->Chips)=Transcations containing both (Beer and Chips)/Transcations containing Beer
# 
# Confidence(Beer-->Chips)=50/150
#                                               =33.3%
#                                               
# ** Lift **
# 
#  Lift(Beer-->Chips)=Confidence(Beer-->Chips)/Support(Chips)
#  
#   Lift(Beer-->Chips)=33.3/10
#                                   =3.3

# ## Importing the modules for Apriori algorithm

# In[ ]:


from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules


# In[ ]:


hot_encoded_Bak=Bak.groupby(['Transaction','Item'])['Item'].count().unstack().reset_index().fillna(0).set_index('Transaction')
#Above line of code is transfrom data to make items as columns and each transaction as a row and count same Items bought in one transaction but fill other cloumns of the row with 0 to represent item which are not bought.


# In[ ]:


hot_encoded_Bak.head()


# Each row represents a transaction and column represents an item.Zero means no purchase and one means an item is purchased.

# ## Encoding the data set

# In[ ]:


def encode_units(x):
    if x <= 0:
        return 0
    if x >= 1:
        return 1
hot_encoded_Bak = hot_encoded_Bak.applymap(encode_units)


# In[ ]:


frequent_itemsets = apriori(hot_encoded_Bak, min_support=0.01, use_colnames=True)


# In[ ]:


rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1)
rules.head(10)


# In[ ]:


rules[ (rules['lift'] >= 1.17) &
       (rules['confidence'] >= 0.5) ]


# The combition of toast and coffee has the maximum lift of 1.48.This means most cases people buying toast also buy coffee.Support 0.023 means that 2.3% of the transaction contains toast..Confidence of 0.70 means 70% of the toast purchases are accompanied with coffee..Lift 1.47 means that coffee is 1.47 times more likely to be brought when toast is brought.
