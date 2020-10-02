#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
print(os.listdir("../input"))


# In[ ]:


import matplotlib.pyplot as plt
import seaborn as sns


# In[ ]:


df = pd.read_csv('../input/BreadBasket_DMS.csv')
df.head()


# In[ ]:


df.Item.value_counts()


# In[ ]:


df.isnull().any()


# In[ ]:


print("unique items :" ,df.Item.nunique())


# In[ ]:


df.Item.unique()


# In[ ]:


df.loc[(df['Item']=="NONE")].head() 


# In[ ]:


len(df.loc[(df['Item']=="NONE")])


# In[ ]:


df.drop(df[df['Item']=='NONE'].index, inplace=True)


# In[ ]:


df.info()


# In[ ]:


df.shape


# ## Feature Engineering

# In[ ]:


df["date"] = pd.to_datetime(df['Date'])
df["dayname"] = df["date"].dt.day_name()
df.drop("date", axis=1, inplace = True)
df.head()


# In[ ]:


df["year"],df["month"],df["day"]=df["Date"].str.split('-').str
df["hour"],df["minute"],df["second"]=df["Time"].str.split(":").str
df.drop("Date", axis=1, inplace = True)
df.drop("Time", axis=1, inplace = True)
df.head()


# In[ ]:


df.info()


# In[ ]:


#season
df["month"]=df["month"].astype(int)
df.loc[(df['month']==12),'season']="winter"
df.loc[(df['month']>=1)&(df['month']<=3),'season']="winter"
df.loc[(df['month']>3)&(df['month']<=6),'season']="spring"
df.loc[(df['month']>6)&(df['month']<=9),'season']="summer"
df.loc[(df['month']>9)&(df['month']<=11),'season']="fall"

df.head()


# ## VISUALIZATION

# ### Item top sales

# In[ ]:


plt.figure(figsize=(20,50))
df['Item'].value_counts().sort_values().plot.barh(title='Top Item Sales',grid=True)


# ### From the above plot we can conclude that as below 
# *   Top 5 purchased Items are 
#     1. Coffee
#     2. Bread
#     3. Tea
#     4. cake
#     5. pastry
#     
#     
#  *  Least 5 purchased Items are :
#      1. Polenta
#      2. Bacon
#      3. Olum& Polenta
#      4. Adjustments
#      5. Raw Bars 

# In[ ]:


plt.figure(figsize=(20,10))
sns.countplot(x='dayname',data=df).set_title('Pattern of Transcation Trend Throughout The Week',fontsize=25)


# In[ ]:


plt.figure(figsize=(15,5))
sns.countplot(x='season',data=df).set_title('Pattern of Transation Trend During Different Season\'s',fontsize=25)


# In[ ]:


df['season'].unique()


# In[ ]:


plt.figure(figsize=(15,5))
sns.countplot(x='month',data=df).set_title('Transation Trend During Month',fontsize=25)


# In[ ]:


plt.figure(figsize=(15,5))
sns.countplot(x='year',data=df).set_title('Transation Trend During Year',fontsize=25)


# In[ ]:


plt.figure(figsize=(15,5))
sns.countplot(df["day"].astype(int)).set_title('pattern of transcation for each day')
plt.show()


# In[ ]:


plt.figure(figsize=(15,5))
sns.countplot(df["hour"].astype(int)).set_title('pattern of transcation for each hour')
plt.show()


# In[ ]:


plt.figure(figsize=(15,5))
sns.barplot(df["hour"].astype(int), df["Transaction"].value_counts()).set_title('transcation per hour')
plt.show()


# In[ ]:


plt.figure(figsize=(15,5))
sns.barplot(df["year"].astype(int), df["Transaction"].value_counts()).set_title("Transcation through year")
plt.show()


# In[ ]:


plt.figure(figsize=(15,5))
sns.barplot(df["season"], df["Transaction"].value_counts()).set_title('Transcation through season')
plt.show()


# In[ ]:


plt.figure(figsize=(15,5))
sns.barplot(df["dayname"], df["Transaction"].value_counts()).set_title('Transcations per day')
plt.show()


# In[ ]:


plt.figure(figsize=(15,5))
sns.countplot(df["Transaction"].value_counts())
plt.show()


# ## Modelling

# ### Import modelling libraries

# In[ ]:


from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules


# In[ ]:


# Convert the units to 1 hot encoded values
def encode_units(x):
    if x <= 0:
        return 0
    if x >= 1:
        return 1    


# In[ ]:


encoding = df.groupby(['Transaction','Item'])['Item'].count().unstack().reset_index().fillna(0).set_index('Transaction').astype(int)
encoding.head()


# In[ ]:


encoding.tail()


# In[ ]:


encoding = encoding.applymap(encode_units)


# In[ ]:


frequent_itemsets = apriori(encoding, min_support=0.01, use_colnames=True)


# In[ ]:


frequent_itemsets.head(21)


# In[ ]:


# Create the rules
rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1)
rules


# In[ ]:


output = rules.sort_values(by=['confidence'], ascending=False)
output


# We required only to see the rules where confidence is greater than or equal to 50% so:
# 
# ### Lift
# *  lift = 1 implies no relationship between A and B. 
#    (ie: A and B occur together only by chance)
# *  lift > 1 implies that there is a positive relationship between A and B.
#    (ie:  A and B occur together more often than random)
# * lift < 1 implies that there is a negative relationship between A and B.
#    (ie:  A and B occur together less often than random)
# 
#  

# In[ ]:


rules[ (rules['lift'] >= 1) &
       (rules['confidence'] >= 0.5) ]


# ## Conclusion:
# we can see that  above toast and coffee are top most commonly bought together. This makes sense since people who purchase toast would like to have coffee with it
# * which has Confidence of 70.44%
# * Lift of 1.47 tells us that coffee is 1.47 times more likely to be bought by the customers who buy toast.
# 
