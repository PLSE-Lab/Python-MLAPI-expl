#!/usr/bin/env python
# coding: utf-8

# ### The dataset is a very typical transaction bill dataset.
# ### How many insights could you find from it?

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('white')
sns.set_palette('Paired')


# In[ ]:


df = pd.read_csv('../input/BreadBasket_DMS.csv')
df.head(2)


# The original variable in dataset is quite few. So the first thing is to add more variables. 
# 
# * I split Date into Year, Month, Day of Week ,Hour of Day
# * Also, I add Morning, Afternoon and evening, the 3 new variables.
# 
# ## How many insights you can find depends on how many good questions you can think. 
# Let's start from very simple ones:
# 1. The popular goods
# 2. How many transactions happens in each year/month/day/hour?
# 3. How many transactions happens in different daytime?(Morning/Afternoon/Evening)
# 4. What is the best time to sell [any good], let s say 'coffee'
# 5. How the goods sale related to each other?
# 6. Can I predict the sales?
# 7. .and so on
# 
# 

# In[ ]:


df.Date = pd.to_datetime(df.Date)
df['Year'] = df.Date.dt.year.astype(int)
df['Month'] = df.Date.dt.month.astype(int)
df['Dow'] = df.Date.dt.dayofweek.astype(int)
df['Hour'] = df.Time.apply(lambda x: x.split(':')[0]).astype(int)


# In[ ]:


df['Morning'] = 0
df['Morning'].loc[df.Hour < 12] =1
df['Afternoon'] = 0
df['Afternoon'].loc[(df.Hour >= 12)&(df.Hour < 18)] =1
df['Evening'] = 0
df['Evening'].loc[df.Hour >= 18] =1
df.head(3)
       


# In[ ]:


df.Item.value_counts()[:10]


# In[ ]:


df= df.drop(df.loc[df['Item'] =='NONE'].index)
#df.Item.value_counts()[:10].plot(kind ='bar',figsize =(8,8),title = 'The top 10 popular goods')


# In[ ]:


values = df.Item.value_counts()[:10]
labels = df.Item.value_counts().index[:10]
plt.figure(figsize = (8,8))
plt.pie(values, autopct='%1.1f%%', labels = labels,
        startangle=90)
plt.title('Top 10 bestselling goods in Piechart')


# In[ ]:


df.Transaction.value_counts().plot(kind ='hist',figsize =(8,8),title = 'Distribution of Transaction')


# ### In the bakery business, generally people will buy 1-2 goods.  Some people will buy 2-4 goods
# ### How could you  attract customers and sell them the best margin or the best selling goods in 1-3 choices?
# ### Could we know more about their behaviors?

# In[ ]:


f,axes= plt.subplots(4,2,figsize =(14,28))
sns.countplot(df.Year,ax =axes[0][0])
sns.barplot(df.Year,df.Transaction,ax =axes[0][1])

sns.countplot(df.Month,ax =axes[1][0])
sns.barplot(df.Month,df.Transaction,ax =axes[1][1])

sns.countplot(df.Dow,ax =axes[2][0])
sns.barplot(df.Dow,df.Transaction,ax =axes[2][1])

sns.countplot(df.Hour,ax =axes[3][0])
sns.barplot(df.Hour,df.Transaction,ax =axes[3][1])


# In[ ]:


cols =['Morning','Afternoon','Evening']
for c in cols:
   print('Number of Transaction in',c, df[c].sum() ,'.The percentage is', df[c].sum()/len(df))


# ### Morning and evening take each half part of sales, especailly in afternoon time.

# In[ ]:


Coffee = df[df['Item'] == 'Coffee']
Coffee_hour = Coffee.groupby('Hour').size().reset_index(name='Counts')

plt.figure(figsize=(12,8))
ax = sns.barplot(x='Hour', y = 'Counts', data = Coffee_hour)
ax.set(xlabel='hour of Day', ylabel='Coffee Sold')
ax.set_title('Distribution of Coffees Sold by Hour of Day')


# ### We can see from 10-11 o'clock the coffee sale reachs peak.
# ### You can check any goods' time-sales trend in this way.

# ## Let's  find Coffee's best mate!

# In[ ]:


Coffee_tran =Coffee.Transaction.tolist()
df_subset = df[df.Transaction.isin(Coffee_tran)]
df_subset.head()


# In[ ]:


Top10_in_df = df.Item.value_counts()[:10]
Top10_in_subset = df_subset.Item.value_counts()[:10]

Top10_in_df = Top10_in_df.drop(labels=['Coffee'])
Top10_in_subset = Top10_in_subset.drop(labels=['Coffee'])

labels = Top10_in_df.index.values.tolist()
values_Top10_in_df =Top10_in_df.tolist()
values_Top10_in_subset = Top10_in_subset.tolist()

values_without_coffee = [values[i]-v for i,v in enumerate(values_Top10_in_df)]
coffee_mate = pd.DataFrame({'Name':labels,'with_':values_Top10_in_df, 'without_':values_Top10_in_subset})
coffee_mate


# In[ ]:


coffee_mate['with on without Percentage'] = coffee_mate.with_ / coffee_mate.without_ 
sns.mpl.rc('figure', figsize=(9,6))
sns.barplot('Name','with on without Percentage',data = coffee_mate)
plt.axhline(y=1, color='r', linestyle='--')
plt.axhline(y =2,color ='r',linestyle ='--')
plt.title('The percentage of item sales with coffee on items without coffee')


# ### All of the mates are over 1 fold.
# ### But the best match are Bread,Tea and Cake. Do you know how to push the cross-selling right now?

# ##  Try some prediction

# In[ ]:


Coffee_count = Coffee.groupby('Date').size().reset_index(name ='Count')
Coffee_count.describe()


# ### The median and mean of daily coffee sale are quite close, 33- 34 cups per day. 
# ### If you sell 34 cups coffee daily, that's a average result, but it you sell just a little more, let's say, 42 cups , then you can beat 75% other sales days.

# In[ ]:


sns.mpl.rc('figure', figsize=(16,6))
for xc in np.arange(0, len(Coffee_count), step=7):
    plt.axvline(x=xc, color='k', linestyle='--')
    
Coffee_count.Count.plot(color ='r')
plt.xticks(np.arange(0, len(Coffee_count), step=7))
plt.ylabel('Coffee Sale')
plt.title('Daily Coffee sales')


# In[ ]:


ls=[]
for item in df['Transaction'].unique():
    ls2=list(set(df[df['Transaction']==item]['Item']))
    if len(ls2)>0:
        ls.append(ls2)
print(ls[0:3],len(ls))


# In[ ]:


from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules
import networkx as nx


# In[ ]:


te=TransactionEncoder()
te_data=te.fit(ls).transform(ls)
data_x=pd.DataFrame(te_data,columns=te.columns_)
print(data_x.head())

frequent_items= apriori(data_x, use_colnames=True, min_support=0.02)
print(frequent_items.head())

rules = association_rules(frequent_items, metric="lift", min_threshold=1)
rules.antecedents = rules.antecedents.apply(lambda x: next(iter(x)))
rules.consequents = rules.consequents.apply(lambda x: next(iter(x)))
rules


# In[ ]:


fig, ax=plt.subplots(figsize=(10,6))
GA=nx.from_pandas_edgelist(rules,source='antecedents',target='consequents')
nx.draw(GA,with_labels=True)

