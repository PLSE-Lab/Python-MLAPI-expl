#!/usr/bin/env python
# coding: utf-8

# ## Objective
# Welcome to this kernel where we will be exploring a bakery in Edinburgh.  Here we will try to answer the folowing questions- what are the bestsellers, how is the bakery doing, extracting frequent itemsets using apriori algorithm and presenting their association using NetworkX.  

# We begin by importing the necessary modules and reading the csv file. 

# In[ ]:


import numpy as np 
import pandas as pd 
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
import seaborn as sns
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules
import networkx as nx
import warnings
warnings.filterwarnings('ignore')


# This is followed by checking some dataset, its shape and statistics. As can be seen there are 4 columns and 21293 rows.

# In[ ]:


data=pd.read_csv('../input/BreadBasket_DMS.csv')
data.head()


# In[ ]:


data.shape


# In[ ]:


data.describe()


# In[ ]:


data.shape


# In[ ]:


data.info()


# While exploring this dataset, I found that although there was no evident null value, some of the items (786)  were labeled as 'NONE'.  So, I removed these items from the data.

# In[ ]:


data.loc[data['Item']=='NONE',:].count()


# In[ ]:


data=data.drop(data.loc[data['Item']=='NONE'].index)


# Next question that comes up is- how many items is this bakery selling? And the answer is that this bakery menu contains 94 items and the best seller among them is Coffee.

# In[ ]:


data['Item'].nunique()


# In[ ]:


data['Item'].value_counts().sort_values(ascending=False).head(10)


# In[ ]:


fig, ax=plt.subplots(figsize=(6,4))
data['Item'].value_counts().sort_values(ascending=False).head(10).plot(kind='bar')
plt.ylabel('Number of transactions')
plt.xlabel('Items')
ax.get_yaxis().get_major_formatter().set_scientific(False)
plt.title('Best sellers')


# Next interesting thing to find out is that when is the bakery doing the most business during the day. Here, it seems that the bakery is most busy during afternoon and morning and has little business during evening and night.

# In[ ]:


data.loc[(data['Time']<'12:00:00'),'Daytime']='Morning'
data.loc[(data['Time']>='12:00:00')&(data['Time']<'17:00:00'),'Daytime']='Afternoon'
data.loc[(data['Time']>='17:00:00')&(data['Time']<'21:00:00'),'Daytime']='Evening'
data.loc[(data['Time']>='21:00:00')&(data['Time']<'23:50:00'),'Daytime']='Night'


# In[ ]:


fig, ax=plt.subplots(figsize=(6,4))
sns.set_style('darkgrid')
data.groupby('Daytime')['Item'].count().sort_values().plot(kind='bar')
plt.ylabel('Number of transactions')
ax.get_yaxis().get_major_formatter().set_scientific(False)
plt.title('Business during the day')


# During its nearly 6 months in business, this bakery has sold over 11569 items during afternoon and only 14 items during night hours.

# In[ ]:


data.groupby('Daytime')['Item'].count().sort_values(ascending=False)


# For further analysis, I needed to extract month and day from the dataset which is done as shown below.

# In[ ]:


data['Date_Time']=pd.to_datetime(data['Date']+' '+data['Time'])
data['Day']=data['Date_Time'].dt.day_name()
data['Month']=data['Date_Time'].dt.month
data['Month_name']=data['Date_Time'].dt.month_name()
data['Year']=data['Date_Time'].dt.year
data['Year_Month']=data['Year'].apply(str)+' '+data['Month_name'].apply(str)
data.drop(['Date','Time'], axis=1, inplace=True)

data.index=data['Date_Time']
data.index.name='Date'
data.drop(['Date_Time'],axis=1,inplace=True)
data.head()


# The plot shows the performance of bakery during different months of its short existence. October and April showed less business which was due to few number of operational days in these months- 2 and 7 respectively.

# In[ ]:


data.groupby('Year_Month')['Item'].count().plot(kind='bar')
plt.ylabel('Number of transactions')
plt.title('Business during the past months')


# In[ ]:


data.loc[data['Year_Month']=='2016 October'].nunique()


# In[ ]:


data.loc[data['Year_Month']=='2017 April'].nunique()


# Next, I was interested in finding out monthly bestseller. This table below shows not only the item that has maximum buyers but one can also check how many quantities of their items of interest  were sold . As expected, coffee is the topseller in all the months.

# In[ ]:


data2=data.pivot_table(index='Month_name',columns='Item', aggfunc={'Item':'count'}).fillna(0)
data2['Max']=data2.idxmax(axis=1)
data2


# Here, I checked for the daytime bestseller. Coffee top the charts during morning, afternoon and evening, but for obvious reasons it is not the favourite during night. Vegan feast is the best seller for nights.

# In[ ]:


data3=data.pivot_table(index='Daytime',columns='Item', aggfunc={'Item':'count'}).fillna(0)
data3['Max']=data3.idxmax(axis=1)
data3


# As expected, Coffee is the best seller from Monday to Sunday.

# In[ ]:


data4=data.pivot_table(index='Day',columns='Item', aggfunc={'Item':'count'}).fillna(0)
data4['Max']=data4.idxmax(axis=1)
data4


# I was curious about the business growth of this bakery. For that I have plotted some line plots. As observed above in the barplot, November showed maximum business for the bakery, followed by February and March, with a dip shown for December and January.

# In[ ]:


data['Item'].resample('M').count().plot()
plt.ylabel('Number of transactions')
plt.title('Business during the past months')


# The next plot shows weekly performance of the bakery. A big dip in business is shown around end of December and start of first week of January

# In[ ]:


data['Item'].resample('W').count().plot()
plt.ylabel('Number of transactions')
plt.title('Weekly business during the past months')


# I zoomed in to the daily performance of the bakery and found that there have been days around December end and January beginning when the bakery sold 0 item.

# In[ ]:


data['Item'].resample('D').count().plot()
plt.ylabel('Number of transactions')
plt.title('Daily business during the past months')


# In[ ]:


data['Item'].resample('D').count().min()


# During the most profitable day the bakery could sell around 292 items and this happened in February (as seen in the daily graph above).  

# In[ ]:


data['Item'].resample('D').count().max()


# ### Apriori Algorithm and Association Rule:
# Next, I plan to perform an association rule analysis which gives an idea about how things are associated to each other. The common metrics to measure association are-
# 1. Support- It is the measure of frequency or abundance of an item in a dataset. It can be 'antecedent support', 'consequent support', and 'support'. 'antecedent support' contains proportion of transactions done for the antecedent while 'consequent support' involves  those for consequent. 'Support' is computed for both antecedent and consequent in question.
# 2. Confidence-This gives the probability of consequent in a transaction given the presence of antecedent.
# 3. Lift- Given that antecedents and consequents are independent, how often do they come together/bought together.
# 4. Leverage- It is the difference between frequency of antecedent and consequent together in transactions to frequency of both in independent transactions.
# 5.Conviction- A higher conviction score means that consequent is highly dependent on antecedent.
# 
# Apriori algorithm is used to extract frequent itemsets that are further used for association rule analysis. In this algorithm, user defines a minimum support that is the minimum threshold that decides if an itemset is considered as 'frequent'. 

# To begin with association rule analysis, I made a dataset that contains lists of items that are bought together.

# In[ ]:


lst=[]
for item in data['Transaction'].unique():
    lst2=list(set(data[data['Transaction']==item]['Item']))
    if len(lst2)>0:
        lst.append(lst2)
print(lst[0:3])
print(len(lst))


# For Apriori algorithm, this dataset needs to be one-hot encoded. This is done using TransactionEncoder as shown here, followed by apriori algorithm to get the frequent itemsets. Then association rules function is used which can take any metric. Here I have used 'lift' and specified minimum threshold as 1.    

# In[ ]:


te=TransactionEncoder()
te_data=te.fit(lst).transform(lst)
data_x=pd.DataFrame(te_data,columns=te.columns_)
print(data_x.head())

frequent_items= apriori(data_x, use_colnames=True, min_support=0.03)
print(frequent_items.head())

rules = association_rules(frequent_items, metric="lift", min_threshold=1)
rules.antecedents = rules.antecedents.apply(lambda x: next(iter(x)))
rules.consequents = rules.consequents.apply(lambda x: next(iter(x)))
rules


# Next I used NetworkX (a Python package for the creation and study of complex networks) to build a network graph to check association between antecedents and consequents obtained after association rule. As can be seen from it, coffee is a very popular item that goes well with pastry, cake, medialuna and sandwich sold in this bakery. So, if a person is buying any of the last four items the chances of buying a coffee is high (also the reverse is true i.e. if a person is buying a coffee then likelihood of her/him buying any of the 4 items is high).
# 
# 

# In[ ]:


fig, ax=plt.subplots(figsize=(10,4))
GA=nx.from_pandas_edgelist(rules,source='antecedents',target='consequents')
nx.draw(GA,with_labels=True)
plt.show()


# ### Conclusion
# Coffee is the bestseller of this bakery and it shows association with 4 items-  pastry, cake, medialuna and sandwich.
# There are a couple of strategies that the bakery can adopt (if it isn't using them yet) to increase its sales considering the association we have seen between coffee and its 4 partners.
# 1. Promotional discount in either one of the 4 partners can entice customers to buy coffee (or the other way round, will also work).
# 2. Placing these 4 items close to coffee ordering counter can be a good strategy to attract customers in buying these (which we see quite often to happen in many bakeries and coffee shops).
# 3. How about some recipes like a coffee cake or coffee pastry? Will that entice coffe and cake/pastry lovers?? 
# 
