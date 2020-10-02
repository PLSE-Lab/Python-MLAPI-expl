#!/usr/bin/env python
# coding: utf-8

# ### Introdution
# Hi guys, here I am going to give you a head start on Market basket analysis and Association rule mining using Apriori algorithm. 
# 
# Market Basket Analysis is one of the most popular methodologies to increase the sale of products in store by finding the associations between them. The idea is to bring the set of products together that have some kind of inter-dependency in terms of their use. Doing so can surely boost the sale because placing them together will remind or encourage customers about their need for the associated product.
# 
# Topic is interesting, and implementation will become easier of you unserstand what Apriori algorithm is and how Association rules works, refer to below post to understand the concept with example:
# 
# [Market Basket Analysis (Part 1|2): Apriori Algorithm and Association Rule Mining](http://https://medium.com/@amardeepchauhan/market-basket-analysis-part-1-2-apriori-algorithm-and-association-rule-mining-693e4fd2d69c)
# 
# 

# In this notebook we'll focus on one practicle example, note that if you understand the concept than you can implement Support, Lift and Confidence functions on your own, but instead of reinventing the wheel, we'll simply use mlextend library which contaisn all the required modules.

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
pd.set_option('display.max_rows', None)

from matplotlib import pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

import seaborn as sns
sns.set_style('whitegrid')

from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


market_basket_df = pd.read_csv('../input/market-basket-optimization/Market_Basket_Optimisation.csv', header=None)
# Any results you write to the current directory are saved as output.


# In[ ]:


market_basket_df.info()


# In[ ]:


market_basket_df.head()


# As you can see, dataset contains no column or index label, so let's go ahead and assume each row as a transaction or basket.

# We need to bring it in proper structure, only then we can do some further analysis. We can create columns for each item and keep its row value as True/False based on its occurance in that transaction.

# ### creating list of itemsets in basket in different transactions

# In[ ]:


basket_items = []
for index, row in market_basket_df.iterrows():
    cleansed_items = [item for item in row if str(item)!='nan']
    #print(f'basket size: {len(cleansed_items)}, basket:\n{cleansed_items}')
    basket_items.append(cleansed_items)
    
basket_items[:3]


# ### Creating transaction DataFrame

# We'll use TranscationEncoder imported from mlextend and pass basket items list that we created. It will basically one hot encode transactions column values based on their occurance as we talked about:

# In[ ]:


tran_encod = TransactionEncoder()
tran_encod_list = tran_encod.fit(basket_items).transform(basket_items)
transaction_df = pd.DataFrame(tran_encod_list, columns=tran_encod.columns_)
transaction_df.head()


# Creating DataFrame for item frequency.

# In[ ]:


item_count = {}
for col in transaction_df.columns:
    item_count[col] = transaction_df[col].sum()

item_freq_df = pd.DataFrame(data=list(item_count.values()), index=list(item_count.keys()), columns=['frequency']).sort_values(by='frequency', ascending=False)
item_freq_df.shape, item_freq_df.head(10)


# ![](http://)ok, so we have 120 unique items, let's check their frequency all at once in bar plot, insight can be helpful in further decison making.

# In[ ]:


plt.figure(figsize=(12,26))
sns.barplot(y=item_freq_df.index, x=item_freq_df.frequency)
plt.xticks(rotation=90)


# Couple of interesting observations;)
# - People are actually becoming health concious, they prefer green tea over tea. Ohhh hold on.. they are actually consuming much more spaghetti, french fries, chocolate, burgers, cake and cookies as compared to oatmeal and veggies.. coincidence!!! Nah, this is how it happens actually, We eat lots of junk and try to balance it out with green tea, we are SMART :D
# - World before Covid19, Napkins are at 3rd last position :D

# ok, lets be serious and come to the point. 
# We have total 7501 observations and only 7 items with frequency greater than 750. It means only 7 items has support greater than 10%. Let's cross validate:

# In[ ]:


apriori(transaction_df, min_support=0.1, use_colnames=True)


# * So now what.. well, now we need to decide some realistic min_support only then we'll be able to find some useful association rules.

# In[ ]:


print(f'freq>200: {item_freq_df[item_freq_df.frequency>200].shape[0]} items')
print(f'freq>100: {item_freq_df[item_freq_df.frequency>100].shape[0]} items')
print(f'freq>50: {item_freq_df[item_freq_df.frequency>50].shape[0]} items')


# In[ ]:


pd.set_option('display.max_rows', 15)
freq_itemset_support = apriori(transaction_df, min_support=0.03, use_colnames=True)
freq_itemset_support


# In[ ]:


overal_association_rules = association_rules(freq_itemset_support, metric="confidence", min_threshold=0.2)
overal_association_rules


# Well, if we take 20% as confidece score, association rules are mostly dominated by mineral watter association. Mineral water is already associated with most of the product, so we better exclude it from transactions to find out other meaningful association rules.

# In[ ]:


trans_wo_water_df = transaction_df.drop(columns=['mineral water'])

freq_itemset_wo_water_supp = apriori(trans_wo_water_df, min_support=0.02, use_colnames=True)
freq_itemset_wo_water_supp


# In[ ]:


wo_water_assoc_rules = association_rules(freq_itemset_wo_water_supp, metric="confidence", min_threshold=0.2)
wo_water_assoc_rules


# * hmm, let's order it by _confidence_ and then _lift_

# In[ ]:


wo_water_assoc_rules.sort_values('confidence', ascending=False)


# In[ ]:


wo_water_assoc_rules.sort_values('lift', ascending=False)


# Ok, now we see few meaningfull associations, like:
# - ground beef -> spaghetti
# - herb & pepper -> ground beef
# - red wine -> spaghetti
# - tomatoes -> frozen vegetables
# - frozen vegetables -> spaghetti
# - (chocolate, spaghetti) -> milk
# - Burgers -> Eggs
# - Burgers -> French fries
# - pancakes -> french fries	
# - Milk -> Chocolate
# - Milk -> Eggs
# - Olive oil -> Spaghetti
# 
# There are few weird associations as well (atleast as per Indian taste):
# - ground beef -> milk
# - champagne -> chocolate
# - olive oil -> chocolate
# - shrimp -> milk
# - green tea -> french fries

# If you've noticed these associations rules are still dominated by few of the most frequenct products, let's take them out and check association again: 
# - eggs
# - spaghetti
# - chocolate
# - milk
# - ground beef
# - frozen vegetables

# You can narrow it down further and apply various filters either on the basis of confidence or lift and generate many other association rules. Just note that quality of association rule is dependent on quality of data, or better I say authenticity of data. For actual problem and real dataset, you can't simply decide min_support and min_confidence on whim, it requires some critical thinking. However, you got to start from somewhere, and that's what I tried to give you. Later we'll explore _Market basket differential analysis_.
# 
# to be continued..

# In[ ]:




