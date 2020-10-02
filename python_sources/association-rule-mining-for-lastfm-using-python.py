#!/usr/bin/env python
# coding: utf-8

# # Association Rule Mining of Last FM
# 
# * Datatset : https://www.biz.uiowa.edu/faculty/jledolter/DataMining/lastfm.csv
# * We have total Transaction : **289955**
# * We have library(apyori) to calculate the association rule using Apriori.
# 
# > Requirement : `# pip install apyori`
# 
# ---
# Notebook outline
# ---
# 1. <b><a href="#1">Import Required Packages</a></b>
# 1. <b><a href="#2">Load Dataset</a></b>
# 1. <b><a href="#3">Generate Shallow Copy</a></b>
# 1. <b><a href="#4">Differentiate Data and Extract useful columns</a></b>
# 1. <b><a href="#5">Drop Duplicate Data</a></b>
# 1. <b><a href="#6">Transform Dataset into form of Transaction into list</a></b>
# 1. <b><a href="#7">Generate Rule using Apriori Algorithm</a></b>
# 1. <b><a href="#8">Display All Rules</a></b>
# 1. <b><a href="#9">Final Result with Support, Confidense and Lift</a></b>
# 
# ---
# ## Algorithm Explaination:
# ---
# 
# * We have provide `min_support`, `min_confidence`, `min_lift`, and `min length` of sample-set for find rule.
# 
# #### Measure 1: Support.
# This says how popular an itemset is, as measured by the proportion of transactions in which an itemset appears. In Table 1 below, the support of {apple} is 4 out of 8, or 50%. Itemsets can also contain multiple items. For instance, the support of {apple, beer, rice} is 2 out of 8, or 25%.
# 
# ![](https://annalyzin.files.wordpress.com/2016/04/association-rule-support-table.png?w=503&h=447)
# 
# If you discover that sales of items beyond a certain proportion tend to have a significant impact on your profits, you might consider using that proportion as your support threshold. You may then identify itemsets with support values above this threshold as significant itemsets.
# 
# #### Measure 2: Confidence. 
# This says how likely item Y is purchased when item X is purchased, expressed as {X -> Y}. This is measured by the proportion of transactions with item X, in which item Y also appears. In Table 1, the confidence of {apple -> beer} is 3 out of 4, or 75%.
# 
# ![](https://annalyzin.files.wordpress.com/2016/03/association-rule-confidence-eqn.png?w=527&h=77)
# 
# One drawback of the confidence measure is that it might misrepresent the importance of an association. This is because it only accounts for how popular apples are, but not beers. If beers are also very popular in general, there will be a higher chance that a transaction containing apples will also contain beers, thus inflating the confidence measure. To account for the base popularity of both constituent items, we use a third measure called lift.
# 
# #### Measure 3: Lift. 
# This says how likely item Y is purchased when item X is purchased, while controlling for how popular item Y is. In Table 1, the lift of {apple -> beer} is 1,which implies no association between items. A lift value greater than 1 means that item Y is likely to be bought if item X is bought, while a value less than 1 means that item Y is unlikely to be bought if item X is bought.
# ![](https://annalyzin.files.wordpress.com/2016/03/association-rule-lift-eqn.png?w=566&h=80)
# 
# ---

# ### 1.Import Required Packages <h2 id="1"> </h2>

# In[ ]:


get_ipython().system('pip install apyori')
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
from apyori import apriori
get_ipython().run_line_magic('matplotlib', 'inline')
import os
print(os.listdir("../input"))


# ### 2.Load Dataset <span id="2"> </span>

# In[ ]:


lastfm1 = pd.read_csv("https://www.biz.uiowa.edu/faculty/jledolter/DataMining/lastfm.csv")


# ### 3.Generate Shallow Copy<span id="3"></span>

# In[ ]:


lastfm = lastfm1.copy()
lastfm.shape


# ### 4.Differentiate Data and Extract useful columns <span id="4"></span>

# In[ ]:


lastfm = lastfm[['user','artist']]


# ### 5.Drop Duplicate Data <span id="5"></span>

# In[ ]:


lastfm = lastfm.drop_duplicates()
lastfm.shape


# ### 6.Transform Dataset into form of Transaction into list <span id="6"></span>

# In[ ]:


records = []
for i in lastfm['user'].unique():
    records.append(list(lastfm[lastfm['user'] == i]['artist'].values))


# In[ ]:


print(type(records))


# ### 7.Generate Rule using Apriori Algorithm <span id="7"></span>

# In[ ]:


association_rules = apriori(records, min_support=0.01, min_confidence=0.4, min_lift=3, min_length=2)
association_results = list(association_rules)


# In[ ]:


print("There are {} Relation derived.".format(len(association_results)))


# ### 8.Display All Rules<span id="8"></span>

# In[ ]:


for i in range(0, len(association_results)):
    print(association_results[i][0])


# ### 9.Final Result with Support, Confidense and Lift <span id="9"></span>

# In[ ]:


for item in association_results:
    # first index of the inner list
    # Contains base item and add item
    pair = item[0]
    items = [x for x in pair]
    print("Rule: With " + items[0] + " you can also listen " + items[1])

    # second index of the inner list
    print("Support: " + str(item[1]))

    # third index of the list located at 0th
    # of the third index of the inner list

    print("Confidence: " + str(item[2][0][2]))
    print("Lift: " + str(item[2][0][3]))
    print("=====================================")

