#!/usr/bin/env python
# coding: utf-8

# # Association Analysis on Kaggle Random Shopping Cart

# Click this [link](https://www.kaggle.com/fanatiks/shopping-cart) for dataset.

# ## Setup

# In[ ]:


import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

# mlxtend library
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules


# ## Data Preprocessing

# Since the provided CSV file has variable length transaction and no header, first of all we need to find the maximum number items in a transaction. So read transaction in a single columns of dataframe and use commas`(,)` to count number of items in each transaction.

# In[ ]:


raw_data = pd.read_csv('../input/dataset.csv', header=None, sep='^')


# In[ ]:


raw_data.head(3)


# In[ ]:


max_col_count = max([row.count(',') for row in raw_data[0]])


# Re-read the dataset, this time also with no header but with NaN for blank items. All this tediousness for making a Pandas dataframe.

# In[ ]:


raw_data = pd.read_csv('../input/dataset.csv', header=None, names=list(range(max_col_count)))


# Since items in `csv` file are separated by `,_` strip each item

# In[ ]:


for col in raw_data.columns:
    raw_data[col] = raw_data[col].str.strip()


# In[ ]:


raw_data.head(3)


# Look at the first column, It is a complete mess, who made this `CSV`??
# 
# Separate date and text from first column.

# In[ ]:


def strip_date_text(s):
    match = re.search(r'(\d{1,2}/\d{1,2}/\d{4})([a-z /-]+)', s)
    date = datetime.strptime(match.groups()[0], '%d/%m/%Y').date()
    return date, match.groups()[1]


# In[ ]:


transac_data = pd.concat([pd.DataFrame(raw_data[0].apply(strip_date_text).tolist()), raw_data.iloc[:, 1:]], axis=1, ignore_index=True)


# In[ ]:


transac_data.head(3)


# Encodes database transaction data in form of a Python list of lists into a NumPy array.
# 
# User guide available [here](http://rasbt.github.io/mlxtend/user_guide/preprocessing/TransactionEncoder/)
# 
# `dataset = [['Apple', 'Beer', 'Rice', 'Chicken'],
#            ['Apple', 'Beer', 'Rice'],
#            ['Apple', 'Beer'],
#            ['Apple', 'Bananas'],
#            ['Milk', 'Beer', 'Rice', 'Chicken'],
#            ['Milk', 'Beer', 'Rice'],
#            ['Milk', 'Beer'],
#            ['Apple', 'Bananas']]
#                 ||
#                 || Transactional Encoder form mlxtend
#                 ||
#                 \/
# array([[ True, False,  True,  True, False,  True],
#        [ True, False,  True, False, False,  True],
#        [ True, False,  True, False, False, False],
#        [ True,  True, False, False, False, False],
#        [False, False,  True,  True,  True,  True],
#        [False, False,  True, False,  True,  True],
#        [False, False,  True, False,  True, False],
#        [ True,  True, False, False, False, False]])`
#        
#        

# In[ ]:


dataset = [transac_data.loc[i].dropna()[1:-1].values for i in range(len(transac_data))]
te = TransactionEncoder()
te_ary = te.fit(dataset).transform(dataset)
df = pd.DataFrame(te_ary, columns=te.columns_)


# In[ ]:


df.head()


# ## Frequent Itemset Detection

# Apriori is a popular algorithm for extracting frequent itemsets with applications in association rule learning. The apriori algorithm has been designed to operate on databases containing transactions, such as purchases by customers of a store. An itemset is considered as "frequent" if it meets a user-specified support threshold.
# 
# <img src="https://lh4.googleusercontent.com/qrCw_Zjn0GoAUp3e36A4_IeaPWZAviTDEw_DHbvGIWY4TeK3CByhj7mpxxzP4HHhxicjERISm7UYyA=w1921-h952-rw">
# 
# <b>Pseudo-code</b> for <b>`Apriori Algorithm`</b>
# <img src="https://lh5.googleusercontent.com/Jb57K58UfPkp8_G_y0ALVL89CmT3D5pK7KeiZ-K-U5r3eyN839dUwoQhzjB08O-JNEWyPyKgEDwgmvVbtu0_=w1921-h952-rw">

# In[ ]:


frequent_itemsets = apriori(df, min_support=0.3, use_colnames=True)
frequent_itemsets['length'] = frequent_itemsets['itemsets'].apply(lambda x: len(x))
frequent_itemsets


# ## Rule Generation

# ### Confidence Based Prunning
# 
# <b>Theorem</b>
#  If a rule `X -->  Y - X` doesnot satisfy the confidence threshold, then any rule `X --> Y - X` , where `X'`, is a subset of `X`, must not satisfy the confidence threshold as well
#  
#  <img src="https://lh5.googleusercontent.com/8NUsgNjmz63ir2dVzmMDPqG2kmwdB-2ExB5iFEg0zmrF0JEi6NhoRUKz5TOg37cjJh4VVFUplgnBE3Eg0KON=w1921-h952-rw">

# In[ ]:


rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.7)
rules


# ## Analysis of Algorithm

# #### min_support vs number of frequent itemsets
#  Lowering the support threshold often results in more itemsets being declared as frequent. This has an adverse effect on the computational complexity of the algorithm because more candidate itemsets must be generated and counted, as shown in Figure 6.13. The maximum size of frequent itemsets also tends to increase with lower support thresholds. As the maximum size of the frequent itemsets increases, the algorithm will need to make more passes over the data set

# In[ ]:


min_supports = np.linspace(0.1, 1, 50, endpoint=True)
no_of_frequent_items = []

for min_support in min_supports:
    no_of_frequent_items.append(apriori(df, min_support=min_support).shape[0])
    
plt.plot(min_supports, no_of_frequent_items)
plt.xlabel('min_support')
plt.ylabel('number of frequent itemsets')
plt.title('min_support vs number of frequent itemsets')


# In[ ]:


min_supports = np.linspace(0.1, 1, 10, endpoint=True)
no_of_frequent_items = {}

plt.figure()
ax = plt.gca()

for min_support in min_supports:
    frequent_itemsets = apriori(df, min_support=min_support, use_colnames=True)
    frequent_itemsets['length'] = frequent_itemsets['itemsets'].apply(lambda x: len(x))
    itemset_sizewise_count = frequent_itemsets.groupby(['length']).size().tolist()
    print(itemset_sizewise_count)
    ax.plot(list(range(1, len(itemset_sizewise_count) + 1)), itemset_sizewise_count,
            label='min_support=' + str(min_support))
    
plt.legend()
plt.xlabel('size of itemsets')
plt.ylabel('number of frequent itemsets')

