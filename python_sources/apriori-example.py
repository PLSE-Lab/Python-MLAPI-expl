#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd


# In[ ]:


df = pd.read_csv('../input/GroceryStoreDataSet.csv',names=['products'],header=None)


# In[ ]:


df


# In[ ]:


df.columns 


# In[ ]:


df.values


# In[ ]:


data = list(df["products"].apply(lambda x:x.split(',')))
data 


# In[ ]:


from mlxtend.preprocessing import TransactionEncoder


# <pre>
# mlxtend need data in below format. 
# 
#              itemname  apple banana grapes
# transaction  1            0    1     1
#              2            1    0     1  
#              3            1    0     0
#              4            0    1     0
# </pre>

# In[ ]:


te = TransactionEncoder()
te_data = te.fit(data).transform(data)
df = pd.DataFrame(te_data,columns=te.columns_)
df.head()


# In[ ]:


from mlxtend.frequent_patterns import apriori


# **Below is example of finding association rules**

# In[ ]:


df1 = apriori(df,min_support=0.01, use_colnames=True)
df1


# **Lets create a function which returns the next suggestions for customer, when he adds some items to his basket**

# In[ ]:


df1.sort_values(by="support",ascending=False)


# In[ ]:


def check_in_list(ruleitem, basketitem):  #helper checking function
#     print("basketitem=", basketitem)
#     print("ruleitem=",list(ruleitem))
    ret = all(t in list(ruleitem) for t in basketitem)
#     print(ret)
    return ret


# In[ ]:


# Uncomment below lines to see what above function does
# df2 = apriori(df,min_support=0.01, max_len=2, use_colnames=True)  #list of only 1 items
# df2 = df2[df2["itemsets"].apply(len) > 1]
# df2["check_in_list"] = df2["itemsets"].apply(check_in_list, args=(['BISCUIT', 'BREAD'],))
# df2 = df2.sort_values(by="support",ascending=False)  #sor
# df2


# In[ ]:


df1 = apriori(df,min_support=0.01, max_len=1, use_colnames=True)  #list of only 1 items
df1 = df1.sort_values(by="support",ascending=False)  #sort desc by support value
def next_item(basketitems):
    if basketitems is None:
        return df1["itemsets"][0]
    max_len_apriori=len(basketitems) + 1
    df2 = apriori(df,min_support=0.01, max_len=max_len_apriori, use_colnames=True)  #list of only 1 items
    df2 = df2[df2["itemsets"].apply(len) > max_len_apriori-1]
    df2["check_in_list"] = df2["itemsets"].apply(check_in_list, args=(basketitems,))
    df2 = df2[df2["check_in_list"] == True]
    df2 = df2.sort_values(by="support",ascending=False)  #sort desc by support value
#     print(df2)
    if(len(df2) > 0):
        return list(df2["itemsets"])[0]


# Usage examples:

# * When nothing in baset:

# In[ ]:


item=next_item(None)
print(list(item))


# * When 1 item is added to basket

# In[ ]:


item=next_item(['BISCUIT'])
print(list(item))


# * When 2 items is added to basket

# In[ ]:


item=next_item(['BISCUIT', 'BREAD'])
print(list(item))


# When 3 items is added to basket

# In[ ]:


item=next_item(['BISCUIT', 'BREAD', 'MILK'])
print(list(item))

