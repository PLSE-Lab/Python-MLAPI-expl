#!/usr/bin/env python
# coding: utf-8

# # Importing Libraries and loading datasets

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns # data visualization
import matplotlib.pyplot as plt # data visualization
get_ipython().run_line_magic('matplotlib', 'inline')

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))


# In[ ]:


# Creating the dataframes
train = pd.read_csv('../input/train.tsv', sep='\t', encoding='utf-8')
test = pd.read_csv('../input/test.tsv', sep='\t', encoding='utf-8')


# # Understanding the datasets

# In[ ]:


train.head(3)


# In[ ]:


test.head(3)


# In[ ]:


# concatenating both train and test
total = pd.concat([train,test])
total = total[["train_id","test_id","name","item_condition_id","category_name","brand_name","shipping","item_description"]] # just rearranging the columns


# In[ ]:


print("In our data, there are a total of:\n",
      len(total), "rows;\n",
      len(total.name.unique()), 'unique name;\n',
      len(total.item_condition_id.unique()), 'item_condition_id categories;\n',
      len(total.category_name.unique()), 'unique category_name;\n',
      len(total.brand_name.unique()), 'unique brand_name;\n', 
      len(total.shipping.unique()), 'shipping categories;\n',
      len(total.item_description.unique()), 'unique item_description;\n'
     )


# In[ ]:


print("In " + str(len(total)) + " rows, there are a total of:\n",
      sum(total.name.isnull()), 'None values in name (%.2f%%)\n' % (100*sum(total.name.isnull())/len(total)),
      sum(total.item_condition_id.isnull()), 'NaNs item_condition_id (%.2f%%)\n' % (100*sum(total.item_condition_id.isnull())/len(total)),
      sum(total.category_name.isnull()), 'None values in category_name (%.2f%%)\n' % (100*sum(total.category_name.isnull())/len(total)),
      sum(total.brand_name.isnull()), 'None values in brand_name (%.2f%%)\n' % (100*sum(total.brand_name.isnull())/len(total)),
      sum(total.shipping.isnull()), 'NaNs in shipping (%.2f%%)\n' % (100*sum(total.shipping.isnull())/len(total)),
      sum(total.item_description.isnull()), 'None values in item_description (%.2f%%)\n' % (100*sum(total.item_description.isnull())/len(total))
     )


# Checking the item_condition_id distribution:

# In[ ]:


plt.figure(figsize=(14,8))
sns.countplot(x="item_condition_id", data=total, palette="GnBu_d")
plt.ylabel('Frequency', fontsize=14)
plt.xlabel('item_condition_id', fontsize=14)
plt.title("Distribution of item_condition_id", fontsize=18)
plt.show()


# In[ ]:


print("\tcategory_name with most occurences:\n",
      total.groupby("category_name")["category_name"]
           .count().sort_values(ascending=False)[:10]
     )


# In[ ]:


print("\tcategory_name with most occurences:\n", total.groupby("category_name")["category_name"].count().sort_values(ascending=True)[:10])


# ### **[TODO] **
# vars graph

# # Analysing the text in item_description:

# In[ ]:


total["description_count"] = total["item_description"].astype(str).apply(lambda x: len(x.split()))
total[["item_description","description_count"]].head(3)


# Comparing description_count values with the item_condition_id classes:

# In[ ]:


total.groupby("item_condition_id")["description_count"].describe()


# In[ ]:


#commented for Kaggle limits - believe me it was a beautiful Violin Plot
'''plt.figure(figsize=(14,8))
sns.violinplot(x="item_condition_id", y="description_count", data=total, palette="GnBu_d")
sns.swarmplot(x="item_condition_id", y="description_count", data=total, color="r", alpha=0.15)
plt.ylabel("description_count", fontsize=14)
plt.xlabel("item_condition_id", fontsize=14)
plt.title("Description length distribution in the item conditions categories", fontsize=18)
plt.show()'''


# We can see through the Violin Plot (a little unusual, but I love this graph) and the above description that we have the same minimum value (1) in all the item_condition_id classes and the maximum value varies between 245 and 197.
# 
# Although this could mean that the number of words in the description rises when the item's condition come to 1, we can't conclude such thing - the mean values oscillate from 30.5 in condition_id=5 to 22.1 in condition_id=3 and then again 31.3 in condition id=1.
# 
# * So, for now, I'm leaving this study for later conclusions, and I'll proceed to a NLP analysis of the description.

# ### **[TODO] **
# NLP

# # [under construction]
