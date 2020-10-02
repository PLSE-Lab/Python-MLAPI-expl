#!/usr/bin/env python
# coding: utf-8

# # EDA Column Name
# 
# The column is anonymized with this cryptic names. Let's check it out! 

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import operator
import os
print(os.listdir("../input"))


# In[ ]:


train = pd.read_csv("../input/train.csv")
test = pd.read_csv("../input/test.csv")


# ## Check Train

# In[ ]:


train.head()


# In[ ]:


count_dash = {}
count_name = {}


# In[ ]:


for col in train.columns:
    if col not in ['id', 'target']:
        names = col.split('-')
        for name in names:
            if not name in count_name:
                count_name[name] = 1
            else:
                count_name[name] += 1

        if not len(names) in count_dash:
            count_dash[len(names)] = 1
        else:
            count_dash[len(names)] += 1


# All column in train consist of 4 item separated by dashes

# In[ ]:


count_dash


# The name that appears most often is 'important'

# In[ ]:


sorted_count_name = sorted(count_name.items(), key=operator.itemgetter(1), reverse=True)
sorted_count_name[0:10]


# In[ ]:


# Saving to file so you can inspect the whole list
with open('count_name_train.txt', 'w') as f:
    for item in sorted_count_name:
        f.write("%s\n" % str(item))


# Overall there is 385 unique name in train data column

# In[ ]:


len(sorted_count_name)


# In[ ]:


count_name_train = count_name


# ## Check Test

# In[ ]:


test.head()


# In[ ]:


count_dash = {}
count_name = {}


# In[ ]:


for col in test.columns:
    if col not in ['id']:
        names = col.split('-')
        for name in names:
            if not name in count_name:
                count_name[name] = 1
            else:
                count_name[name] += 1
        
        if not len(names) in count_dash:
            count_dash[len(names)] = 1
        else:
            count_dash[len(names)] += 1


# All column in test also consist of 4 item separated by dashes

# In[ ]:


count_dash


# The name that appears most often is also 'important'

# In[ ]:


sorted_count_name = sorted(count_name.items(), key=operator.itemgetter(1), reverse=True)
sorted_count_name[0:10]


# In[ ]:


# Saving to file so you can inspect the whole list
with open('count_name_test.txt', 'w') as f:
    for item in sorted_count_name:
        f.write("%s\n" % str(item))


# Overall there is 385 unique name in test data column

# In[ ]:


len(sorted_count_name)


# Every name is present in both train and test column

# In[ ]:


set(count_name.keys()) - set(count_name_train.keys())


# In[ ]:


set(count_name_train.keys()) - set(count_name.keys())


# That's all for now. I'll update this later when there is new findings.

# ## Uniqueness Between Columns
# Do any words from one column show up in another column? (For example does a word from column 1 show up in column 2, 3, or 4?) as asked by [@cdeotte](https://www.kaggle.com/cdeotte) [here](https://www.kaggle.com/c/instant-gratification/discussion/92604#533156)

# In[ ]:


# We can use only train column because it's exactly the same between test & train
print(set(train.columns) - set(test.columns))
print(set(test.columns) - set(train.columns))


# Let's store name for each column

# In[ ]:


column_1 = {}
column_2 = {}
column_3 = {}
column_4 = {}


# In[ ]:


for col in train.columns:
    if col not in ['id', 'target']:
        names = col.split('-')
        if not names[0] in column_1:
            column_1[names[0]] = 1
        else:
            column_1[names[0]] += 1

        if not names[1] in column_2:
            column_2[names[1]] = 1
        else:
            column_2[names[1]] += 1

        if not names[2] in column_3:
            column_3[names[2]] = 1
        else:
            column_3[names[2]] += 1

        if not names[3] in column_4:
            column_4[names[3]] = 1
        else:
            column_4[names[3]] += 1
        


# There is a duplicate name between column 2 & 3!

# In[ ]:


col_list = [column_1, column_2, column_3, column_4]

for i in range(0,3):
    for j in range(i+1,4):
        print("Duplicate name between column {} and {}".format(i+1, j+1))
        print(set(col_list[i].keys()) & set(col_list[j].keys()))


# ## Column with magic
# As shown in other kernel, some column have a high impact and one of the names in that column is 'magic'. Let's find more magic.

# In[ ]:


print([x for x in train.columns if 'magic' in x.split('-')])


# In[ ]:


train['stealthy-chocolate-urchin-kernel']


# In[ ]:


train['bluesy-chocolate-kudu-fepid']


# Unfortunately there only one magic :(  
# 

# ## Ideas
# Here some crazy idea that I've tried out:
# * Using only column with 'important' in the names (bad result)  
# * Removing column with 'noise' in the names (bad result)

# That's all for now. I'll add more when I find new ideas.
