#!/usr/bin/env python
# coding: utf-8

# This was a short test to see if I could get useful data from the Cast/Crew columns of this dataset. Unfortunately I discovered there is not enough overlap between the train and test data of these sets.

# In[ ]:


#imports 
import math
from random import shuffle
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn import svm
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# Load Data

# In[ ]:


# load data
train = pd.read_csv('../input/train.csv')
train[['id','cast','popularity','budget','revenue']].head()


# Show a dist plot of revenue

# In[ ]:


# budget / revenue hist
sns.distplot(train['revenue'])


# Here is the function to evaluate and filter out unique names from the cast/crew data objects that are serialized in these columns. I filter for only Writer and Director jobs
# to reduce the output size of the set.

# In[ ]:


# getNameSets : function to transform the serialized python dictionaries in cast/crew
# into a table of unique names
jobs = ['Writer','Director']
def getNameSets(dataset):
    idnameset = {}
    namecountset = {}
    for tup in dataset.itertuples():
        # eval python code in cast/crew columns into filtered name lists
        shortlist = []
        if isinstance(tup.cast, "".__class__):
            evaled_cast = eval(tup.cast)
            shortlist = shortlist + evaled_cast[0:5] # first 5 (presorted by 'order')
        if isinstance(tup.crew, "".__class__):
            evaled_crew = eval(tup.crew)
            crewlist = [x for x in evaled_crew if x['job'] in jobs] # match jobs
            shortlist = shortlist + crewlist
        for obj in shortlist:
            if not (tup.id in idnameset):
                idnameset[tup.id] = {}
            idnameset[tup.id][obj['name']] = True 
            if not (obj['name'] in namecountset):
                namecountset[obj['name']] = 0
            namecountset[obj['name']] += 1
    return (idnameset, namecountset)
(idnameset, namecountset) = getNameSets(train)
print(f'unique names count {len(namecountset.keys())}, records count {len(idnameset.keys())}')


# Check out the most popular names

# In[ ]:


# names that appear in more than one record
repeat_names = [{'name': name, 'records': count} for name,count in namecountset.items() if count > 1]
# print(len(repeat_names))
distdf = pd.DataFrame(repeat_names)['records']
sns.distplot(distdf)


# In[ ]:


# top 10
top_10_names = sorted(repeat_names, key=lambda x: -x['records'])[0:10]
sns.barplot(y="name",x="records", data=pd.DataFrame(top_10_names))


# Let's compare what names we found to the test data

# In[ ]:


# load test data
test = pd.read_csv('../input/test.csv')
len(test)


# In[ ]:


# build names and y dfs for test data, only using names used from training set
(test_idnameset) = getNameSets(test)
test_list = []
relevant_ids = {}
def defaultNameObj():
    obj = {}
    for name in namecountset.keys():
        obj[name] = 0.0
    return obj
# make train rows from known names
for tup in test.itertuples():
    robj = defaultNameObj()
    if tup.id in test_idnameset:
        for name in test_idnameset[tup.id].keys():
            if name in namecountset:
                relevant_ids[tup.id] = True
                robj[name] = 1.0
    test_list.append(robj)
len(relevant_ids.keys())


# Unfortunately, there are no overlapping names from our training set in our test set. So it doesn't look like we can continue with this strategy.

# In[ ]:




