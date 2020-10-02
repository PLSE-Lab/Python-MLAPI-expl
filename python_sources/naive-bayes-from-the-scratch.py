#!/usr/bin/env python
# coding: utf-8

# # Naive Bayes from the scratch
# Naive Bayes algorithm is based on Bayes theory on conditional probability. Naive Bayes is one of the simplest algorithms. But it is not very easy to implement. Because to find individual probabilities, you have to separate the different classes, then for each feature in each class, you have to find the probability of each value of feature.
# 
# This notebook contines native implementation of Naive Bayes only using numpy and pandas library.

# In[16]:


import numpy as np
import pandas as pd


# In[6]:


mush = pd.read_csv("../input/mushrooms.csv")


# The data consists of missing data with '?' as value. All the missing values are from single column. Lets remove it.

# In[8]:


mush.replace('?',np.nan,inplace=True)
print(len(mush.columns),"columns, after dropping NA,",len(mush.dropna(axis=1).columns))
mush.dropna(axis=1,inplace=True)


# In[9]:


target = 'class'
features = mush.columns[mush.columns != target]
classes = mush[target].unique()


# In[10]:


test = mush.sample(frac=0.3)
mush = mush.drop(test.index)


# ### Probabilities Calculation
# 
# Here we calculate probabilities and store them in dictionary structure.
# ```
# dict: 
#   keys: class
#   values: dict: 
#     keys: feature
#     values: dict:
#       keys: categorical value
#       values: probability of value
# ```
# Thus the probability of the class can be accessed using multiple dict access.

# In[11]:


probs = {}
probcl = {}
for x in classes:
    mushcl = mush[mush[target]==x][features]
    clsp = {}
    tot = len(mushcl)
    for col in mushcl.columns:
        colp = {}
        for val,cnt in mushcl[col].value_counts().iteritems():
            pr = cnt/tot
            colp[val] = pr
        clsp[col] = colp
    probs[x] = clsp
    probcl[x] = len(mushcl)/len(mush)


# In[12]:


def probabs(x):
    #X - pandas Series with index as feature
    if not isinstance(x,pd.Series):
        raise IOError("Arg must of type Series")
    probab = {}
    for cl in classes:
        pr = probcl[cl]
        for col,val in x.iteritems():
            try:
                pr *= probs[cl][col][val]
            except KeyError:
                pr = 0
        probab[cl] = pr
    return probab


# In[13]:


def classify(x):
    probab = probabs(x)
    mx = 0
    mxcl = ''
    for cl,pr in probab.items():
        if pr > mx:
            mx = pr
            mxcl = cl
    return mxcl


# In[14]:


#Train data
b = []
for i in mush.index:
    #print(classify(mush.loc[i,features]),mush.loc[i,target])
    b.append(classify(mush.loc[i,features]) == mush.loc[i,target])
print(sum(b),"correct of",len(mush))
print("Accuracy:", sum(b)/len(mush))


# In[15]:


#Test data
b = []
for i in test.index:
    #print(classify(mush.loc[i,features]),mush.loc[i,target])
    b.append(classify(test.loc[i,features]) == test.loc[i,target])
print(sum(b),"correct of",len(test))
print("Accuracy:",sum(b)/len(test))


# In[ ]:




