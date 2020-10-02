#!/usr/bin/env python
# coding: utf-8

# ### Mining the Bible 
# 
# Author: liu431

# 1. Randomly select a verse containing keywords
# 
# 2. Visualize number of verses in each book
# 
# 3. Counting words and splitting text
# 
# 4. Compare a verse across difference versions
# 
# 5. Text Similarity Metrics: Jaccard Similarity
# 
# ...More work in progress(NLP, CNN)...

# In[ ]:


import numpy as np
import pandas as pd 
import numpy as np 
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns

import re

get_ipython().run_line_magic('matplotlib', 'inline')
pd.options.display.max_colwidth = 1000 #to print complet verses


# In[ ]:


index=pd.read_csv('../input/bible_version_key.csv')
#Drop the columns where at least one element is missing.
index=index.dropna(axis='columns')
index


# In[ ]:


#American Standard-ASV1901
asv = pd.read_csv('../input/t_asv.csv')

#Bible in Basic English
bbe = pd.read_csv('../input/t_bbe.csv')

#Darby English Bible
dby = pd.read_csv('../input/t_dby.csv',encoding='latin-1')

#King James Version
kjv = pd.read_csv('../input/t_kjv.csv')

#Webster's Bible
wbt = pd.read_csv('../input/t_wbt.csv')

#World English Bible
web = pd.read_csv('../input/t_web.csv')

#Young's Literal Translation
ylt = pd.read_csv('../input/t_ylt.csv')


# #### Randomly select a verse containing keywords

# In[ ]:


#Find verses containing "LOVE". 
love=asv[asv['t'].str.contains('love',case=False)]
sel=np.random.randint(1,love.shape[0])
print("Verse Number:",love['b'].iloc[sel],love['c'].iloc[sel])
print(love['t'].iloc[sel])


# In[ ]:


#Find verses containing "christ"
chri=asv[asv['t'].str.contains('christ',case=False)]
sel=np.random.randint(1,chri.shape[0])
print("Verse Number:",chri['b'].iloc[sel],chri['c'].iloc[sel])
print(chri['t'].iloc[sel])


# #### Visualize number of verses in each book

# In[ ]:


ct=asv.groupby(['b'])['t'].count()
plt.bar(range(1,67),ct)


# #### Counting Words

# In[ ]:


counts = dict()
for text in asv['t']:
    tokens=text.lower().split()
    tokens=[re.sub(r'[^\w\s]','',i) for i in tokens]
    for i in tokens: 
        if i in counts:
            counts[i]+=1
        else:
            counts[i]=1
sorted_counts = sorted(counts.items(), key=lambda pair: pair[1], reverse=True)
print("10 most common words:\nWord\tCount")
for word, count in sorted_counts[:10]:
    print("{}\t{}".format(word, count))

print("\n10 least common words:\nWord\tCount")
for word, count in sorted_counts[-10:]:
    print("{}\t{}".format(word, count))


# #### Splitting text
# 
# Regular Expression package `re` to split the text into sentences, and each sentence into words (tokens)
# 
# `str.strip`: Remove leading and trailing spaces from each sentence
# `re.split(r"<your regexp>", text)`:regular expression that matches sentence delimiters
# 
# `r`: preceding the regexp string - this denotes a raw string and tells Python not to interpret the characters in any special way (e.g. escape sequences like '\n' do not get converted to newlines, etc.).
#  
# link: https://docs.python.org/3.5/library/re.html

# In[ ]:


for text in asv['t']:
    sentence=list(map(str.strip, re.split(r"[.?](?!$.)", text)))[:-1]
    for sent in sentence:
        list(map(str.strip, 
                       re.split("(?:(?:[^a-zA-Z]+')|(?:'[^a-zA-Z]+))|(?:[^a-zA-Z']+)",sent)))


# #### Compare a verse across difference versions

# Corinthians 13:4

# In[ ]:


#Book index for Corinthians, Chapter and verse number
b,c,vn=46,13,4

diff=pd.DataFrame(index['version'])
ver=[asv,bbe,dby,kjv,wbt,web,ylt]
for i,v in enumerate(ver):
    diff.loc[[i],'verse'] =v[(v['b']==b) & (v['c']==c) &(v['v']==vn)]['t'].values
    
diff


# ### Text Similarity Metrics 

# #### Jaccard Similarity
# Metric=size of intersection of the set divided by total size of set

# In[ ]:


def get_jaccard_sim(str1, str2): 
    a = set(str1.split()) 
    b = set(str2.split())
    c = a.intersection(b)
    return float(len(c)) / (len(a) + len(b) - len(c))


# In[ ]:


#Example: get certain verse without index 
asv.loc[[0],'t'].to_string(index=False)


# In[ ]:


#Compare two verses
a=diff.loc[[0],'verse'].to_string(index=False)
b=diff.loc[[4],'verse'].to_string(index=False)
get_jaccard_sim(a,b)


# In[ ]:


#Metric Matrix
jac=pd.DataFrame(index=range(7))
for it in range(7):
    jac[it]=[get_jaccard_sim(diff.loc[[it],'verse'].to_string(index=False),
                          diff.loc[[i],'verse'].to_string(index=False)) for i in range(7)]

sns.heatmap(jac, annot=True)


# In[ ]:


#Compare two books!
def com_book(b1,b2):
    if b1.shape[0]==b2.shape[0]:
        sim=[]
        for i in range(b1.shape[0]):
            a=b1.loc[[i],'t'].to_string(index=False)
            b=b2.loc[[i],'t'].to_string(index=False)    
            sim.append(get_jaccard_sim(a,b))
        return np.mean(sim)
    else:
        #print("Lengths differ. Something is wrong in the dataset :(")
        return np.nan


# In[ ]:


com_book(asv,bbe)


# In[ ]:


com_book(asv,dby)


# In[ ]:


#DataFrame Setup
ver=["asv","bbe","dby","kjv","wbt","web","ylt"]
jacsim=pd.DataFrame(index=ver)
for i in ver:
    jacsim[i]=np.nan


# In[ ]:


#Calculate Jaccard Similarity of any of the two versions.
#Could be optimized by calculating (i,j) and (j,i) once.
ver=[asv,bbe,dby,kjv,wbt,web,ylt]
for i in range(7):
    for j in range(7):
        jacsim.iloc[i,j]=com_book(ver[i],ver[j])


# In[ ]:


#sns.heatmap(jacsim, annot=True)


# #### Future work: discover patterns in the text using unsupervised learning and NLP

# In[ ]:




