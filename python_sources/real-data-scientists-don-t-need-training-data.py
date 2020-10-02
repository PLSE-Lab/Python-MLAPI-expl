#!/usr/bin/env python
# coding: utf-8

# *This script is based on the earlier conclusions from the script here : https://www.kaggle.com/artimous/d/quora/question-pairs-dataset/deciphering-the-quora-bot*
# 
# 
# Getting common words
# --------------------
# 
# No syntax and semantics analysis here. Going simple to analyse all the common words in the two question sets and visualizing them.

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

import warnings
warnings.filterwarnings('ignore')

try:
    t_file = pd.read_csv('../input/test.csv', encoding='ISO-8859-1')
    tr_file = pd.read_csv('../input/train.csv', encoding ='ISO-8859-1')
    print('File load: Success')
except:
    print('File load: Failed')


# **Removing stop words from ntlk**

# In[ ]:


from nltk.corpus import stopwords
stop = stopwords.words('english')
print(stop)


# CLEANING
# --------
# 
# Simply dropped null values, split each of the question strings and removed stops

# In[ ]:


t_file = t_file.dropna()
t_file['question1'] = t_file['question1'].str.lower().str.split()
t_file['question2'] = t_file['question2'].str.lower().str.split()
t_file['question1'] = t_file['question1'].apply(lambda x: [item for item in x if item not in stop])
t_file['question2'] = t_file['question2'].apply(lambda x: [item for item in x if item not in stop])


# In[ ]:


tr_file = tr_file.dropna()
tr_file['question1'] = tr_file['question1'].str.lower().str.split()
tr_file['question2'] = tr_file['question2'].str.lower().str.split()
tr_file['question1'] = tr_file['question1'].apply(lambda x: [item for item in x if item not in stop])
tr_file['question2'] = tr_file['question2'].apply(lambda x: [item for item in x if item not in stop])


# **Finding common word percentage and average word lengths**

# In[ ]:


tr_file['Common'] = tr_file.apply(lambda row: len(list(set(row['question1']).intersection(row['question2']))), axis=1)
tr_file['Average'] = tr_file.apply(lambda row: 0.5*(len(row['question1'])+len(row['question2'])), axis=1)
tr_file['Percentage'] = tr_file.apply(lambda row: row['Common']*100.0/(row['Average']+1), axis=1)


# In[ ]:


t_file['Common'] = t_file.apply(lambda row: len(list(set(row['question1']).intersection(row['question2']))), axis=1)
t_file['Average'] = t_file.apply(lambda row: 0.5*(len(row['question1'])+len(row['question2'])), axis=1)
t_file['Percentage'] = t_file.apply(lambda row: 1 if row['Average'] == 0 else row['Common']/(row['Average']), axis=1)


# **True and False plotting of data**

# Cheating the title
# ------------------
# 
# We can take a look at the training file right? No training still counts as good work? 

# In[ ]:


y = tr_file['Percentage'][tr_file['is_duplicate']==0].values
x = tr_file['Average'][tr_file['is_duplicate']==0].values

fig, axs = plt.subplots(ncols=2, sharey=True, figsize=(7, 4))
fig.subplots_adjust(hspace=0.5, left=0.07, right=0.93)
ax = axs[0]
hb = ax.hexbin(x, y, gridsize=70, bins='log', cmap='inferno')
ax.axis([0, 20, 0, 100])
ax.set_title("Duplicates")
cb = fig.colorbar(hb, ax=ax)
cb.set_label('log10(N)')


y = tr_file['Percentage'][tr_file['is_duplicate']==1].values
x = tr_file['Average'][tr_file['is_duplicate']==1].values
ax = axs[1]
hb = ax.hexbin(x, y, gridsize=70, bins='log', cmap='inferno')
ax.axis([0, 20, 0, 100])
ax.set_title("Not duplicates")
cb = fig.colorbar(hb, ax=ax)
cb.set_label('log10(N)')

plt.show()


# **Scatters are a must**

# In[ ]:


x = tr_file['Percentage'][tr_file['is_duplicate']==0].values
y = tr_file['qid1'][tr_file['is_duplicate']==0].values
area = tr_file['Average'][tr_file['is_duplicate']==0].values

plt.scatter(x, y, s=area*3, c='r', alpha=0.1)

x = tr_file['Percentage'][tr_file['is_duplicate']==1].values
y = tr_file['qid1'][tr_file['is_duplicate']==1].values
area = tr_file['Average'][tr_file['is_duplicate']==1].values

plt.scatter(x, y, s=area*3, c='b', alpha=0.1)

plt.ylabel('Question IDs')
plt.xlabel('Percentage of common words')

plt.title("Percentages of common words in questions")
plt.show()


# **Observations**
# ----------------
# 
# From the final plot it is pretty clearly visible that the ones that are clustered towards the 100% mark are nearly all blue. This states the fact that questions having a lot of common strings are termed as equivalent more often than not.
# Also as seen in the hex plot, non duplicates are clustered towards the 100% area more than the duplicate ones.
# Have we decoded the Quora bot? Not at all.

# The final step
# --------------
# 
# Penning down the final result into and output file. This approach is just naive.

# In[ ]:


df2 = pd.DataFrame({'test_id' : range(0,2345796)})
df2['is_duplicate']=pd.Series(t_file['Percentage'])
df2.fillna(0, inplace = True)
print(df2.shape)


# In[ ]:


df2.to_csv('submit_naive.csv', index=False)


#  Cosine, Jaccard and Shingling
# --------------------------------
# 
# The first, naive approach towards identifying question pairs -- Strip the stopwords, stem the remaining and do a simple Cosine/Jaccard Test. K-Shingling is also a popular technique, where continuous subsets of "k" words are matched between the two documents.
# However, a major drawback with the above is that of a lack of semantic understanding -- There might be two questions with a high percentage of common words, but different meanings.

# In[ ]:


from collections import Counter

def get_cosine(vec1, vec2):
    vec1 = Counter(vec1)
    vec2 = Counter(vec2)
    intersection = set(vec1.keys()) & set(vec2.keys())
    numerator = sum([vec1[x] * vec2[x] for x in intersection])
    sum1 = sum([vec1[x]**2 for x in vec1.keys()])
    sum2 = sum([vec2[x]**2 for x in vec2.keys()])
    denominator = math.sqrt(sum1) * math.sqrt(sum2)
    if not denominator:
        return 0.0
    else:
        return float(numerator) / denominator


# In[ ]:


t_file['Cosine'] = t_file.apply(lambda row: get_cosine(row['question1'],row['question2']), axis=1)
print(t_file)


# Jaccard Similarity
# ------------------
# 
# Jaccard Similarity is given by s=p/(p+q+r)
# where,
# 
# - p = # of attributes positive for both objects 
# - q = # of attributes 1 for i and 0 for j 
# - r = # of attributes 0 for i and 1 for j 
# 

# In[ ]:


t_file['Jaccard'] = t_file.apply(lambda row: float(row['Common'])/((len(row['question1'])+len(row['question2'])-row['Common'])), axis=1)
print(t_file)

