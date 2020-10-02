#!/usr/bin/env python
# coding: utf-8

# Details on Analysis.
# 
# It also showcases  on online community which is most popular in terms of User's and counts of words.
# At the End of the Analysis there is code which will give words which is present for that source.
# 
# There is no/very less correlation between online communities and every online community is different from each other.
# Also gives the most frequently used words on the online communities.
# 

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
import seaborn as sns

data=pd.read_csv('../input/wordgame_20170628.csv')
print(data.head())
np.shape(data)
obs=len(data.index)
##g= plt.hist(data['source'])
##g= plt.xlabel("number of words in a source")
##g= plt.ylabel("percent of words")
g= sns.factorplot(x='source', y = 'author', data = data, kind='bar',size=2.5, aspect =5.0)
plt.title('Most popular Online community with respect to Author')
plt.show()


# In[2]:


g1= sns.factorplot(x='source', data = data, kind='count',size=2.5, aspect =5.0)
plt.title('Most popular Online community with respect to count')
plt.show()


# In[3]:


#Creating Dummy variables
data_source = pd.get_dummies(data['source'])
data_new=pd.concat([data, data_source], axis=1)
data_new.head()
np.shape(data_new)


# In[4]:


#corelation between sources
type(data_new)
data_ne1 = data_new[data_new.columns[4:]]
data_ne1.head()
corrmat=data_ne1.corr()
g= sns.heatmap(corrmat, vmax=0.8,square=1)
plt.show()
print(" There is negative or no Co-Relation between online communities")


# In[5]:


## Coorelations -Authors with online community##
corrmat1=data_new.corr()
g1= sns.heatmap(corrmat1, vmax=0.8,square=1)
plt.show()
type(data_new)
data_new.head()
print(" Tell us that Author has the highest corelation with the online community 'Wrongplanet'.")


# In[6]:


data_testing = data_new[:]
data_testing


# In[7]:


## Gives the important words used on Online community sources##
## By changing the Sources in result it will showcase the important/frequent words used for that online community##
## Also removed duplicated words which are present for online community in word1 and word2 col. This will make sure##
## the words are not identical for same online community##
from wordcloud import WordCloud, STOPWORDS
def source_word(source):
    data_source = data_testing.set_index(source)
    w1=data_source.get_value(1,'word1')
    w2=data_source.get_value(1,'word2')
    source=np.concatenate((w1,w2),axis=0)
    source_rm_dup=str(list(set(source)))
    word_source = WordCloud(stopwords=STOPWORDS,
                          background_color='black',
                          width=5000,
                          height=5000
                         ).generate(source_rm_dup)
    return word_source
result = source_word('aspiecentral')
plt.title('For Source aspiecentral -  no duplicate words. combined words for words1 and words2')
plt.imshow(result)
plt.axis('off')
plt.show()


# In[23]:


## Result for Source ##
result1 = source_word('atu2')
plt.imshow(result1)
plt.axis('off')
plt.title('For Source atu2 -  no duplicate words. combined words for words1 and words2')
plt.show()


# In[22]:


## Result for Source ##
result2 = source_word('bleeping_computer')
plt.imshow(result2)
plt.axis('off')
plt.title('For Source bleeping_computer-  no duplicate words. combined words for words1 and words2')
plt.show()


# In[21]:


## Result for Source ##
result3 = source_word('classic_comics')
plt.imshow(result3)
plt.axis('off')
plt.title('For Source classic_comics -  no duplicate words. combined words for words1 and words2')
plt.show()


# In[20]:


## Result for Source ##
result4 = source_word('gog')
plt.imshow(result4)
plt.axis('off')
plt.title('For Source gog -  no duplicate words. combined words for words1 and words2')
plt.show()


# In[19]:


## Result for Source ##
result5 = source_word('learn_english')
plt.imshow(result5)
plt.axis('off')
plt.title('For Source learn_english -  no duplicate words. combined words for words1 and words2')
plt.show()


# In[13]:


## Result for Source ##
result6 = source_word('sas')
plt.imshow(result6)
plt.axis('off')
plt.title('For Source sas -  no duplicate words. combined words for words1 and words2')
plt.show()


# In[14]:


## Result for Source ##
result7 = source_word('the_fishy')
plt.imshow(result7)
plt.axis('off')
plt.title('For Source the_fishy -  no duplicate words. combined words for words1 and words2')
plt.show()


# In[15]:


## Result for Source ##
result8 = source_word('wrongplanet')
plt.imshow(result8)
plt.axis('off')
plt.title('For Source wrongplanet -  no duplicate words. combined words for words1 and words2')
plt.show()


# In[16]:


## Result for Source ##
result9 = source_word('ecig')
plt.imshow(result9)
plt.axis('off')
plt.title('For Source ecig -  no duplicate words. combined words for words1 and words2')
plt.show()


# In[17]:


## With source and word as text it will show whether the word is found for that online community##
## By changing the Source and word it will give the output as word found for that source or not##
import re
def source_txt(source,text):
    data_source = data_testing.set_index(source)
    w1=data_source.get_value(1,'word1')
    w2=data_source.get_value(1,'word2')
    source=np.concatenate((w1,w2),axis=0)
    source_rm_dup=str(list(set(source)))
    if text in source_rm_dup:
        text_value = (text , 'found')
    else:
        text_value = (text , 'not found')
    return text_value

## By changing the source and text , it will give if the text is found in that online community or not##
result_txt = source_txt('aspiecentral','e')
print('For Source Result is ',result_txt)


# In[18]:


##Gives the txt / word - sakiller is not found in online community aspiecentral
result_txt1 = source_txt('aspiecentral',"sakiller")
print('For Source Result is ',result_txt)

