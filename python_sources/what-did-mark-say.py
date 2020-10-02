#!/usr/bin/env python
# coding: utf-8

# <h1><center>What did Mark Say?</center></h1>
# <img src="https://cdn.wccftech.com/wp-content/uploads/2018/03/facebook-2060x1217.jpg">

# ### Let's import the data and do some basic summary

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import nltk.corpus as corpus
import nltk


# In[ ]:


data=pd.read_csv("../input/mark.csv")


# In[ ]:


## Let's find out hou long were Mark's Responses?
def get_count(x):
    return len(nltk.word_tokenize(x))
data['len']=data['Text'].map(get_count)


# In[ ]:


data.head()


# ### How verbose were Mark's responses?

# In[ ]:


print("The total words spoken by Mark were {} words".format(data.query("Person=='ZUCKERBERG:'")['len'].sum()))
print("The average length of his response weas {} words".format(round(data.query("Person=='ZUCKERBERG:'")['len'].mean(),2)))
print("The maximum length of Mark's response was {} words".format(data.query("Person=='ZUCKERBERG:'")['len'].max()))


# ### Which senator spoke the most

# In[ ]:


data.query("Person !='ZUCKERBERG:'").groupby("Person").sum().rename(columns={'len':'Total Words'}).sort_values("Total Words",ascending=False).head(20).plot(kind="barh",colormap="Set2",figsize=(8,8))
plt.title("Total Words Spoken",fontsize=30)
plt.ylabel("Senator",fontsize=25)
plt.yticks(fontsize=15)
plt.xlabel("Count",fontsize=15)


# ### Common Themes Mark talked about?

# In[ ]:


##  Most commonly used words by Mark
from sklearn.feature_extraction import text
def get_imp(bow,mf,ngram):
    tfidf=text.CountVectorizer(bow,ngram_range=(ngram,ngram),max_features=mf,stop_words='english')
    matrix=tfidf.fit_transform(bow)
    return pd.Series(np.array(matrix.sum(axis=0))[0],index=tfidf.get_feature_names()).sort_values(ascending=False).head(100)


# #### Most common bigrams

# In[ ]:


mark=data[data['Person']=="ZUCKERBERG:"]['Text'].tolist()
get_imp(mark,mf=5000,ngram=2).head(10)


# #### Most common trigrams

# In[ ]:


get_imp(mark,mf=5000,ngram=3).head(10)


# > ### Rinse and repeat using tfidf

# In[ ]:


def get_imp_tf(bow,mf,ngram):
    tfidf=text.TfidfVectorizer(bow,ngram_range=(ngram,ngram),max_features=mf,stop_words='english')
    matrix=tfidf.fit_transform(bow)
    return pd.Series(np.array(matrix.sum(axis=0))[0],index=tfidf.get_feature_names()).sort_values(ascending=False).head(100)


# ### Common bigrams (Tfidf)

# In[ ]:


get_imp_tf(mark,mf=5000,ngram=2).head(10)


# ### Common trigrams (tfidf)

# In[ ]:


get_imp_tf(mark,mf=5000,ngram=3).head(10)


# In[ ]:




