#!/usr/bin/env python
# coding: utf-8

# In[10]:


import numpy as np 
import pandas as pd

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))


# In[11]:


df = pd.read_csv("../input/abcnews-date-text.csv")
df.head()


# In[12]:


df.dtypes


#  In order to add a column, "Year" I need the following:
#  
#  Convert the data type (dtype) to string (str) using astype.
#  
#  Then used vectorised str method to slice the string and then convert back to int64 dtype again.

# In[13]:


df["year"] = df["publish_date"].astype(str).str[:4].astype(np.int64)
df.head()


# I can also add a "month" column:

# In[14]:


df["month"] = df["publish_date"].astype(str).str[4:6].astype(np.int64)
df.head()


# Making sure all "Years" and "Months" are correct:

# In[15]:


df.year.unique()


# In[16]:


df.month.unique()


# Add another column, Word_Count:

# In[17]:


df["word_count"] = df["headline_text"].str.len()
df.head()


# ## Number of Articles per year:

# In[18]:


get_ipython().run_line_magic('matplotlib', 'inline')

import matplotlib.pyplot as plt
import seaborn as sns

with sns.color_palette("GnBu_d", 10):
    ax= sns.countplot(y="year",data=df)
    ax.set(xlabel='Number of Articles', ylabel='Year')
plt.title("Number of Articles per Year")


# In[19]:


df["headline_text"][0]


# Remove words that don't carry much meaning: such words are called "stop words"(such as "a", "and", "is", and "the")

# In[22]:


from nltk.corpus import stopwords
stopwords.words("english")


# In[24]:


# Remove stop words from "words"

words = [w for w in words if not w in stopwords.words("english")]
words


# In[25]:


df.shape


# In[30]:


from sklearn.feature_extraction.text import CountVectorizer


vectorizer = CountVectorizer(analyzer = "word",   
                             tokenizer = None,    
                             preprocessor = None, 
                             stop_words = "english",   
                             max_features = 30)

news_array = vectorizer.fit_transform(df["headline_text"])

# Numpy arrays are easy to work with, so convert the result to an array
news_array = news_array.toarray()

# Lets take a look at the words in the vocabulary and  print the counts of each word in the vocabulary:
vocab = vectorizer.get_feature_names()

# Sum up the counts of each vocabulary word
dist = np.sum(news_array, axis=0)

# For each, print the vocabulary word and the number of times it appears in the training set
for tag, count in zip(vocab, dist):
    print (count, tag)


# In[ ]:




