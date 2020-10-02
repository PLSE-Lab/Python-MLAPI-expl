#!/usr/bin/env python
# coding: utf-8

# In[73]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

import os
print(os.listdir("../input"))


# In[74]:


# import dataset
# column is index of data, ignore
df = pd.read_csv('../input/train.txt', index_col=0)
df.head()


# In[75]:


from nltk.tokenize import TweetTokenizer

# data cleansing
# replace "class" as tone: Neg -> False(-ve); Pos -> True(+ve)
df['class'] = [True if i == 'Pos' else False for i in df['class']]
# add word count
tknzr = TweetTokenizer()
df['word_count'] = [len(tknzr.tokenize(record)) for record in df['text']]


# In[76]:


df.head()


# In[77]:


print("Total Neg comment:", len(df[df['class']==False]))
print("Total Pos comment:", len(df[df['class']==True]))


# Hence, Dataset is Balanced Data

# In[78]:


sns.boxplot(y="word_count", x='class', data=df) # relationship between word count and class


# ### Word Cloud

# In[79]:


from wordcloud import WordCloud, STOPWORDS

sw = set(STOPWORDS)
wc = WordCloud(max_words=50, background_color="white",margin=2, stopwords=sw)


# In[80]:


plt.imshow(wc.generate(' '.join(df[df['class'] == False]['text']))) 
plt.axis("off")
plt.title("Frequent Words appeared in Neg Text")


# In[81]:


plt.imshow(wc.generate(' '.join(df[df['class'] == True]['text'])))
plt.axis("off")
plt.title("Frequent Words appeared in Pos Text")


# ### check word cloud again after stemming

# In[82]:


training_data = df.iloc[:, 0:2]
training_data.tail()


# In[83]:


# data cleansing
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer 
from nltk.stem import PorterStemmer 
import re


def remove_numbers(text):
    '''function for remove 0-9 in text'''
    return re.sub(r'[0-9\.]+', '', text) 

def remove_stopwords(text):
    '''a function for removing the stopword'''
    # extracting the stopwords from nltk library
    sw = stopwords.words('english')
    sw.extend(
        ["ford", "focus", "br", "quot", "good", "great", "new"]
    )
    # removing the stop words and lowercasing the selected words
    text = [word.lower() for word in text.split() if word.lower() not in sw]
    # joining the list of words with space separator
    return " ".join(text)

def lemmatize(text):
    '''function for lemmatize'''    
    lemmatizer = WordNetLemmatizer()
#     ps = PorterStemmer()
#     text = " ".join([ps.stem(i) for i in text.split(" ")])
    return " ".join([lemmatizer.lemmatize(i) for i in text.split(" ")])

# Apply data cleaning
# punctuation is already cleaned before obtain
training_data['text'] = training_data['text'].apply(remove_numbers)
training_data['text'] = training_data['text'].apply(remove_stopwords)
training_data['text'] = training_data['text'].apply(lemmatize)
training_data['text'] = training_data['text'].apply(remove_stopwords)


# In[84]:


training_data.tail()


# In[85]:


wc2 = WordCloud(max_words=20, background_color="white",margin=2)


# In[86]:


import nltk


# In[87]:


neg_text = ' '.join(training_data[training_data['class'] == False]['text'])
neg_token = nltk.word_tokenize(neg_text)
plt.imshow(wc2.generate(neg_text))
plt.axis("off")
plt.title("Frequent Words appeared in Pos Text")


# In[88]:


pos_text = ' '.join(training_data[training_data['class'] == True]['text'])
pos_token = nltk.word_tokenize(pos_text)
plt.imshow(wc2.generate(pos_text))
plt.axis("off")
plt.title("Frequent Words appeared in Pos Text")


# In[92]:


_neg_token = ' '.join([i[0] for i in nltk.pos_tag(neg_token) if "JJ" in i[1]])
plt.imshow(wc2.generate(_neg_token))
plt.axis("off")
plt.title("Frequent Adjective appeared in Pos Text")


# In[93]:


_pos_token = ' '.join([i[0] for i in nltk.pos_tag(pos_token) if "JJ" in i[1]])
plt.imshow(wc2.generate(_pos_token))
plt.axis("off")
plt.title("Frequent Adjective appeared in Pos Text")


# In[96]:


from sklearn.pipeline import make_pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score

classifier = make_pipeline(TfidfVectorizer(), TruncatedSVD(4500), SVC(gamma='scale', class_weight='balanced'))
result = cross_val_score(classifier, training_data['text'].values.tolist(), training_data['class'].values.tolist(), cv=10, groups=df['class'])
print(result)
print("Average:", result.mean())

