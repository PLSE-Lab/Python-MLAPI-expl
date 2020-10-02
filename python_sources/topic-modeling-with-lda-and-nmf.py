#!/usr/bin/env python
# coding: utf-8

# # Topic Modeling 
# 
# Topic modeling is a statistical model to discover the abstract "topics" that occur in a collection of documents.  
# I will be focusing on two medhods seen as follows. 
# * LDA 
# * Non-negative matrix factorization  

# # Import libraries
# 
# I used LDA model from sklearn. Other option is using gensim.

# In[ ]:


import pandas as pd
import numpy as np


# ## Read the data

# In[ ]:


# Input from csv
df = pd.read_csv('../input/voted-kaggle-dataset.csv')

# sample data
print(df['Description'][0])


# In[ ]:


df['Title'][0]


# In[ ]:


df.columns


# In[ ]:


df.head()


# In[ ]:


# shape of data frame
len(df)


# In[ ]:


# is there any NaN values
df.isnull().sum()


# In[ ]:


# nan value in Description
df.Description.isnull().sum()


# In[ ]:


df.Tags[0]


# In[ ]:


#REMOVE NaN VALUES
df['Description'].dropna(inplace=True,axis=0)

# check if there is any NaN values
df.Description.isnull().sum()


# In[ ]:


# REMOVE EMPTY STRINGS:
blanks = []  # start with an empty list

for rv in df['Description']:  # iterate over the DataFrame
    if type(rv)==str:            # avoid NaN values
        if rv.isspace():         # test 'review' for whitespace
            blanks.append(i)     # add matching index numbers to the list
print(blanks)
df['Description'].drop(blanks, inplace=True)


# # Data preprocessing

# # 1.Initiating Tokenizer and Lemmatizer
# 
# Initiate the tokenizer, stop words, and lemmatizer from the libraries.
# 
# * Tokenizer is used to split the sentences into words.  
# * Lemmatizer (a quite similar term to Stemmer) is used to reduce words to its base form.   
# The simple difference is that Lemmatizer considers the meaning while Stemmer does not. 
# 

# In[ ]:


from nltk.tokenize import RegexpTokenizer
from stop_words import get_stop_words
from nltk.stem.wordnet import WordNetLemmatizer
import re
from nltk.corpus import stopwords

pattern = r'\b[^\d\W]+\b'
# \b is word boundry
# [^] is neget
# \d is digit and \W is not word

#tokenize from nltk
tokenizer = RegexpTokenizer(pattern)
#I created by myself
def tokenizer_man(doc,remove_stopwords=False):
    doc_rem_puct = re.sub(r'[^a-zA-Z]',' ',doc)
    words = doc_rem_puct.lower().split()    
    if remove_stopwords:
        stops = set(stopwords.words("english"))     
        words = [w for w in words if not w in stops]
    return words

en_stop = get_stop_words('en')
lemmatizer = WordNetLemmatizer()


# In[ ]:


#NTLK stopwords

#check how many stopwords you have
stops1=set(stopwords.words('english'))
print(stops1)
#lenght of stopwords
len(stopwords.words('english'))


# In[ ]:


#adding new element to the set
stops1.add('newWords') #newWord added into the stopwords
print(len(stops1))


# In[ ]:


raw = str(df['Description'][0]).lower()
tokens = tokenizer.tokenize(raw)
" ".join(tokens)
len(tokens)


# In[ ]:


#test manual 
string=df['Description'][0]
vocab = tokenizer_man(string)
" ".join(vocab)
len(vocab)


# In[ ]:


remove_words = ['data','dataset','datasets','content','context','acknowledgement','inspiration']


# ## Perform Tokenization, Words removal, and Lemmatization

# In[ ]:


# list for tokenized documents in loop
texts = []

# loop through document list
for i in df['Description'].iteritems():
    # clean and tokenize document string
    raw = str(i[1]).lower()
    tokens = tokenizer.tokenize(raw)

    # remove stop words from tokens
    stopped_tokens = [raw for raw in tokens if not raw in en_stop]
    
    # remove stop words from tokens
    stopped_tokens_new = [raw for raw in stopped_tokens if not raw in remove_words]
    
    # lemmatize tokens
    lemma_tokens = [lemmatizer.lemmatize(tokens) for tokens in stopped_tokens_new]
    
    # remove word containing only single char
    new_lemma_tokens = [raw for raw in lemma_tokens if not len(raw) == 1]
    
    # add tokens to list
    texts.append(new_lemma_tokens)

# sample data
print(texts[0])


# In[ ]:


len(texts)


# In[ ]:


df['desc_preprocessed'] = ""
for i in range(len(texts)):
    df['desc_preprocessed'][i] = ' '.join(map(str, texts[i]))


# In[ ]:


print(df['desc_preprocessed'][0])


# In[ ]:


df.shape


# In[ ]:


df.columns


# # Feature Extraction

# In[ ]:


from sklearn.feature_extraction.text import TfidfVectorizer


# In[ ]:


tfidf = TfidfVectorizer(max_df=0.9,min_df=2,stop_words='english')


# In[ ]:


dtm = tfidf.fit_transform(df['desc_preprocessed'])

dtm


# In[ ]:


from sklearn.decomposition import NMF,LatentDirichletAllocation


# # Non-negative Matrix Factorization

# In[ ]:


nmf_model = NMF(n_components=7,random_state=42)
nmf_model.fit(dtm)


# # LDA modelling

# In[ ]:


LDA = LatentDirichletAllocation(n_components=7,random_state=42)
LDA.fit(dtm)


# # Displaying Topics 

# In[ ]:


len(tfidf.get_feature_names())


# In[ ]:


# words for NMF modeling
for index,topic in enumerate(nmf_model.components_):
    print(f'THE TOP 15 WORDS FOR TOPIC #{index}')
    print([tfidf.get_feature_names()[i] for i in topic.argsort()[-15:]])
    print('\n')


# In[ ]:


# words for LDA modeling
for index,topic in enumerate(LDA.components_):
    print(f'THE TOP 15 WORDS FOR TOPIC #{index}')
    print([tfidf.get_feature_names()[i] for i in topic.argsort()[-15:]])
    print('\n')


# In[ ]:


topic_results = nmf_model.transform(dtm)
df['NMF_Topic'] = topic_results.argmax(axis=1)


# In[ ]:


LDA_topic_results = LDA.transform(dtm)
df['LDA_Topic'] = LDA_topic_results.argmax(axis=1)


# In[ ]:


mytopic_dict = {0:'public',
                1:'sports',
                2:'machine_learning',
                3:'neuron_network',
                4:'politic',
                5:'economy',
                6:'text analysis'
               }

df['topic_label_NMF']=df['NMF_Topic'].map(mytopic_dict)


# In[ ]:


df.head(-5)


# In[ ]:


df['LDA_Topic'].unique()


# In[ ]:


from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt

first_topic = nmf_model.components_[0]
first_topic_words = [tfidf.get_feature_names()[i] for i in first_topic.argsort()[:-15 - 1 :-1]]

firstcloud = WordCloud(
                          stopwords=STOPWORDS,
                          background_color='black',
                          width=4000,
                          height=2500
                         ).generate(" ".join(first_topic_words))
plt.imshow(firstcloud)
plt.axis('off')
plt.show()


# In[ ]:




