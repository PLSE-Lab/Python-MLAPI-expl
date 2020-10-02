#!/usr/bin/env python
# coding: utf-8

# ## Topic Modelling using Latent Semantic Analysis (LSA) and Latent Dirichlet Allocation (LDA) in sklearn

# In[ ]:





# ### **There also exists implementation using the Gensim library. Checkout the same [here](https://www.analyticsvidhya.com/blog/2016/08/beginners-guide-to-topic-modeling-in-python/)  ,   [here](https://nlpforhackers.io/topic-modeling/) and [here](https://www.machinelearningplus.com/nlp/topic-modeling-gensim-python/) and also in [this](https://github.com/susanli2016/NLP-with-Python/blob/master/LDA_news_headlines.ipynb) notebook.**

# In[ ]:





# ## [Please star/upvote in case u like it. ]

# In[ ]:





# #### IMPORTING MODULES

# In[ ]:


# data visualisation and manipulation
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import style
import seaborn as sns
#configure
# sets matplotlib to inline and displays graphs below the corressponding cell.
get_ipython().run_line_magic('matplotlib', 'inline')
style.use('fivethirtyeight')
sns.set(style='whitegrid',color_codes=True)

#import nltk
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize,sent_tokenize

#preprocessing
from nltk.corpus import stopwords  #stopwords
from nltk import word_tokenize,sent_tokenize # tokenizing
from nltk.stem import PorterStemmer,LancasterStemmer  # using the Porter Stemmer and Lancaster Stemmer and others
from nltk.stem.snowball import SnowballStemmer
from nltk.stem import WordNetLemmatizer  # lammatizer from WordNet

# for named entity recognition (NER)
from nltk import ne_chunk

# vectorizers for creating the document-term-matrix (DTM)
from sklearn.feature_extraction.text import TfidfVectorizer,CountVectorizer

#stop-words
stop_words=set(nltk.corpus.stopwords.words('english'))


# In[ ]:





# #### LOADING THE DATASET

# In[ ]:


df=pd.read_csv(r'../input/abcnews-date-text.csv')


# In[ ]:


df.head()


# In[ ]:





# We will drop the **'publish_date'** column as it is useless for our discussion.

# In[ ]:


# drop the publish date.
df.drop(['publish_date'],axis=1,inplace=True)


# In[ ]:


df.head(10)


# In[ ]:





# #### DATA CLEANING & PRE-PROCESSING

# Here I have done the data pre-processing. I have used the lemmatizer and can also use the stemmer. Also the stop words have been used along with the words wit lenght shorter than 3 characters to reduce some stray words.

# In[ ]:


def clean_text(headline):
  le=WordNetLemmatizer()
  word_tokens=word_tokenize(headline)
  tokens=[le.lemmatize(w) for w in word_tokens if w not in stop_words and len(w)>3]
  cleaned_text=" ".join(tokens)
  return cleaned_text
  
  


# In[ ]:


# time taking
df['headline_cleaned_text']=df['headline_text'].apply(clean_text)


# In[ ]:


df.head()


# Can see the difference after removal of stopwords and some shorter words. aslo the words have been lemmatized as in **'calls'--->'call'.**

# In[ ]:





# Now drop the unpre-processed column.

# In[ ]:


df.drop(['headline_text'],axis=1,inplace=True)


# In[ ]:


df.head()


# In[ ]:





# We can also see any particular news headline.

# In[ ]:


df['headline_cleaned_text'][0]


# In[ ]:





# #### EXTRACTING THE FEATURES AND CREATING THE DOCUMENT-TERM-MATRIX ( DTM )
# 
# In DTM the values are the TFidf values.
# 
# Also I have specified some parameters of the Tfidf vectorizer.
# 
# Some important points:-
# 
# **1) LSA is generally implemented with Tfidf values everywhere and not with the Count Vectorizer.**
# 
# **2) max_features depends on your computing power and also on eval. metric (coherence score is a metric for topic model). Try the value that gives best eval. metric and doesn't limits processing power.**
# 
# **3) Default values for min_df & max_df worked well.**
# 
# **4) Can try different values for ngram_range.**

# In[ ]:


vect =TfidfVectorizer(stop_words=stop_words,max_features=1000) # to play with. min_df,max_df,max_features etc...


# In[ ]:


vect_text=vect.fit_transform(df['headline_cleaned_text'])


# In[ ]:





# #### We can now see the most frequent and rare words in the news headlines based on idf score. The lesser the value; more common is the word in the news headlines.

# In[ ]:


print(vect_text.shape)
print(vect_text)


# In[ ]:


idf=vect.idf_


# In[ ]:


dd=dict(zip(vect.get_feature_names(), idf))
l=sorted(dd, key=(dd).get)
# print(l)
print(l[0],l[-1])
print(dd['police'])
print(dd['forecast'])  # police is most common and forecast is least common among the news headlines.


# We can therefore see that on the basis of the **idf value** , **'police'** is the **most frequent** word while **'forecast'** is **least frequently** occuring among the news.

# In[ ]:





# ### TOPIC MODELLING

# In[ ]:





# ## Latent Semantic Analysis (LSA)

# The first approach that I have used is the LSA. **LSA is basically singular value decomposition.**
# 
# **SVD decomposes the original DTM into three matrices S=U.(sigma).(V.T). Here the matrix U denotes the document-topic matrix while (V) is the topic-term matrix.**
# 
# **Each row of the matrix U(document-term matrix) is the vector representation of the corresponding document. The length of these vectors is the number of desired topics. Vector representation for the terms in our data can be found in the matrix V (term-topic matrix).**
# 
# So, SVD gives us vectors for every document and term in our data. The length of each vector would be k. **We can then use these vectors to find similar words and similar documents using the cosine similarity method.**
# 
# We can use the truncatedSVD function to implement LSA. The n_components parameter is the number of topics we wish to extract.
# The model is then fit and transformed on the result given by vectorizer. 
# 
# **Lastly note that LSA and LSI (I for indexing) are the same and the later is just sometimes used in information retrieval contexts.**

# In[ ]:


from sklearn.decomposition import TruncatedSVD
lsa_model = TruncatedSVD(n_components=10, algorithm='randomized', n_iter=10, random_state=42)

lsa_top=lsa_model.fit_transform(vect_text)


# In[ ]:


print(lsa_top)
print(lsa_top.shape)  # (no_of_doc*no_of_topics)


# In[ ]:


l=lsa_top[0]
print("Document 0 :")
for i,topic in enumerate(l):
  print("Topic ",i," : ",topic*100)
  


# Similalry for other documents we can do this. However note that values dont add to 1 as in LSA it is not probabiltiy of a topic in a document.

# In[ ]:


print(lsa_model.components_.shape) # (no_of_topics*no_of_words)
print(lsa_model.components_)


# In[ ]:





# #### Now e can get a list of the important words for each of the 10 topics as shown. For simplicity here I have shown 10 words for each topic.

# In[ ]:


# most important words for each topic
vocab = vect.get_feature_names()

for i, comp in enumerate(lsa_model.components_):
    vocab_comp = zip(vocab, comp)
    sorted_words = sorted(vocab_comp, key= lambda x:x[1], reverse=True)[:10]
    print("Topic "+str(i)+": ")
    for t in sorted_words:
        print(t[0],end=" ")
    print("\n")
         


# In[ ]:





# ## Latent Dirichlet Allocation (LDA)  

# LDA is the most popular technique.**The topics then generate words based on their probability distribution. Given a dataset of documents, LDA backtracks and tries to figure out what topics would create those documents in the first place.**
# 
# **To understand the maths it seems as if knowledge of Dirichlet distribution (distribution of distributions) is required which is quite intricate and left fior now.**
# 
# To get an inituitive explanation of LDA checkout these blogs: [this](https://www.analyticsvidhya.com/blog/2016/08/beginners-guide-to-topic-modeling-in-python/)  ,  [this](https://tedunderwood.com/2012/04/07/topic-modeling-made-just-simple-enough/)  ,[this](https://en.wikipedia.org/wiki/Topic_model)  ,  [this kernel on Kaggle](https://www.kaggle.com/arthurtok/spooky-nlp-and-topic-modelling-tutorial)  ,  [this](http://blog.echen.me/2011/08/22/introduction-to-latent-dirichlet-allocation/) .

# In[ ]:


from sklearn.decomposition import LatentDirichletAllocation
lda_model=LatentDirichletAllocation(n_components=10,learning_method='online',random_state=42,max_iter=1) 
# n_components is the number of topics


# In[ ]:


lda_top=lda_model.fit_transform(vect_text)


# In[ ]:


print(lda_top.shape)  # (no_of_doc,no_of_topics)
print(lda_top)


# In[ ]:


sum=0
for i in lda_top[0]:
  sum=sum+i
print(sum)  


# #### Note that the values in a particular row adds to 1. This is beacuse each value denotes the % of contribution of the corressponding topic in the document.

# In[ ]:


# composition of doc 0 for eg
print("Document 0: ")
for i,topic in enumerate(lda_top[0]):
  print("Topic ",i,": ",topic*100,"%")


# #### As we can see Topic 7 & 8 are dominantly present in document 0.
# 
#  

# In[ ]:


print(lda_model.components_)
print(lda_model.components_.shape)  # (no_of_topics*no_of_words)


# #### Most important words for a topic. (say 10 this time.)

# In[ ]:


# most important words for each topic
vocab = vect.get_feature_names()

for i, comp in enumerate(lda_model.components_):
    vocab_comp = zip(vocab, comp)
    sorted_words = sorted(vocab_comp, key= lambda x:x[1], reverse=True)[:10]
    print("Topic "+str(i)+": ")
    for t in sorted_words:
        print(t[0],end=" ")
    print("\n")


# In[ ]:





# #### To better visualize words in a topic we can see the word cloud. For each topic top 50 words are plotted.

# In[ ]:


from wordcloud import WordCloud
# Generate a word cloud image for given topic
def draw_word_cloud(index):
  imp_words_topic=""
  comp=lda_model.components_[index]
  vocab_comp = zip(vocab, comp)
  sorted_words = sorted(vocab_comp, key= lambda x:x[1], reverse=True)[:50]
  for word in sorted_words:
    imp_words_topic=imp_words_topic+" "+word[0]

  wordcloud = WordCloud(width=600, height=400).generate(imp_words_topic)
  plt.figure( figsize=(5,5))
  plt.imshow(wordcloud)
  plt.axis("off")
  plt.tight_layout()
  plt.show()
 


# In[ ]:


# topic 0
draw_word_cloud(0)


# In[ ]:


# topic 1
draw_word_cloud(1)  # ...


# In[ ]:





# ## THE END !!!

# ## [Please star/upvote in case u liked it. ]

# In[ ]:




