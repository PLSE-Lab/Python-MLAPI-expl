#!/usr/bin/env python
# coding: utf-8

# # Quora data exploration (EDA) and Topic modeling (LSA vs LDA)

# ## Import of libraries

# In[ ]:


import numpy as np 
import pandas as pd 
import matplotlib as mp
import os
from nltk.corpus import stopwords
from nltk.corpus import opinion_lexicon
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation,TruncatedSVD
import matplotlib.pyplot as plt


# ## Data upload

# In[ ]:


#uploading data in dataframe
train=pd.read_csv("../input/train.csv",sep=',')
test=pd.read_csv("../input/test.csv",sep=',')


# In[ ]:


#displayin shapes
print ('train shapes : %s'%str(train.shape))
print ('test shapes : %s'%str(test.shape))


# In[ ]:


#displaying exemple data
train.head(5)


# In[ ]:


#displaying exemple of insincere data 
train[train.target==1].head(5)


# In[ ]:


#displayin dataframe info
train.info()


# There is no missing values 

# In[ ]:


#counting target values
train.target.value_counts()


# We have to deal with unbalanced target Feature...

# ## Feature extraction

# I tried first to make prediction with only extracted features but the result wasn't good

# In[ ]:


positive=opinion_lexicon.positive()
negative=opinion_lexicon.negative()
stop = stopwords.words('english')
print(len(positive))
print(len(negative))
print(len(stop))


# In[ ]:


train['word_count'] = train['question_text'].apply(lambda x: len(str(x).split(" ")))
#train['char_count'] = train['question_text'].str.len()
#stop = stopwords.words('english')
#train['stopwords'] = train['question_text'].apply(lambda x: len([x for x in x.split() if x in stop]))
#train['numerics'] = train['question_text'].apply(lambda x: len([x for x in x.split() if x.isdigit()]))
#train['upper'] = train['question_text'].apply(lambda x: len([x for x in x.split() if x.isupper()]))
positive=opinion_lexicon.positive()
negative=opinion_lexicon.negative()
#train['postive'] = train['question_text'].apply(lambda x: len([x for x in x.split() if x in positive]))
train['negative'] = train['question_text'].apply(lambda x: len([x for x in x.split() if x in negative]))


# In[ ]:


#basic statistic about word_count
train.word_count.describe()


# In[ ]:


#ploting box plot of word_count by target without outlier
train.boxplot(column='word_count', by='target', grid=False,showfliers=False)


# As we can see, Quora's questions is composed from few word (mainly <25 words ). the distribution for in insincere question is more spread out.

# In[ ]:


#ploting box plot of word_count by target without outlier
#train.boxplot(column='positive', by='target', grid=False,showfliers=False)


# In[ ]:


#ploting box plot of word_count by target without outlier
train.boxplot(column='negative', by='target', grid=False,showfliers=False)


# ## Text transformation

# In[ ]:


#lower case
train['question_text'] = train['question_text'].apply(lambda x: " ".join(x.lower() for x in x.split()))
#Removing Punctuation
train['question_text'] = train['question_text'].str.replace('[^\w\s]','')
#Removing numbers
train['question_text'] = train['question_text'].str.replace('[0-9]','')
#Remooving stop words and words with length <=2
from nltk.corpus import stopwords
stop = stopwords.words('english')
train['question_text'] = train['question_text'].apply(lambda x: " ".join(x for x in x.split() if x not in stop and len(x)>2))
#Stemming
#from nltk.stem import SnowballStemmer
#ss=SnowballStemmer('english')
#train['question_text'] = train['question_text'].apply(lambda x: " ".join(ss.stem(x) for x in x.split()))
from nltk.stem import WordNetLemmatizer
wl = WordNetLemmatizer()
train['question_text'] = train['question_text'].apply(lambda x: " ".join(wl.lemmatize(x,'v') for x in x.split()))


# I tested several words on nltk stemmer et lemmatizer and i choose to use snowball stemmer

# In[ ]:


from nltk.stem import SnowballStemmer,WordNetLemmatizer,PorterStemmer,LancasterStemmer
wl = WordNetLemmatizer()
ss=SnowballStemmer('english')
ps=PorterStemmer()
ls=LancasterStemmer()
test_list=['does','peaople','writing','beards','enjoyment','bought','leaves','gave','given','generaly','would']
for item in test_list :
    print('lemmatizer : %s'%wl.lemmatize(item,'v'))
    print('SS stemmer : %s'%ss.stem(item))
    print('PS stemmer : %s'%ps.stem(item))
    print('LS stemmer : %s'%ls.stem(item))


# In[ ]:


train.head(5)


# ## Most frequent terms for sincere and insincere questions
# 
# get_words_freq return for a corpus the sorted list of words by frequency  

# In[ ]:


def get_words_freq(corpus):
    vec = CountVectorizer(ngram_range={1,2}).fit(corpus)
    #bag of words its a sparse document item matrix
    bag_of_words = vec.transform(corpus)
    #we calculate the occurrence for each term. warning, the sum of matrix is a 1 row matrix
    sum_words = bag_of_words.sum(axis=0) 
    # Vocabulary_ its a dictionary { word :position }  
    words_freq = [(word, sum_words[0, idx]) for word, idx in     vec.vocabulary_.items()]
    words_freq =sorted(words_freq, key = lambda x: x[1],reverse=True)
    return words_freq


# Let' see the 30 most frequent terms of sincere question :

# In[ ]:


top_sincere=get_words_freq(train[train.target==0].question_text)
print(top_sincere[:30])


# Let' see the 30 most frequent terms of insincere question :

# In[ ]:


top_insincere=get_words_freq(train[train.target==1].question_text)
print(top_insincere[:30])


# In[ ]:


#[y[0] for y in top_sincere].index('black people')


# We can also use wordcloud to visualize the most frequent terms for insincere questions

# In[ ]:


from wordcloud import WordCloud
wc=WordCloud(background_color='white')
wc.generate(''.join(train[train.target==1].question_text))


# In[ ]:


#let's plot
plt.figure(1, figsize=(15, 15))
plt.axis('off')
plt.imshow(wc)
plt.show()


# ## Topic Modeling insincere questions
# For topic modeling we are going to use a TFIDF matrix transformation.

# In[ ]:


tfidf_v = TfidfVectorizer(min_df=20,max_df=0.8,sublinear_tf=True,ngram_range={1,2})
#matrixTFIDF= tfidf_v.fit_transform(train.question_text)
matrixTFIDF= tfidf_v.fit_transform(train[train.target==1].question_text)


# In[ ]:


print(matrixTFIDF.shape)


# In[ ]:


plt.boxplot(np.array(matrixTFIDF.mean(axis=0).transpose()),showfliers=False)
plt.show()


# ### Topic modeling using LSA

# In[ ]:


svd=TruncatedSVD(n_components=15, n_iter=10,random_state=42)
X=svd.fit_transform(matrixTFIDF)             


# In[ ]:


plt.plot(svd.singular_values_[0:15])


# In[ ]:


#Explained variance by our components
np.sum(svd.explained_variance_ratio_[0:15])


# 5%! it's very low...

# In[ ]:


#components_ give the word contribution for each component 
svd.components_.shape


# get_topics give the n most contributif words in a topic

# In[ ]:


def get_topics(components, feature_names, n=15):
    for idx, topic in enumerate(components):
        print("Topic %d:" % (idx))
        print([(feature_names[i], topic[i])
                        for i in topic.argsort()[:-n - 1:-1]])


# In[ ]:


get_topics(svd.components_,tfidf_v.get_feature_names())


# ### Topic modeling using LDA

# In[ ]:


lda=LatentDirichletAllocation(n_components=15,random_state=42,max_iter=10)
Z=lda.fit_transform(matrixTFIDF)  


# In[ ]:


get_topics(lda.components_,tfidf_v.get_feature_names(),n=15)

