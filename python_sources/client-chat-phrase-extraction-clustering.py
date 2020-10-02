#!/usr/bin/env python
# coding: utf-8

# **Client Chat Phrase extraction & clustering **
# 
# **Getting my hands dirty with Client Chat Phrase extraction & clustering I used different techniques to clean text data to improve accuracy. I tried stopwards removal,stemming, tfidfvectorization, kmeans algorithm for clustering. Please note that most of the data in dataset consists of words which are in hinglish - "hindi language words written in english" and hence prebuilt pos taggers and word2vec will not give good accuracy on such data.
# And due to short on time for this kernal right now I will not be able to build custom pos tagger or word2vec.**
# 

# 1. **Importing Important functions/Libraries
# NLTK
# spacy
# ngrams
# stopwards**
# 
# 2. **Load dataset from csv to pandas datadrame**
# 

# In[ ]:


import pandas as pd
from nltk.corpus import stopwords 
from nltk.tokenize import word_tokenize 
import nltk,collections
from nltk.util import ngrams
import string
import re
import spacy
import matplotlib as mpl
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
from wordcloud import WordCloud

data = pd.read_csv("../input/question2.csv")
print(data['transcript'].head())

stop_words = set(stopwords.words('english')) 
concatinated_data = data.to_string(header=False,index=False,index_names=False)
word_tokens = word_tokenize(concatinated_data) 


# I have created this function "**cleanDataForNLP**"  .  I am doing below mentioned things in this function
# 1.  **Convert data to lowercase**
# 2.  **Remove stopwords**
# 3.  **Remove words having length less then 3**
# 4.  **Remove punctuation marks as these are of no use to us**
# 5.  **Also removing numeric characters**

# In[ ]:


def cleanDataForNLP(TextData):
    TextData.lower()
    TextData = re.sub('[^A-Za-z]+', ' ', TextData)    
    word_tokens = word_tokenize(TextData)
    filteredText = ""
    for w in word_tokens:
        if w not in stop_words and len(w) > 2 and not w.isnumeric() and w not in string.punctuation:
            filteredText = filteredText + " " + w
    
    return filteredText.strip()


# **I have used n-grams to get context present in dataset.
# The basic point of n-grams is that they capture the language structure from the statistical point of view, like what letter or word is likely to follow the given one. The longer the n-gram (the higher the n), the more context you have to work with.**

# In[ ]:


textData=cleanDataForNLP(concatinated_data)
tokenized = textData.split()
ngram = ngrams(tokenized, 3)
ngramFreq = collections.Counter(ngram)
mostcommonngrams=ngramFreq.most_common(10)
print(mostcommonngrams)


# **Phrase Selection
# Now to get important phrases present in our data, I will first try to tag words using POS tagger and then use chunking on noun words to finally form noun phrases.**
# 

# In[ ]:


nlp = spacy.load('en')
nlpdata=nlp(textData)
posTags=[(x.text, x.pos_) for x in nlpdata if x.pos_ != u'SPACE']
nounPhrases=[np.text for np in nlpdata.noun_chunks]
nounPhrasesFiltered=[]
for nounPhrase in nounPhrases:
    if(len(nounPhrase.split(" "))>1):
        nounPhrasesFiltered.append(nounPhrase)

df = pd.DataFrame({'nounPhrases':nounPhrasesFiltered})
nounPhrasesCount=df['nounPhrases'].value_counts()
print("Extracted Phrases")
print(nounPhrasesCount.head(10))


# Let us do basic word cloud formation to check most occuring words for Dataset Analysis.

# In[ ]:


wordcloud = WordCloud(
                          background_color='white',
                          max_words=200,
                          max_font_size=40, 
                          random_state=42
                         ).generate(str(df['nounPhrases']))

print(wordcloud)
fig = plt.figure(figsize=(10,6))
plt.imshow(wordcloud)
plt.axis('off')
plt.show()


# **Phrase Clustering -**
# **To do clustering on phrases which are extracted from dataset above now we will convert words to vector format using tfidftransformer.
# Then Kmeans clustering is used cluster import words in different 5 clusters.
# Then Kmeans model is used to predict previously extracted phrases.**

# In[ ]:


from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
vectorizer = TfidfVectorizer()
vectorizedText = vectorizer.fit_transform(df['nounPhrases'])
words = vectorizer.get_feature_names()

kmeans = KMeans(n_clusters = 5, n_init = 20, n_jobs = 1) 
kmeans.fit(vectorizedText)
common_words = kmeans.cluster_centers_.argsort()[:,-1:-26:-1]
for num, centroid in enumerate(common_words):
    print(str(num) + ' : ' + ', '.join(words[word] for word in centroid))


# In[ ]:


for index, row in df.iterrows():
    y = vectorizer.transform([row['nounPhrases']])
    print (row['nounPhrases'] + " - " + str(kmeans.predict(y)))

