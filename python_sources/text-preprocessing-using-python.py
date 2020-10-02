#!/usr/bin/env python
# coding: utf-8

# We cannot work with the text data in machine learning so we need to convert them into numerical vectors, This kernel has all techniques for conversion

# > Text data needs to be cleaned and encoded to numerical values before giving them to machine learning models, this process of cleaning and encoding is called as **Text Preprocessing**

# In this kernel we are going to see some basic text cleaning steps and techniques for encoding text data. We are going ot see about
# 1. **Understanding the data** - See what's data is all about. what should be considered for cleaning for data (Punctuations , stopwords etc..).
# 2. **Basic Cleaning** -We will see what parameters need to be considered for cleaning of data (like Punctuations , stopwords etc..)  and its code.
# 3. **Techniques for Encoding** - All the popular techniques that are used for encoding that I personally came across.
#     *           **Bag of Words**
#     *           **Binary Bag of Words**
#     *           **Bigram, Ngram**
#     *           **TF-IDF**( **T**erm  **F**requency - **I**nverse **D**ocument **F**requency)
#     *           **Word2Vec**
#     *           **Avg-Word2Vec**
#     *           **TF-IDF Word2Vec**
# 
# Now, it's time to have fun!

#  **Importing Libraries**

# In[ ]:


import warnings
warnings.filterwarnings("ignore")                     #Ignoring unnecessory warnings

import numpy as np                                  #for large and multi-dimensional arrays
import pandas as pd                                 #for data manipulation and analysis
import nltk                                         #Natural language processing tool-kit

from nltk.corpus import stopwords                   #Stopwords corpus
from nltk.stem import PorterStemmer                 # Stemmer

from sklearn.feature_extraction.text import CountVectorizer          #For Bag of words
from sklearn.feature_extraction.text import TfidfVectorizer          #For TF-IDF
from gensim.models import Word2Vec                                   #For Word2Vec


# In[ ]:


data_path = "../input/Reviews.csv"
data = pd.read_csv(data_path)
data_sel = data.head(10000)                                #Considering only top 10000 rows


# In[ ]:


# Shape of our data
data_sel.columns


# 1. **Understanding the data**
# 
# Looks like our main objective from the dataset is to predict whether a review is **Positive** or **Negative** based on the Text.
#  
#  if we see the Score column, it has values 1,2,3,4,5 .  Considering 1, 2 as Negative reviews and 4, 5 as Positive reviews.
#  For Score = 3 we will consider it as Neutral review and lets delete the rows that are neutral, so that we can predict either Positive or Negative
#  
#  HelfulnessNumerator says about number of people found that review usefull and HelpfulnessDenominator is about usefull review count + not so usefull count.
#  So, from this we can see that HelfulnessNumerator is always less than or equal to HelpfulnesDenominator.

# In[ ]:


data_score_removed = data_sel[data_sel['Score']!=3]       #Neutral reviews removed


# Converting Score values into class label either Posituve or Negative.

# In[ ]:


def partition(x):
    if x < 3:
        return 'positive'
    return 'negative'

score_upd = data_score_removed['Score']
t = score_upd.map(partition)
data_score_removed['Score']=t


# 2. **Basic Cleaning**
#  
# **Deduplication** means removing duplicate rows, It is necessary to remove duplicates in order to get unbaised results. Checking duplicates based on UserId, ProfileName, Time, Text. If all these values are equal then we will remove those records. (No user can type a review on same exact time for different products.)
# 
# 
# We have seen that HelpfulnessNumerator should always be less than or equal to HelpfulnessDenominator so checking this condition and removing those records also.
# 

# In[ ]:


final_data = data_score_removed.drop_duplicates(subset={"UserId","ProfileName","Time","Text"})


# In[ ]:


final = final_data[final_data['HelpfulnessNumerator'] <= final_data['HelpfulnessDenominator']]


# In[ ]:


final_X = final['Text']
final_y = final['Score']


# Converting all words to lowercase and removing punctuations and html tags if any
# 
# **Stemming**- Converting the words into their base word or stem word ( Ex - tastefully, tasty,  these words are converted to stem word called 'tasti'). This reduces the vector dimension because we dont consider all similar words  
# 
# **Stopwords** - Stopwords are the unnecessary words that even if they are removed the sentiment of the sentence dosent change.
# 
# Ex -    This pasta is so tasty ==> pasta tasty    ( This , is, so are stopwords so they are removed)
# 
# To see all the stopwords see the below code cell.

# In[ ]:


stop = set(stopwords.words('english')) 
print(stop)


# In[ ]:


import re
temp =[]
snow = nltk.stem.SnowballStemmer('english')
for sentence in final_X:
    sentence = sentence.lower()                 # Converting to lowercase
    cleanr = re.compile('<.*?>')
    sentence = re.sub(cleanr, ' ', sentence)        #Removing HTML tags
    sentence = re.sub(r'[?|!|\'|"|#]',r'',sentence)
    sentence = re.sub(r'[.|,|)|(|\|/]',r' ',sentence)        #Removing Punctuations
    
    words = [snow.stem(word) for word in sentence.split() if word not in stopwords.words('english')]   # Stemming and removing stopwords
    temp.append(words)
    
final_X = temp    


# In[ ]:


print(final_X[1])


# In[ ]:


sent = []
for row in final_X:
    sequ = ''
    for word in row:
        sequ = sequ + ' ' + word
    sent.append(sequ)

final_X = sent
print(final_X[1])


# 3. **Techniques for Encoding**
# 
#       **BAG OF WORDS**
#       
#       In BoW we construct a dictionary that contains set of all unique words from our text review dataset.The frequency of the word is counted here. if there are **d** unique words in our dictionary then for every sentence or review the vector will be of length **d** and count of word from review is stored at its particular location in vector. The vector will be highly sparse in such case.
#       
#       Ex. pasta is tasty and pasta is good
#       
#      **[0]....[1]............[1]...........[2]..........[2]............[1]..........**             <== Its vector representation ( remaining all dots will be represented as zeroes)
#      
#      **[a]..[and].....[good].......[is].......[pasta]....[tasty].......**            <==This is dictionary
#       .
#       
#     Using scikit-learn's CountVectorizer we can get the BoW and check out all the parameters it consists of, one of them is max_features =5000 it tells about to consider only top 5000 most frequently repeated words to place in a dictionary. so our dictionary length or vector length will be only 5000
#     
# 
# 
#    **BINARY BAG OF WORDS**
#     
#    In binary BoW, we dont count the frequency of word, we just place **1** if the word appears in the review or else **0**. In CountVectorizer there is a parameter **binary = true** this makes our BoW to binary BoW.
#    
#   

# In[ ]:


count_vect = CountVectorizer(max_features=5000)
bow_data = count_vect.fit_transform(final_X)
print(bow_data[1])


#  **Drawbacks of BoW/ Binary BoW**
#  
#  Our main objective in doing these text to vector encodings is that similar meaning text vectors should be close to each other, but in some cases this may not possible for Bow
#  
# For example, if we consider two reviews **This pasta is very tasty** and **This pasta is not tasty** after stopwords removal both sentences will be converted to **pasta tasty** so both giving exact same meaning.
# 
# The main problem is here we are not considering the front and back words related to every word, here comes Bigram and Ngram techniques.

# **BI-GRAM BOW**
# 
# Considering pair of words for creating dictionary is Bi-Gram , Tri-Gram means three consecutive words so as NGram.
# 
# CountVectorizer has a parameter **ngram_range** if assigned to (1,2) it considers Bi-Gram BoW
# 
# But this massively increases our dictionary size 

# In[ ]:


final_B_X = final_X


# In[ ]:


count_vect = CountVectorizer(ngram_range=(1,2))
Bigram_data = count_vect.fit_transform(final_B_X)
print(Bigram_data[1])


# **TF-IDF**
# 
# **Term Frequency -  Inverse Document Frequency** it makes sure that less importance is given to most frequent words and also considers less frequent words.
# 
# **Term Frequency** is number of times a **particular word(W)** occurs in a review divided by totall number of words **(Wr)** in review. The term frequency value ranges from 0 to 1.
# 
# **Inverse Document Frequency** is calculated as **log(Total Number of Docs(N) / Number of Docs which contains particular word(n))**. Here Docs referred as Reviews.
# 
# 
# **TF-IDF** is **TF * IDF** that is **(W/Wr)*LOG(N/n)**
# 
# 
#  Using scikit-learn's tfidfVectorizer we can get the TF-IDF.
# 
# So even here we get a TF-IDF value for every word and in some cases it may consider different meaning reviews as similar after stopwords removal. so to over come we can use BI-Gram or NGram.

# In[ ]:


final_tf = final_X
tf_idf = TfidfVectorizer(max_features=5000)
tf_data = tf_idf.fit_transform(final_tf)
print(tf_data[1])


# so to actually overcome the problem of semantical reviews having close distance we have Word2Vec
# 
# **Word2Vec**
# 
# 
# Word2Vec actually takes the semantic meaning of the words and their relationships between other words. it learns all the internal relationships between the words.It represents the word in dense vector form.
# 
# Using **Gensim's** library we have Word2Vec which takes parameters like **min_count = 5** considers only if word repeats more than 5 times in entire data.
# **size = 50** gives a vector length of size 50 and **workers** are cores to run this.
# 
# 
# **Average Word2Vec**
# 
# Compute the Word2vec of each of the words and add the vectors of each words of the sentence and divide the vector with the number of words of the sentence.Simply Averaging the Word2Vec of all words.

# In[ ]:


w2v_data = final_X


# In[ ]:


splitted = []
for row in w2v_data: 
    splitted.append([word for word in row.split()])     #splitting words


# In[ ]:


train_w2v = Word2Vec(splitted,min_count=5,size=50, workers=4)


# In[ ]:


avg_data = []
for row in splitted:
    vec = np.zeros(50)
    count = 0
    for word in row:
        try:
            vec += train_w2v[word]
            count += 1
        except:
            pass
    avg_data.append(vec/count)
    


# In[ ]:


print(avg_data[1])


# **TF-IDF WORD2VEC**
# 
# in TF-IDF Word2Vec the Word2Vec value of each word is multiplied by the tfidf value of that word and summed up and then divided by the sum of the tfidf values of the sentence.
# 
# Something like  
# 
#                         V = ( t(W1)*w2v(W1) + t(W2)*w2v(W2) +.....+t(Wn)*w2v(Wn))/(t(W1)+t(W2)+....+t(Wn))

# In[ ]:


tf_w_data = final_X
tf_idf = TfidfVectorizer(max_features=5000)
tf_idf_data = tf_idf.fit_transform(tf_w_data)


# In[ ]:


tf_w_data = []
tf_idf_data = tf_idf_data.toarray()
i = 0
for row in splitted:
    vec = [0 for i in range(50)]
    
    temp_tfidf = []
    for val in tf_idf_data[i]:
        if val != 0:
            temp_tfidf.append(val)
    
    count = 0
    tf_idf_sum = 0
    for word in row:
        try:
            count += 1
            tf_idf_sum = tf_idf_sum + temp_tfidf[count-1]
            vec += (temp_tfidf[count-1] * train_w2v[word])
        except:
            pass
    vec = (float)(1/tf_idf_sum) * vec
    tf_w_data.append(vec)
    i = i + 1

print(tf_w_data[1])
    


# **Conclusion**
# 
# Throughout this kernel we have seen different techniques for encoding text data into numerical vectors. But which technique is appropriate for our machine learning model depends on the structure of the data and the objective of our model. 
