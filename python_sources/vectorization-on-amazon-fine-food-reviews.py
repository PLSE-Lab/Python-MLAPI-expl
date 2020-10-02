#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[ ]:


import os
os.getcwd()


# In[ ]:


import sqlite3
con = sqlite3.connect('../input/database.sqlite')


# In[ ]:


import pandas as pd
filtered_data = pd.read_sql_query('''SELECT * FROM Reviews WHERE Score != 3''', con)
type(filtered_data)
filtered_data.head()


# In[ ]:


def posneg(x):
    if x<3:
        return "Negative"
    return "Positive"
actual_score = filtered_data['Score']
PosNeg = list(map(posneg, actual_score))
filtered_data['Score'] = PosNeg
filtered_data.shape


# In[ ]:


filtered_data.head()


# In[ ]:


filtered_data[filtered_data.isnull().any(axis = 1)]


# In[ ]:


dup_data = pd.read_sql_query('''SELECT * FROM Reviews WHERE Score != 3 AND UserId = 'AR5J8UI46CURR' ORDER BY ProductId''', con)
dup_data


# In[ ]:


filtered = filtered_data.sort_values(by = 'ProductId', ascending = True, axis = 0, inplace = False)
filtered.head()


# In[ ]:


filtered.tail()
filtered.shape


# In[ ]:


final_data= filtered_data.drop_duplicates(subset = {'UserId', 'ProfileName', 'Time', 'Text'}, keep = 'first', inplace = False)
final_data.shape


# In[ ]:


(final_data['Id'].size/filtered_data['Id'].size) * 100


# In[ ]:


final_data.head()


# In[ ]:


final_data[final_data.HelpfulnessNumerator > final_data.HelpfulnessDenominator]


# Need to remove the above two rows where Helpfulness Numerator > helpfulness Denominator

# In[ ]:


final = final_data[final_data.HelpfulnessNumerator <= final_data.HelpfulnessDenominator]
final.shape


# In[ ]:


final['Score'].value_counts()


# Data Mining
# 1. Remove HTML tags 

# In[ ]:


import re
i = 0
count = 0
for sent in final['Text'].values:
    if(re.findall('<.*?>', sent)):
        print(i)
        print(sent)
        break
    i +=1
           
   
    


# In[ ]:


def cleanHtml(sentence):
    cleaner = re.compile('<.*?>')
    clean_html = re.sub(cleaner, ' ', sentence)
    return clean_html
def cleanPunc(sentence):
    cleaned = re.sub(r'[?|!|\\]',r'',sentence)
    cleaned = re.sub(r'[.|,|)|(|\\|/]',r' ',cleaned)
    return cleaned 
import string
import nltk                     
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.stem.wordnet import WordNetLemmatizer
stop = set(stopwords.words('english'))
sno = nltk.stem.SnowballStemmer('english')
print(stop)
print(sno.stem('tasty'))
                     


# Code for implementing step-by-step the checks mentioned in the pre-processing phase

# In[ ]:


i = 0
str1 = ' '
final_string = []
all_positive_words = []
all_negative_words = []
s = ''
for sent in final['Text'].values:
    filtered_sentence = []
    sent = cleanHtml(sent)
    for w in sent.split():
        for cleaned_words in cleanPunc(w).split():
            if ((cleaned_words.isalpha()) & (len(cleaned_words) > 2)):
                if (cleaned_words.lower() not in stop):
                    s = (sno.stem(cleaned_words.lower())).encode('utf8')
                    filtered_sentence.append(s)
                    if(final['Score'].values)[i] == 'Positive':
                        all_positive_words.append(s)
                    if(final['Score'].values)[i] == 'Negative': 
                        all_negative_words.append(s)
                else:
                    continue
            else:
                continue
    str1 = b" ".join(filtered_sentence)
    final_string.append(str1)
    i += 1


# In[ ]:


final['Cleaned_text'] = final_string
final['Cleaned_text'] = final['Cleaned_text'].str.decode('utf-8')
final.head(3)


# In[ ]:


final['Cleaned_text'].values


# Vectorization Techniques:
# 1. Bag Of Words(BOW)
#    --> Create an instance for CountVectorizer
#    --> Fit will learn Vocabulay to fit  from unique word corpus of all the review data
#    --> Trnasform will create a Sparse matrix by counting tokens from raw documents fitted to the fot function 

# In[ ]:


from sklearn.feature_extraction.text import CountVectorizer
count_vect = CountVectorizer()
final_counts1 = count_vect.fit(final['Cleaned_text'].values)
type(final_counts1)


# In[ ]:


final_counts = count_vect.transform(final['Cleaned_text'].values)
type(final_counts)


# In[ ]:


final_counts.shape


# Number of Unique words = 68,984

# In[ ]:


freq_dist_pos = nltk.FreqDist(all_positive_words)
freq_dist_neg = nltk.FreqDist(all_negative_words)
print("Most Common Positive words", freq_dist_pos.most_common(20))
print("Most Common Negative words", freq_dist_neg.most_common(20))


# Observation:
# Here the most common words for Positve and Negative reviews are 'like', 'good', 'taste'. Actually for negative it should be prefixed with 'not'. But as the text is cleaned by removing Stop words in which 'not' is one of the words. So to overcome this, n-grams can be used where the sequential text meaning is considered as important.

# In[ ]:


print(stop)
stop.remove('not')
print('*' * 50)
print(stop)
#stop1.remove('not')
i = 0
str2 = ' '
final_string1 = []
all_positive_words1 = []
all_negative_words1 = []
s1 = ''
for sent1 in final['Text'].values:
    filtered_sentence1 = []
    sent1 = cleanHtml(sent1)
    for w1 in sent1.split():
        for cleaned_words1 in cleanPunc(w1).split():
            if ((cleaned_words1.isalpha()) & (len(cleaned_words1) > 2)):
                if (cleaned_words1.lower() not in stop):
                    s1 = (sno.stem(cleaned_words1.lower())).encode('utf8')
                    filtered_sentence1.append(s1)
                    if(final['Score'].values)[i] == 'Positive':
                        all_positive_words1.append(s1)
                    if(final['Score'].values)[i] == 'Negative': 
                        all_negative_words1.append(s1)
                else:
                    continue
            else:
                continue
    str2 = b" ".join(filtered_sentence1)
    final_string1.append(str2)
    i += 1    


# In[ ]:


final['cleaned_not'] = final_string1
final['cleaned_not'] = final['cleaned_not'].str.decode('utf-8')
print(final.shape)
print(final['cleaned_not'].values)


# In[ ]:


freq_dist_pos1 = nltk.FreqDist(all_positive_words1)
freq_dist_neg1 = nltk.FreqDist(all_negative_words1)
print("Most Common Positive words with 'not' stopwrd", freq_dist_pos1.most_common(20))
print("Most Common Negative words with 'not' stopwrd", freq_dist_neg1.most_common(20))


# Observation: 'not' can be seen as most common word in both Positive and Negative words. Probably positve review may contain words like 'not sure' .

# In[ ]:


# BiGrams
count_vect = CountVectorizer(ngram_range = (1,2))
final_counts_bigrams = count_vect.fit_transform(final['cleaned_not'].values)
print(final_counts_bigrams.shape)
print(type(final_counts_bigrams))


# Note: Using bigrams number of features has increased to 2816967.

# TFIDF Vectorizer 
# 1. TF increases if a word is present more frequemtly in a review/text. It repersents the probability of occurence of word in a review.
# 2. IDF increases if a word is most rarely used in the corpus of reviews.

# In[ ]:


from sklearn.feature_extraction.text import TfidfVectorizer
tfidf_count = TfidfVectorizer(ngram_range = (1,2))
final_counts_tfidf = tfidf_count.fit_transform(final['cleaned_not'].values)
print(final_counts_tfidf.shape)
print(type(final_counts_tfidf))


# In[ ]:


# To get feature names
features = tfidf_count.get_feature_names()
print(len(features))
print(features[100000:100010])


# In[ ]:





# **Word2vec**
# It will convert each word in to a 300 dimensional vector.

# In[ ]:


# Train your own Word2Vec model using your own text corpus
import gensim
from gensim.models import Word2Vec
from gensim.models import KeyedVectors
list_of_sent=[]
for sent in final['Cleaned_text'].values:
    list_of_sent.append(sent.split())
print(final['Cleaned_text'].values[0])
print("*****************************************************************")
print(list_of_sent[0])


# In[ ]:


# min_count = 5 considers only words that occured atleast 5 times
W2V_model = Word2Vec(list_of_sent, min_count = 5, size = 50, workers = 4)
w2v_words = list(W2V_model.wv.vocab)
print("Number of words that occured minimum 5 times", len(w2v_words))
print("Sample words", w2v_words[0:25])


# In[ ]:


W2V_model.wv.most_similar('tasti')


# In[ ]:





# In[ ]:


count_vect_feat = count_vect.get_feature_names()
print(count_vect_feat[count_vect_feat.index('like')])


# **Avg W2V***
# 1. Calculate vector for each word in the sentence and sum up all the word vectors of a sentence and divide it by number of words for that sentence present in word corpus.

# In[ ]:


import numpy as np
sent_vectors = []
for sent in list_of_sent:
    sent_vec = np.zeros(50)
    cnt_words = 0
    for word in sent:
        if word in w2v_words:
            vec = W2V_model.wv[word]
            sent_vec += vec
            cnt_words +=1
    if cnt_words != 0:
        sent_vec /= cnt_words
    sent_vectors.append(sent_vec)
print(len(sent_vectors))
print(len(sent_vec))
print(sent_vectors[0])


# **TFIDF Weighted Word2Vec**

# In[ ]:


from sklearn.feature_extraction.text import TfidfVectorizer
tfidf_count1 = TfidfVectorizer()
final_counts1_tfidf = tfidf_count1.fit_transform(final['Cleaned_text'].values)
print(final_counts1_tfidf.shape)                                   


# In[ ]:


import numpy as np
tfidf_feat = tfidf_count1.get_feature_names()
type(tfidf_feat)
tfidf_sent_vectors = []
row = 0
for sent in list_of_sent[:100]:
    sent_vec = np.zeros(50)
    weighted_sum = 0
    for word in sent:
        if word in w2v_words:
            vec = W2V_model.wv[word]
            tf_idf = final_counts1_tfidf[row, tfidf_feat.index(word)]
            sent_vec = sent_vec + (vec * tf_idf)
            weighted_sum += tf_idf
    if weighted_sum != 0:
        sent_vec /= weighted_sum
    tfidf_sent_vectors.append(sent_vec)
    row += 1
print(len(tfidf_sent_vectors))
print(len(sent_vec))
print(tfidf_sent_vectors[0])  

