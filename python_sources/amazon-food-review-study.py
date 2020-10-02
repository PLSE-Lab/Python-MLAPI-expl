#!/usr/bin/env python
# coding: utf-8

# # Amazon Food Review Study

# ## Importing basic modules

# In[ ]:


import numpy as np
import pandas as pd
import sqlite3
import string
import matplotlib.pyplot as plt
import seaborn as sn


# ## Reading Data

# In[ ]:


#Using sqlite3 to read the data
con = sqlite3.connect('../input/database.sqlite')


# In[ ]:


#Filtering positive(5 & 4 stars) and negative(1 & 2 stars) reviews and discarding 3 star reviews.
filtered_data = pd.read_sql_query("""
SELECT *
FROM Reviews
WHERE Score != 3
""", con)


# In[ ]:


#Give reviews greater than 3 positive and less than 3 as negative
def partition(x):
    if x < 3:
        return 'negative'
    return 'positive'


# In[ ]:


#Changing the reviews based on the star value
actualScore = filtered_data["Score"]
positiveNegative = actualScore.map(partition)
filtered_data['Score'] = positiveNegative


# In[ ]:


filtered_data.shape


# In[ ]:


filtered_data.head()


# ## Cleaning Data

# #### There can be so many duplicte values in the data that we get. So its extremely important to remove the duplicates. If we feed garbage data we get garbage results. Example garbage data is shown below.

# In[ ]:


display = pd.read_sql_query("""
SELECT *
FROM Reviews
WHERE Score != 3 AND UserId = "AR5J8UI46CURR"
ORDER BY ProductID
""", con)
display


# ### Now lets remove the duplicate values

# In[ ]:


#Sorting data according to ProductId in ascending order
sorted_data = filtered_data.sort_values('ProductId', axis=0, ascending=True)


# In[ ]:


#Deduplication of entries
final = sorted_data.drop_duplicates(subset={"UserId", "ProfileName", "Time", "Text"}, keep="first", inplace=False)


# In[ ]:


final.shape #If we comare this with the size of input data, we can see that there is a significat reduction. Size of the read data was final.shape #If we comare this with the size of input data, we can see that there is a significat reduction. Size of the read data was (525814, 10) 


# In[ ]:


#Lets check how much data is left after removing duplicates
(final['Id'].size*1.0)/(filtered_data['Id'].size*1.0)*100


# In[ ]:


#In reviews the helpful numerator has to be greater that helpful denominator. But there are some reviews that has a probelm with this
display = pd.read_sql_query("""
SELECT *
FROM Reviews
WHERE Score != 3 AND Id = 44737 OR Id = 64422
ORDER BY ProductID
""", con)
display


# In[ ]:


final = final[final.HelpfulnessNumerator <= final.HelpfulnessDenominator]


# In[ ]:


#We have this many reviews left
final.shape


# In[ ]:


#Number of positive and negative reviews in dataset
final['Score'].value_counts()


# ## Bag of Words(BoW)

# In[ ]:


import scipy
from sklearn.feature_extraction.text import CountVectorizer
count_vect = CountVectorizer()


# In[ ]:


final_counts = count_vect.fit_transform(final['Text'].values)


# In[ ]:


type(final_counts) #Here the final count is a sparse matrix


# In[ ]:


final_counts.get_shape()


# In[ ]:


import re
import string
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.stem import SnowballStemmer
from nltk.stem.wordnet import WordNetLemmatizer

from gensim.models import Word2Vec
from gensim.models import KeyedVectors
import pickle

from tqdm import tqdm
import os


# In[ ]:


stop = set(stopwords.words('english'))
sno = SnowballStemmer('english')


# In[ ]:


print(stop)


# In[ ]:


print(sno.stem('tasty'))


# In[ ]:


print(sno.stem('worked'))


# In[ ]:


def cleanHtml(sentence):
    cleanr = re.compile('<.?>')
    cleartext = re.sub(cleanr, ' ', sentence)
    return cleartext
def cleanPunc(sentence):
    cleaned = re.sub(r'[?|!|\'|\"|#]', r'',sentence)
    cleaned = re.sub(r'[.|,|)|(|\|/]', r'',cleaned)
    return cleaned


# In[ ]:


#removing html and punctuations
i = 0
str1 = ' '
final_string = []
all_positive_words=[] #store positive reviews
all_negative_words=[] #store negative reviews
s = ''

for sent in final['Text'].values:
    filtered_sentence=[]
    sent = cleanHtml(sent)
    for w in sent.split():
        for cleaned_word in cleanPunc(w).split():
            if((cleaned_word.isalpha()) & (len(cleaned_word)>2)):
                if(cleaned_word.lower() not in stop):
                    s=(sno.stem(cleaned_word.lower())).encode('utf8')
                    filtered_sentence.append(s)
                    if(final['Score'].values)[i] == 'positive':
                        all_positive_words.append(s)
                    if(final['Score'].values)[i] == 'positive':
                        all_negative_words.append(s)
                else:
                    continue
            else:
                continue
    str1=b" ".join(filtered_sentence)
    final_string.append(str1)
    i+=1


# In[ ]:


final['CleanedText'] = final_string


# In[ ]:


final.head(3)


# In[ ]:


conn = sqlite3.connect('final.sqlite')
c=conn.cursor()
conn.text_factory = str
final.to_sql('Reviews', conn, schema=None, if_exists='replace')


# In[ ]:




