#!/usr/bin/env python
# coding: utf-8

# # [A] Amazon Fine Food Reviews Analysis
# 
# Data Source: https://www.kaggle.com/snap/amazon-fine-food-reviews
# 
# The Amazon Fine Food Reviews dataset consists of reviews of fine foods from Amazon.
# 
# Number of reviews: 568,454<br>
# Number of users: 256,059<br>
# Number of products: 74,258<br>
# Timespan: Oct 1999 - Oct 2012<br>
# Number of Attributes/Columns in data: 10<br>
# 
# Attribute Information:<br>
# 
# Id<br>
# ProductId - unique identifier for the product<br>
# UserId - unqiue identifier for the user<br>
# ProfileName<br>
# HelpfulnessNumerator - number of users who found the review helpful<br>
# HelpfulnessDenominator - number of users who indicated whether they found the review helpful or not<br>
# Score - rating between 1 and 5<br>
# Time - timestamp for the review<br>
# Summary - brief summary of the review<br>
# Text - text of the review<br>
# <b>Objective:</b>
# Given a review, determine whether the review is positive (Rating of 4 or 5) or negative (rating of 1 or 2).
# 
# 
# [Q] How to determine if a review is positive or negative?
# 
# [Ans] We could use the Score/Rating. A rating of 4 or 5 could be cosnidered a positive review. A review of 1 or 2 could be considered negative. A review of 3 is nuetral and ignored. This is an approximate and proxy way of determining the polarity (positivity/negativity) of a review.

# # [B] Loading the data
# 
# The dataset is available in two forms
# 
# .csv file<br>
# SQLite Database<br>
# In order to load the data, We have used the SQLITE dataset as it easier to query the data and visualise the data efficiently. 
# Here as we only want to get the global sentiment of the recommendations (positive or negative), we will purposefully ignore all Scores equal to 3. If the score id above 3, then the recommendation wil be set to "positive". Otherwise, it will be set to "negative".

# ### Imports

# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')
import warnings
warnings.filterwarnings("ignore")

import sqlite3
import pandas as pd
import numpy as np
import nltk
import string
import matplotlib.pyplot as plt
import seaborn as sn
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import confusion_matrix
from sklearn import metrics
from sklearn.metrics import roc_curve, auc
from nltk.stem.porter import PorterStemmer

import re
# Tutorial about Python regular expressions: https://pymotw.com/2/re/
import string
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.stem.wordnet import WordNetLemmatizer

from gensim.models import Word2Vec
from gensim.models import KeyedVectors
import pickle

from tqdm import tqdm
import os


# In[ ]:


#using the SQLite table to read data.
con = sqlite3.connect('../input/database.sqlite')

#Filetring reviews with only positive or negative reviews i.e not considering score=3
filtered_data = pd.read_sql_query("""
SELECT *
FROM Reviews
Where Score!=3
""", con)

#Replace the score with positive or negative (1,2 - negative and 4,5 - positive) 

def partition(x):
    if x<3:
        return 'negative'
    return 'positive'

actualScore = filtered_data['Score']
positiveNegative = actualScore.map(partition)
filtered_data['Score'] = positiveNegative


# In[ ]:


filtered_data.shape
filtered_data.head()


# # [C] Data Cleaning: Deduplication

# It is observed that the reviews data had many duplicate entries. Hence it was necessary to remove duplicates in order to get unbiased results for the analysis of data.

# In[ ]:


display = pd.read_sql_query("""
SELECT *
FROM Reviews
WHERE SCORE != 3 AND UserId = "AR5J8UI46CURR"
ORDER BY ProductId
""", con)
display


# As can be seen above the same user has multiple reviews of the with the same values for HelpfulnessNumerator, HelpfulnessDenominator, Score, Time, Summary and Text  and on doing analysis it was found that <br>
# <br> 
# ProductId=B000HDOPZG was Loacker Quadratini Vanilla Wafer Cookies, 8.82-Ounce Packages (Pack of 8)<br>
# <br> 
# ProductId=B000HDL1RQ was Loacker Quadratini Lemon Wafer Cookies, 8.82-Ounce Packages (Pack of 8) and so on<br>
# 
# It was inferred after analysis that reviews with same parameters other than ProductId belonged to the same product just having different flavour or quantity. Hence in order to reduce redundancy it was decided to eliminate the rows having same parameters.<br>
# 
# The method used for the same was that we first sort the data according to ProductId and then just keep the first similar product review and delelte the others. for eg. in the above just the review for ProductId=B000HDL1RQ remains. This method ensures that there is only one representative for each product and deduplication without sorting would lead to possibility of different representatives still existing for the same product.

# In[ ]:


#Sorting data according to ProducId in dataframe
sorted_data = filtered_data.sort_values('ProductId', axis=0, ascending = True, inplace = False, kind='quicksort', na_position='last')


# In[ ]:


#Deduplication of entries
final = sorted_data.drop_duplicates(subset = {"UserId", "ProfileName", "Time", "Text"}, keep='first', inplace = False)
final.shape


# In[ ]:


#Checking to see how much % of data still remains
(final['Id'].size*1.0)/(filtered_data['Id'].size*1.0)*100


# <b>Observation</b>:- It was also seen that in two rows given below the value of HelpfulnessNumerator is greater than HelpfulnessDenominator which is not practically possible hence these two rows too are removed from calcualtions

# In[ ]:


display= pd.read_sql_query("""
SELECT *
FROM Reviews
WHERE Score != 3 AND Id=44737 OR Id=64422
ORDER BY ProductID
""", con)

display.head()


# In[ ]:


final = final[final.HelpfulnessNumerator<=final.HelpfulnessDenominator]


# In[ ]:


#How many positive and negative reviews left?
final['Score'].value_counts()
#final.shape


# # [D] Bag of Words

# In[ ]:


#count_vec = CountVectorizer() #in scikit-learn
#final_counts = count_vec.fit_transform(final['Text'].values)


# In[ ]:


#type(final_counts)


# In[ ]:


#final_counts.shape


# # [E] Text Preprocessing

# In the preprocessing phase, we will do following things in order given below :- <br>
# <ol>
#     <li>Begin by removing HTML tags.</li>
#     <li>Remove any punctuations or limited set of special characters like , or . or # etc.</li>
#     <li>Check if the word is made up of english letters and is not alpha-numeric</li>
#     <li>Check to see if the length of the word is greater than 2 (as it was researched that there is no adjective in 2-letters)</li>
#     <li>Convert the word to lowercase</li>
#     <li>Remove Stopwords</li>
#     <li>Finally Snowball Stemming the word (it was obsereved to be better than Porter Stemming</li>
# </ol>
# <br>
# 
# After which we collect the words used to describe positive and negative reviews

# In[ ]:


final['Text'].values[6]


# In[ ]:


#Find sentences having HTML tags
#i=0
#for sentence in final['Text'].values:
#    if(len(re.findall('<.*?>', sentence))):
#        print(i)
#        print(sentence)
#        break
#    i += 1


# In[ ]:


#Functions to clean HTML and Punctuation

import re #https://pymotw.com/2/re
import string
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer

stop = set(stopwords.words('english')) #set of all stopwords 
sno = SnowballStemmer('english') #Initialize stemmer

def cleanhtml(sentence): #Function to clean html
    cleanr = re.compile('<.*?>')
    cleantext = re.sub(cleanr, ' ',sentence)
    return cleantext

def cleanpunctuation(sentence): #Function to clean all punctuation
    cleaned = re.sub(r'[?|!|\'|"|#]', r'',sentence)
    cleaned = re.sub(r'[.|,|)|(|\|/]', r' ',cleaned)
    return cleaned

print(stop)
print(sno.stem('tasty'))


# In[ ]:


#Root word for tasty#Code for implementing step-by-step the checks mentioned in the pre-processing phase
#this code takes a while to run as it needs to run on 500k sentences.
if not os.path.isfile('finals.sqlite'):
    i=0
    str1=' '
    final_string=[]
    all_positive_words=[] # store words from +ve reviews here
    all_negative_words=[] # store words from -ve reviews here.
    s=''
    for sent in tqdm(final['Text'].values):
        filtered_sentence=[]
        #print(sent);
        sent=cleanhtml(sent) # remove HTMl tags
        for w in sent.split():
            for cleaned_words in cleanpunctuation(w).split():
                if((cleaned_words.isalpha()) & (len(cleaned_words)>2)):    
                    if(cleaned_words.lower() not in stop):
                        s=(sno.stem(cleaned_words.lower())).encode('utf8')
                        filtered_sentence.append(s)
                        if (final['Score'].values)[i] == 'positive': 
                            all_positive_words.append(s) #list of all words used to describe positive reviews
                        if(final['Score'].values)[i] == 'negative':
                            all_negative_words.append(s) #list of all words used to describe negative reviews reviews
                    else:
                        continue
                else:
                    continue 
        #print(filtered_sentence)
        str1 = b" ".join(filtered_sentence) #final string of cleaned words
        #print("***********************************************************************")

        final_string.append(str1)
        i+=1

    #############---- storing the data into .sqlite file ------########################
    final['CleanedText']=final_string #adding a column of CleanedText which displays the data after pre-processing of the review 
    final['CleanedText']=final['CleanedText'].str.decode("utf-8")
        # store final table into an SQlLite table for future.
    conn = sqlite3.connect('finals.sqlite')
    c=conn.cursor()
    conn.text_factory = str
    final.to_sql('Reviews', conn,  schema=None, if_exists='replace',                  index=True, index_label=None, chunksize=None, dtype=None)
    conn.close()
    
    
    with open('positive_words.pkl', 'wb') as f:
        pickle.dump(all_positive_words, f)
    with open('negitive_words.pkl', 'wb') as f:
        pickle.dump(all_negative_words, f)

print("cell exec")


# ## Code to load final sqlite table

# In[ ]:


#using the SQLite table to read data.
con = sqlite3.connect('finals.sqlite')

#Filetring reviews with only positive or negative reviews i.e not considering score=3
final = pd.read_sql_query("""
SELECT *
FROM Reviews
""", con)


# In[ ]:


final['Text'].iloc[6]


# In[ ]:


final['CleanedText'].iloc[6]


# # [F] Bi-grams and n-grams (Bag of Words)

#  **Motivation**
# 
# Now that we have our list of words describing positive and negative reviews lets analyse them.<br>
# 
# We begin analysis by getting the frequency distribution of the words as shown below

# In[ ]:


#freq_dist_positive = nltk.FreqDist(all_positive_words)
#freq_dist_negative = nltk.FreqDist(all_negative_words)


# In[ ]:


#print("Most Common Positive Words : ",freq_dist_positive.most_common(20))
#print("Most Common Negative Words : ",freq_dist_negative.most_common(20))


# <b>Observation:-</b> From the above it can be seen that the most common positive and the negative words overlap for eg. 'like' could be used as 'not like' etc. <br>
# So, it is a good idea to consider pairs of consequent words (bi-grams) or q sequnce of n consecutive words (n-grams)

# In[ ]:


#bi-gram, tri-gram and n-gram

#removing stop words like "not" should be avoided before building n-grams
count_vect = CountVectorizer() #in scikit-learn
final_unigram_counts = count_vect.fit_transform(final['CleanedText'].values)
#print("the type of count vectorizer ",type(final_bigram_counts))
print("the shape text BOW vectorizer ",final_unigram_counts.get_shape())
#print("the number of unique words including both unigrams and bigrams ", final_bigram_counts.get_shape()[1])


# In[ ]:




