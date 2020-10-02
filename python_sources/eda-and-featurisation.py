#!/usr/bin/env python
# coding: utf-8

# IMPORT LIBRARIES

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
import seaborn as sns
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


# LETS HAVE A LOOK AT THE DATASET

# In[ ]:


df=pd.read_csv("/kaggle/input/amazon-fine-food-reviews/Reviews.csv")
df.head()


# In[ ]:


df.info()


# In[ ]:


df.describe()


# In[ ]:


df.Score.value_counts()


# WE HAVE MANY REVIEWS WITH SCORE 5

# In[ ]:


sns.countplot(x="Score",data=df,palette="RdBu")
plt.xlabel("Score")
plt.ylabel("Count")


# In[ ]:


df.columns


# DROPPING ID COLUMN, AS IT IS OF NO USE
# SIMILARLY CAN DROP OTHER COLUMNS

# In[ ]:


df.drop("Id",axis=1)


# CREATING NEW DATAFRAME
# ADDING A COLUMN CLASSIFY THAT CLASSIFIES SCORE AS POSITIVE ,NEGATIVE OR NEUTRAL

# In[ ]:


new_df=df[['Id', 'ProductId', 'UserId', 'ProfileName', 'HelpfulnessNumerator',
       'HelpfulnessDenominator', 'Score', 'Time', 'Summary', 'Text']].copy()


# In[ ]:


new_df["Classify"]=new_df["Score"].apply(lambda score: "Pos" if score >3 else "Neg" if score<3 else "Neutral")


# In[ ]:


new_df.head()


# NOW ADDING COLUMN USEFULNESS TO SEE HOW MANY REVIEWS ARE USEFUL

# In[ ]:


new_df["Usefulness"] = (new_df["HelpfulnessNumerator"]/new_df["HelpfulnessDenominator"]).apply(lambda n: ">75%" if n > 0.75 else ("<25%" if n < 0.25 else ("25-75%" if n >= 0.25 and                                                                        n <= 0.75 else "useless")))


# In[ ]:


new_df.head()


# WE HAVE MANY POSITIVE REVIEWS

# In[ ]:


sns.countplot(x='Classify', order=["Pos", "Neg"], data=new_df, palette='RdBu')
plt.xlabel('Classification')
plt.show()


# In[ ]:


new_df.Classify.value_counts()


# NEXT WE MAKE A WORD CLOUD OF POPULAR WORDS IN BOTH POSTIVE AND NEGATIVE REVIEW

# Popular Words in a review

# In[ ]:


pos=new_df.loc[new_df["Classify"]=="Pos"]
neg=new_df.loc[new_df["Classify"]=="Neg"]


# In[ ]:


from wordcloud import WordCloud
def create_Word_Corpus(temp):
    words_corpus = ''
    for val in temp["Summary"]:
        text = str(val).lower()
        tokens = nltk.word_tokenize(text)
        tokens = [word for word in tokens if word not in stopwords.words('english')]
        for words in tokens:
            words_corpus = words_corpus + words + ' '
    return words_corpus
        
# Generate a word cloud image
pos_wordcloud = WordCloud(width=900, height=500).generate(create_Word_Corpus(pos))
neg_wordcloud = WordCloud(width=900, height=500).generate(create_Word_Corpus(neg))


# In[ ]:


def plot_Cloud(wordCloud):
    plt.figure( figsize=(20,10), facecolor='w')
    plt.imshow(wordCloud)
    plt.axis("off")
    plt.tight_layout(pad=0)
    plt.show()


# In[ ]:


plot_Cloud(pos_wordcloud)


# In[ ]:


plot_Cloud(pos_wordcloud)


# NEXT WE SEE MANY REVIEWS ARE USELESS

# In[ ]:


new_df.Usefulness.value_counts()


# In[ ]:


sns.countplot(x='Usefulness', order=['useless', '>75%', '25-75%', '<25%'], data=new_df, palette='RdBu')
plt.xlabel('Usefulness')


# In[ ]:


new_df[new_df.Score==5].Usefulness.value_counts()


# In[ ]:


new_df[new_df.Score==1].Usefulness.value_counts()


# MANY POSITIVE REVIEWS ARE USEFUL

# In[ ]:


sns.countplot(x='Classify', hue='Usefulness', order=["Pos", "Neg"],               hue_order=['>75%', '25-75%', '<25%'], data=new_df, palette='RdBu')
plt.xlabel('Classification')


# NEXT WE LOOK FOR FREQUENT REVIEWS OF USERS

# In[ ]:


x = new_df.UserId.value_counts()
x.to_dict()
new_df["reviewer_freq"] = new_df["UserId"].apply(lambda counts: "Frequent (>50 reviews)" if x[counts]>50 else "Not Frequent (1-50)")
ax = sns.countplot(x='Score', hue='reviewer_freq', data=new_df, palette='RdBu')
ax.set_xlabel('Score (Rating)')


# AROUND 1000 FREQUENT REVIEWS ARE USELESS

# In[ ]:


sns.countplot(x='Usefulness', order=['useless', '>75%', '25-75%', '<25%'],               hue='reviewer_freq', data=new_df, palette='RdBu')
plt.xlabel('Helpfulness')


# # FEATURISING i.e. CONVERTING TEXT TO VECTOR
#  (Reading data using sqlite)

# FILTERING POSITIVE AND NEGATIVE REVIEWS

# In[ ]:


con = sqlite3.connect('/kaggle/input/amazon-fine-food-reviews/database.sqlite')
filtered_data = pd.read_sql_query(""" SELECT * FROM Reviews WHERE Score != 3""", con)

def partition(x):
    if x < 3:
        return "negative"
    return "positive"

#changing reviews with score less than 3 to be positive and vice-versa
actualScore = filtered_data['Score']
positiveNegative = actualScore.map(partition) 
filtered_data['Score'] = positiveNegative
print("Number of data points in our data", filtered_data.shape)
filtered_data.head(3)


# In[ ]:


display = pd.read_sql_query("""
SELECT UserId, ProductId, ProfileName, Time, Score, Text, COUNT(*)
FROM Reviews
GROUP BY UserId
HAVING COUNT(*)>1
""", con)


# In[ ]:


print(display.shape)
display.head()


# In[ ]:


display['COUNT(*)'].sum()


# NEXT WE SEE DUPLICATE REVIEWS

# In[ ]:


display= pd.read_sql_query("""
SELECT *
FROM Reviews
WHERE Score != 3 AND UserId="AR5J8UI46CURR"
ORDER BY ProductID
""", con)
display.head()


# SORTING AND THEN REMOVING DUPLICATES

# In[ ]:


#Sorting data according to ProductId in ascending order
sorted_data=filtered_data.sort_values('ProductId', axis=0, ascending=True, inplace=False, kind='quicksort', na_position='last')


# In[ ]:


#Deduplication of entries
final=sorted_data.drop_duplicates(subset={"UserId","ProfileName","Time","Text"}, keep='first', inplace=False)
final.shape


# In[ ]:


#Checking to see how much % of data still remains
(final['Id'].size*1.0)/(filtered_data['Id'].size*1.0)*100


# In[ ]:


display= pd.read_sql_query("""
SELECT *
FROM Reviews
WHERE Score != 3 AND Id=44737 OR Id=64422
ORDER BY ProductID
""", con)

display.head()


# REMOVING REVIEWS WHERE NUMERATOR<= DENOMINATOR AS IT DOES NOT MAKES SENSE

# In[ ]:


final=final[final.HelpfulnessNumerator<=final.HelpfulnessDenominator]


# In[ ]:


#Before starting the next phase of preprocessing lets see the number of entries left
print(final.shape)

#How many positive and negative reviews are present in our dataset?
final['Score'].value_counts()


# PRE PROCESSING

# 1. REMOVING STOPWORDS
# 2. DECONTRACTING SHORT FORMS
# 3. REMOVING URLS, TAGS, PUNCTUATION, ETC USING REGULAR EXPRESSION

# In[ ]:


stopwords= set(['br', 'the', 'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're", "you've",            "you'll", "you'd", 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself',             'she', "she's", 'her', 'hers', 'herself', 'it', "it's", 'its', 'itself', 'they', 'them', 'their',            'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', "that'll", 'these', 'those',             'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does',             'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of',             'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through', 'during', 'before', 'after',            'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further',            'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more',            'most', 'other', 'some', 'such', 'only', 'own', 'same', 'so', 'than', 'too', 'very',             's', 't', 'can', 'will', 'just', 'don', "don't", 'should', "should've", 'now', 'd', 'll', 'm', 'o', 're',             've', 'y', 'ain', 'aren', "aren't", 'couldn', "couldn't", 'didn', "didn't", 'doesn', "doesn't", 'hadn',            "hadn't", 'hasn', "hasn't", 'haven', "haven't", 'isn', "isn't", 'ma', 'mightn', "mightn't", 'mustn',            "mustn't", 'needn', "needn't", 'shan', "shan't", 'shouldn', "shouldn't", 'wasn', "wasn't", 'weren', "weren't",             'won', "won't", 'wouldn', "wouldn't"])

def decontracted(phrase):
    # general
    phrase = re.sub(r"n\'t", " not", phrase)
    phrase = re.sub(r"\'re", " are", phrase)
    phrase = re.sub(r"\'s", " is", phrase)
    phrase = re.sub(r"\'d", " would", phrase)
    phrase = re.sub(r"\'ll", " will", phrase)
    phrase = re.sub(r"\'t", " not", phrase)
    phrase = re.sub(r"\'ve", " have", phrase)
    phrase = re.sub(r"\'m", " am", phrase)
    return phrase

from bs4 import BeautifulSoup

preprocessed_reviews=[]
for sentance in final['Text'].values:
    sentance=re.sub(r"http\S+","",sentance)
    sentance=BeautifulSoup(sentance,'lxml').get_text()
    sentance=decontracted(sentance)
    sentance=re.sub("\S*\d\S*","",sentance).strip()
    sentance=re.sub("[^A-Za-z]+"," ",sentance)
    sentance = ' '.join(e.lower() for e in sentance.split() if e.lower() not in stopwords)
    preprocessed_reviews.append(sentance.strip())
    


# FEATURISATION
# 1. BAG OF WORDS

# In[ ]:


#BoW
count_vect = CountVectorizer() #in scikit-learn
count_vect.fit(preprocessed_reviews)
print("some feature names ", count_vect.get_feature_names()[:10])
print('='*50)

final_counts = count_vect.transform(preprocessed_reviews)
print("the type of count vectorizer ",type(final_counts))
print("the shape of out text BOW vectorizer ",final_counts.get_shape())
print("the number of unique words ", final_counts.get_shape()[1])


# 2. NGRAMS

# In[ ]:


#bi/tri/n-grams
count_vect = CountVectorizer(ngram_range=(1,2), min_df=10, max_features=5000)
final_bigram_counts = count_vect.fit_transform(preprocessed_reviews)
print("the type of count vectorizer ",type(final_bigram_counts))
print("the shape of out text BOW vectorizer ",final_bigram_counts.get_shape())
print("the number of unique words including both unigrams and bigrams ", final_bigram_counts.get_shape()[1])


# 3. TF IDF

# In[ ]:


#tf-idf
tf_idf_vect = TfidfVectorizer(ngram_range=(1,2), min_df=10)
tf_idf_vect.fit(preprocessed_reviews)

final_tf_idf = tf_idf_vect.transform(preprocessed_reviews)
print("the type of count vectorizer ",type(final_tf_idf))
print("the shape of out text TFIDF vectorizer ",final_tf_idf.get_shape())
print("the number of unique words including both unigrams and bigrams ", final_tf_idf.get_shape()[1])


# Converting text into vector

# AVERAGE WORD2VEC

# HERE GOOGLE'S TRAINED WORD2VEC MODEL IS USED
# IT TAKES MANY HOURS AND 16GB OF RAM

# In[ ]:


i=0
list_of_sentance=[]
for sentance in preprocessed_reviews:
    list_of_sentance.append(sentance.split())
w2v_model=KeyedVectors.load_word2vec_format('/kaggle/input/googles-trained-word2vec-model-in-python/GoogleNews-vectors-negative300.bin', binary=True)
w2v_words = list(w2v_model.wv.vocab)
sent_vectors = []; # the avg-w2v for each sentence/review is stored in this list
for sent in list_of_sentance: # for each review/sentence
    sent_vec = np.zeros(300) # as word vectors are of zero length 50, you might need to change this to 300 if you use google's w2v
    cnt_words =0; # num of words with a valid vector in the sentence/review
    for word in sent: # for each word in a review/sentence
        if word in w2v_words:
            vec = w2v_model.wv[word]
            sent_vec += vec
            cnt_words += 1
    if cnt_words != 0:
        sent_vec /= cnt_words
    sent_vectors.append(sent_vec)
print ("Done")


# TF IDF WORD TO VEC

# In[ ]:


# S = ["abc def pqr", "def def def abc", "pqr pqr def"]
model = TfidfVectorizer()
model.fit(preprocessed_reviews)
# we are converting a dictionary with word as a key, and the idf as a value
dictionary = dict(zip(model.get_feature_names(), list(model.idf_)))



# TF-IDF weighted Word2Vec
tfidf_feat = model.get_feature_names() # tfidf words/col-names
# final_tf_idf is the sparse matrix with row= sentence, col=word and cell_val = tfidf

tfidf_sent_vectors = []; # the tfidf-w2v for each sentence/review is stored in this list
row=0;
for sent in tqdm(list_of_sentance): # for each review/sentence 
    sent_vec = np.zeros(300) # as word vectors are of zero length
    weight_sum =0; # num of words with a valid vector in the sentence/review
    for word in sent: # for each word in a review/sentence
        if word in w2v_words and word in tfidf_feat:
            vec = w2v_model.wv[word]
#             tf_idf = tf_idf_matrix[row, tfidf_feat.index(word)]
            # to reduce the computation we are 
            # dictionary[word] = idf value of word in whole courpus
            # sent.count(word) = tf valeus of word in this review
            tf_idf = dictionary[word]*(sent.count(word)/len(sent))
            sent_vec += (vec * tf_idf)
            weight_sum += tf_idf
    if weight_sum != 0:
        sent_vec /= weight_sum
    tfidf_sent_vectors.append(sent_vec)
    row += 1
print("Done")


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




