#!/usr/bin/env python
# coding: utf-8

# # Amazon Fine Food Reviews Analysis
# 
# 
# Data Source: https://www.kaggle.com/snap/amazon-fine-food-reviews <br>
# 
# EDA: https://nycdatascience.com/blog/student-works/amazon-fine-foods-visualization/
# 
# 
# The Amazon Fine Food Reviews dataset consists of reviews of fine foods from Amazon.<br>
# 
# Number of reviews: 568,454<br>
# Number of users: 256,059<br>
# Number of products: 74,258<br>
# Timespan: Oct 1999 - Oct 2012<br>
# Number of Attributes/Columns in data: 10 
# 
# Attribute Information:
# 
# 1. Id
# 2. ProductId - unique identifier for the product
# 3. UserId - unqiue identifier for the user
# 4. ProfileName
# 5. HelpfulnessNumerator - number of users who found the review helpful
# 6. HelpfulnessDenominator - number of users who indicated whether they found the review helpful or not
# 7. Score - rating between 1 and 5
# 8. Time - timestamp for the review
# 9. Summary - brief summary of the review
# 10. Text - text of the review
# 
# 
# #### Objective:
# Given a review, determine whether the review is positive (Rating of 4 or 5) or negative (rating of 1 or 2).
# 
# <br>
# [Q] How to determine if a review is positive or negative?<br>
# <br> 
# [Ans] We could use the Score/Rating. A rating of 4 or 5 could be cosnidered a positive review. A review of 1 or 2 could be considered negative. A review of 3 is nuetral and ignored. This is an approximate and proxy way of determining the polarity (positivity/negativity) of a review.
# 
# 
# 

# ## Loading the data
# 
# The dataset is available in two forms
# 1. .csv file
# 2. SQLite Database
# 
# In order to load the data, We have used the SQLITE dataset as it easier to query the data and visualise the data efficiently.
# <br> 
# 
# Here as we only want to get the global sentiment of the recommendations (positive or negative), we will purposefully ignore all Scores equal to 3. If the score id above 3, then the recommendation wil be set to "positive". Otherwise, it will be set to "negative".

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


# # [1]. Reading Data

# In[ ]:


os.listdir()


# In[ ]:


con = sqlite3.connect('../input/database.sqlite')


# In[ ]:


#filtering only positive and negative reviews i.e. 
# not taking into consideration those reviews with Score=3
# SELECT * FROM Reviews WHERE Score != 3 LIMIT 500000, will give top 500000 data points
# you can change the number to any other number based on your computing power

# filtered_data = pd.read_sql_query(""" SELECT * FROM Reviews WHERE Score != 3 LIMIT 500000""", con) 
# for tsne assignment you can take 5k data points

filtered_data = pd.read_sql_query(""" SELECT * FROM Reviews WHERE Score != 3 LIMIT 5000""", con) 


# In[ ]:


# Give reviews with Score>3 a positive rating, and reviews with a score<3 a negative rating.
def partition(x):
    if x < 3:
        return 0
    return 1


# In[ ]:


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


display[display['UserId']=='AZY10LLTJ71NX']


# In[ ]:


display['COUNT(*)'].sum()


# #  Exploratory Data Analysis
# 
# ## [2] Data Cleaning: Deduplication
# 
# It is observed (as shown in the table below) that the reviews data had many duplicate entries. Hence it was necessary to remove duplicates in order to get unbiased results for the analysis of the data.  Following is an example:

# In[ ]:


display= pd.read_sql_query("""
SELECT *
FROM Reviews
WHERE Score != 3 AND UserId="AR5J8UI46CURR"
ORDER BY ProductID
""", con)
display.head()


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


#Sorting data according to ProductId in ascending order
sorted_data=filtered_data.sort_values('ProductId', axis=0, ascending=True, inplace=False, kind='quicksort', na_position='last')


# In[ ]:


#Deduplication of entries
final=sorted_data.drop_duplicates(subset={"UserId","ProfileName","Time","Text"}, keep='first', inplace=False)
final.shape


# In[ ]:


#Checking to see how much % of data still remains
(final['Id'].size*1.0)/(filtered_data['Id'].size*1.0)*100


# <b>Observation:-</b> It was also seen that in two rows given below the value of HelpfulnessNumerator is greater than HelpfulnessDenominator which is not practically possible hence these two rows too are removed from calcualtions

# In[ ]:


display= pd.read_sql_query("""
SELECT *
FROM Reviews
WHERE Score != 3 AND Id=44737 OR Id=64422
ORDER BY ProductID
""", con)

display.head()


# In[ ]:


final=final[final.HelpfulnessNumerator<=final.HelpfulnessDenominator]


# In[ ]:


#Before starting the next phase of preprocessing lets see the number of entries left
print(final.shape)

#How many positive and negative reviews are present in our dataset?
final['Score'].value_counts()


# In[ ]:


# Code referred from https://stackoverflow.com/questions/31749448/how-to-add-percentages-on-top-of-bars-in-seaborn
ax = final['Score'].value_counts().plot(kind='bar', 
                                         fontsize=13);
ax.set_alpha(0.8)
ax.set_title("Score class distribution", fontsize=18)
ax.set_ylabel("Count", fontsize=18);
#ax.set_yticks([0, 5, 10, 15, 20])
ax.set_xticklabels(['Positive', 'Negative'], rotation=0, fontsize=11)

# create a list to collect the plt.patches data
totals = []

# find the values and append to list
for i in ax.patches:
    totals.append(i.get_height())

# set individual bar lables using above list
total = sum(totals)

# set individual bar lables using above list
for i in ax.patches:
    # get_x pulls left or right; get_height pushes up or down
    #ax.text(i.get_x()-.03, i.get_height()+.5, \
     #       str(round((i.get_height()/total)*100, 2))+'%', fontsize=15,
      #          color='dimgrey')
      # Decreasing the i.get_x()+.12 will shift the text to left side and decreasing the i.get_height()-14 will bring the text down
    ax.text(i.get_x()+.04, i.get_height()-350,             str(round((i.get_height()/total)*100, 2))+'%', fontsize=20,
                color='white')


# **Observations:** It is evident that the data points that we have selected contains 84% of positive score and 16% of negative score.

# # [3].  Text Preprocessing.
# 
# Now that we have finished deduplication our data requires some preprocessing before we go on further with analysis and making the prediction model.
# 
# Hence in the Preprocessing phase we do the following in the order below:-
# 
# 1. Begin by removing the html tags
# 2. Remove any punctuations or limited set of special characters like , or . or # etc.
# 3. Check if the word is made up of english letters and is not alpha-numeric
# 4. Check to see if the length of the word is greater than 2 (as it was researched that there is no adjective in 2-letters)
# 5. Convert the word to lowercase
# 6. Remove Stopwords
# 7. Finally Snowball Stemming the word (it was obsereved to be better than Porter Stemming)<br>
# 
# After which we collect the words used to describe positive and negative reviews

# In[ ]:


# printing some random reviews
sent_0 = final['Text'].values[0]
print(sent_0)
print("="*50)

sent_1000 = final['Text'].values[1000]
print(sent_1000)
print("="*50)

sent_1500 = final['Text'].values[1500]
print(sent_1500)
print("="*50)

sent_4900 = final['Text'].values[4900]
print(sent_4900)
print("="*50)


# In[ ]:


# remove urls from text python: https://stackoverflow.com/a/40823105/4084039
sent_0 = re.sub(r"http\S+", "", sent_0)
sent_1000 = re.sub(r"http\S+", "", sent_1000)
sent_150 = re.sub(r"http\S+", "", sent_1500)
sent_4900 = re.sub(r"http\S+", "", sent_4900)

print(sent_0)


# In[ ]:


# https://stackoverflow.com/questions/16206380/python-beautifulsoup-how-to-remove-all-tags-from-an-element
from bs4 import BeautifulSoup

soup = BeautifulSoup(sent_0, 'lxml')
text = soup.get_text()
print(text)
print("="*50)

soup = BeautifulSoup(sent_1000, 'lxml')
text = soup.get_text()
print(text)
print("="*50)

soup = BeautifulSoup(sent_1500, 'lxml')
text = soup.get_text()
print(text)
print("="*50)

soup = BeautifulSoup(sent_4900, 'lxml')
text = soup.get_text()
print(text)


# In[ ]:


# https://stackoverflow.com/a/47091490/4084039
import re

def decontracted(phrase):
    # specific
    phrase = re.sub(r"won't", "will not", phrase)
    phrase = re.sub(r"can\'t", "can not", phrase)

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


# In[ ]:


sent_1500 = decontracted(sent_1500)
print(sent_1500)
print("="*50)


# In[ ]:


#remove words with numbers python: https://stackoverflow.com/a/18082370/4084039
sent_0 = re.sub("\S*\d\S*", "", sent_0).strip()
print(sent_0)


# In[ ]:


#remove spacial character: https://stackoverflow.com/a/5843547/4084039
sent_1500 = re.sub('[^A-Za-z0-9]+', ' ', sent_1500)
print(sent_1500)


# In[ ]:


# https://gist.github.com/sebleier/554280
# we are removing the words from the stop words list: 'no', 'nor', 'not'
# <br /><br /> ==> after the above steps, we are getting "br br"
# we are including them into stop words list
# instead of <br /> if we have <br/> these tags would have revmoved in the 1st step

stopwords= set(['br', 'the', 'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're", "you've",            "you'll", "you'd", 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself',             'she', "she's", 'her', 'hers', 'herself', 'it', "it's", 'its', 'itself', 'they', 'them', 'their',            'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', "that'll", 'these', 'those',             'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does',             'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of',             'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through', 'during', 'before', 'after',            'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further',            'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more',            'most', 'other', 'some', 'such', 'only', 'own', 'same', 'so', 'than', 'too', 'very',             's', 't', 'can', 'will', 'just', 'don', "don't", 'should', "should've", 'now', 'd', 'll', 'm', 'o', 're',             've', 'y', 'ain', 'aren', "aren't", 'couldn', "couldn't", 'didn', "didn't", 'doesn', "doesn't", 'hadn',            "hadn't", 'hasn', "hasn't", 'haven', "haven't", 'isn', "isn't", 'ma', 'mightn', "mightn't", 'mustn',            "mustn't", 'needn', "needn't", 'shan', "shan't", 'shouldn', "shouldn't", 'wasn', "wasn't", 'weren', "weren't",             'won', "won't", 'wouldn', "wouldn't"])


# In[ ]:


# Combining all the above stundents 
from tqdm import tqdm
preprocessed_reviews = []
# tqdm is for printing the status bar
for sentance in tqdm(final['Text'].values):
    sentance = re.sub(r"http\S+", "", sentance)
    sentance = BeautifulSoup(sentance, 'lxml').get_text()
    sentance = decontracted(sentance)
    sentance = re.sub("\S*\d\S*", "", sentance).strip()
    sentance = re.sub('[^A-Za-z]+', ' ', sentance)
    # https://gist.github.com/sebleier/554280
    sentance = ' '.join(e.lower() for e in sentance.split() if e.lower() not in stopwords)
    preprocessed_reviews.append(sentance.strip())


# In[ ]:


preprocessed_reviews[1500]


# <h2><font color='red'>[3.2] Preprocess Summary</font></h2>

# In[ ]:


## Similartly you can do preprocessing for review summary also.
# Combining all the above stundents 
from tqdm import tqdm
preprocessed_summary = []
# tqdm is for printing the status bar
for sentence in tqdm(final['Summary'].values):
    sentence = re.sub(r"http\S+", "", sentence)
    sentence = BeautifulSoup(sentence, 'lxml').get_text()
    sentence = decontracted(sentence)
    sentence = re.sub("\S*\d\S*", "", sentence).strip()
    sentence = re.sub('[^A-Za-z]+', ' ', sentence)
    # https://gist.github.com/sebleier/554280
    sentence = ' '.join(e.lower() for e in sentence.split() if e.lower() not in stopwords)
    preprocessed_summary.append(sentence.strip())


# In[ ]:


preprocessed_summary[150]


# # [4] Featurization

# ## [4.1] BAG OF WORDS

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


# **Note:** The final_counts is a sparse matrix and we will need to convert it to a dense matrix before applying t-sne.

# ## [4.2] Bi-Grams and n-Grams.

# In[ ]:


#bi-gram, tri-gram and n-gram

#removing stop words like "not" should be avoided before building n-grams
# count_vect = CountVectorizer(ngram_range=(1,2))
# please do read the CountVectorizer documentation http://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.CountVectorizer.html
# you can choose these numebrs min_df=10, max_features=5000, of your choice
count_vect = CountVectorizer(ngram_range=(1,2), min_df=10, max_features=5000)
final_bigram_counts = count_vect.fit_transform(preprocessed_reviews)
print("the type of count vectorizer ",type(final_bigram_counts))
print("the shape of out text BOW vectorizer ",final_bigram_counts.get_shape())
print("the number of unique words including both unigrams and bigrams ", final_bigram_counts.get_shape()[1])


# **Note:** The final_bigram_counts is a sparse matrix and we will need to convert it to a dense matrix before applying t-sne.

# ## [4.3] TF-IDF

# In[ ]:


tf_idf_vect = TfidfVectorizer(ngram_range=(1,2), min_df=10)
tf_idf_vect.fit(preprocessed_reviews)
print("some sample features(unique words in the corpus)",tf_idf_vect.get_feature_names()[0:10])
print('='*50)

final_tf_idf = tf_idf_vect.transform(preprocessed_reviews)
print("the type of count vectorizer ",type(final_tf_idf))
print("the shape of out text TFIDF vectorizer ",final_tf_idf.get_shape())
print("the number of unique words including both unigrams and bigrams ", final_tf_idf.get_shape()[1])


# **Note:** The final_tf_idf is a sparse matrix and we will need to convert it to a dense matrix before applying t-sne.

# ## [4.4] Word2Vec

# In[ ]:


# Train your own Word2Vec model using your own text corpus
i=0
list_of_sentance=[]
for sentance in preprocessed_reviews:
    list_of_sentance.append(sentance.split())


# In[ ]:


# Using Google News Word2Vectors

# in this project we are using a pretrained model by google
# its 3.3G file, once you load this into your memory 
# it occupies ~9Gb, so please do this step only if you have >12G of ram
# we will provide a pickle file wich contains a dict , 
# and it contains all our courpus words as keys and  model[word] as values
# To use this code-snippet, download "GoogleNews-vectors-negative300.bin" 
# from https://drive.google.com/file/d/0B7XkCwpI5KDYNlNUTTlSS21pQmM/edit
# it's 1.9GB in size.


# http://kavita-ganesan.com/gensim-word2vec-tutorial-starter-code/#.W17SRFAzZPY
# you can comment this whole cell
# or change these varible according to your need

is_your_ram_gt_16g= False
want_to_use_google_w2v = False
want_to_train_w2v = True

if want_to_train_w2v:
    # min_count = 5 considers only words that occured atleast 5 times
    w2v_model=Word2Vec(list_of_sentance,min_count=5,size=50, workers=4)
    print(w2v_model.wv.most_similar('great'))
    print('='*50)
    print(w2v_model.wv.most_similar('worst'))
    
elif want_to_use_google_w2v and is_your_ram_gt_16g:
    if os.path.isfile('GoogleNews-vectors-negative300.bin'):
        w2v_model=KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin', binary=True)
        print(w2v_model.wv.most_similar('great'))
        print(w2v_model.wv.most_similar('worst'))
    else:
        print("you don't have gogole's word2vec file, keep want_to_train_w2v = True, to train your own w2v ")


# In[ ]:


w2v_words = list(w2v_model.wv.vocab)
print("number of words that occured minimum 5 times ",len(w2v_words))
print("sample words ", w2v_words[0:50])


# ## [4.4.1] Converting text into vectors using wAvg W2V, TFIDF-W2V

# #### [4.4.1.1] Avg W2v

# In[ ]:


# average Word2Vec
# compute average word2vec for each review.
sent_vectors = []; # the avg-w2v for each sentence/review is stored in this list
for sent in tqdm(list_of_sentance): # for each review/sentence
    sent_vec = np.zeros(50) # as word vectors are of zero length 50, you might need to change this to 300 if you use google's w2v
    cnt_words =0; # num of words with a valid vector in the sentence/review
    for word in sent: # for each word in a review/sentence
        if word in w2v_words:
            vec = w2v_model.wv[word]
            sent_vec += vec
            cnt_words += 1
    if cnt_words != 0:
        sent_vec /= cnt_words
    sent_vectors.append(sent_vec)
print(len(sent_vectors))
print(len(sent_vectors[0]))


# In[ ]:


print("the type of count vectorizer ",type(sent_vectors))
print("the shape of out text TFIDF vectorizer ",len(sent_vectors))


# **Note:** The sent_vectors is a dense list and doesn't require conversion to a dense array.

# #### [4.4.1.2] TFIDF weighted W2v

# In[ ]:


# S = ["abc def pqr", "def def def abc", "pqr pqr def"]
model = TfidfVectorizer()
model.fit(preprocessed_reviews)
# we are converting a dictionary with word as a key, and the idf as a value
dictionary = dict(zip(model.get_feature_names(), list(model.idf_)))


# In[ ]:


# TF-IDF weighted Word2Vec
tfidf_feat = model.get_feature_names() # tfidf words/col-names
# final_tf_idf is the sparse matrix with row= sentence, col=word and cell_val = tfidf

tfidf_sent_vectors = []; # the tfidf-w2v for each sentence/review is stored in this list
row=0;
for sent in tqdm(list_of_sentance): # for each review/sentence 
    sent_vec = np.zeros(50) # as word vectors are of zero length
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


# In[ ]:


print("the type of count vectorizer ",type(tfidf_sent_vectors))
print("the shape of out text TFIDF vectorizer ",len(tfidf_sent_vectors))


# **Note:** The tfidf_sent_vectors is a dense list and doesn't require conversion to a dense array.

# # [5] Applying TSNE

# <ol> 
#     <li> you need to plot 4 tsne plots with each of these feature set
#         <ol>
#             <li>Review text, preprocessed one converted into vectors using (BOW)</li>
#             <li>Review text, preprocessed one converted into vectors using (TFIDF)</li>
#             <li>Review text, preprocessed one converted into vectors using (AVG W2v)</li>
#             <li>Review text, preprocessed one converted into vectors using (TFIDF W2v)</li>
#         </ol>
#     </li>
#     <li> <font color='blue'>Note 1: The TSNE accepts only dense matrices</font></li>
#     <li> <font color='blue'>Note 2: Consider only 5k to 6k data points </font></li>
# </ol>

# **NOTE:** From the [paper](http://www.jmlr.org/papers/volume9/vandermaaten08a/vandermaaten08a.pdf) :
# 
# > The perplexity can be interpreted as a smooth measure of the effective number of neighbors. The
# performance of SNE is fairly robust to changes in the perplexity, and typical values are between 5
# and 50.
# 
# Takling reference from https://distill.pub/2016/misread-tsne/ and the paper, I will use use perplexity values : 5, 30, 50, 100

# In[ ]:


from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler


# In[ ]:


# configuring the parameteres
# the number of components = 2
# default perplexity = 30
# default learning rate = 200
# default Maximum number of iterations for the optimization = 1000


# ## [5.1] Applying TNSE on Text BOW vectors

# In[ ]:


# Convert the sparse matrix to a dense matrix
final_counts_dense = final_counts.todense()


# In[ ]:


# Standardize the data
bow_standardized_data = StandardScaler().fit_transform(final_counts_dense)


# In[ ]:


# bow model 1, perplexity = 5, n_iter = 250
# perplexity is the number of points in the neighborhood
# n_iter is the step size

bow_model_1 = TSNE(n_components=2, perplexity=5, n_iter=250, random_state = 507)


# In[ ]:


# fit_transform(raw_documents[, y]): Learn the vocabulary dictionary and return term-document matrix. 
# This is equivalent to fit followed by the transform, but more efficiently implemented.
# reference: https://stackoverflow.com/questions/23838056/what-is-the-difference-between-transform-and-fit-transform-in-sklearn#answer-53032201

bow_data_1 = bow_model_1.fit_transform(bow_standardized_data)


# In[ ]:


bow_data_1.T # taking a look at the values of bow_data_1.T


# In[ ]:


final['Score'][0:5]


# In[ ]:


bow_final_data_1 = np.vstack((bow_data_1.T,final['Score'])).T
bow_final_data_1 = pd.DataFrame(bow_final_data_1,columns=('Dim_1', 'Dim_2', "Review"))


# In[ ]:


bow_final_data_1.head()


# In[ ]:


# Plotting the bow model #1
ll = sns.FacetGrid(bow_final_data_1,hue='Review',height=8).map(plt.scatter,'Dim_1','Dim_2').add_legend()

# https://stackoverflow.com/questions/45201514/edit-seaborn-legend#answer-45211976
new_labels = ['Negative', 'Positive']
for t, l in zip(ll._legend.texts, new_labels): t.set_text(l);

plt.title('bow model #1 with perplexity = 5, n_iter = 250')
plt.show()


# **Observation:** If we take the minimum value of n_iter = 250, we dont get a well differentiated plot. The positive comment is covered by the negative ones and henceforth, we will consider values greater than 250.

# In[ ]:


# bow model 2, perplexity = 60, n_iter = 1000
bow_model_2 = TSNE(n_components=2, perplexity=60, n_iter=1000, random_state = 507)
bow_data_2 = bow_model_2.fit_transform(bow_standardized_data)
bow_final_data_2 = np.vstack((bow_data_2.T,final['Score'])).T
bow_final_data_2 = pd.DataFrame(bow_final_data_2,columns=('Dim_1', 'Dim_2', "Review"))


# In[ ]:


bow_final_data_2.head()


# In[ ]:


#Plotting the bow model #2
ll = sns.FacetGrid(bow_final_data_2,hue='Review',height=8).map(plt.scatter,'Dim_1','Dim_2').add_legend()

new_labels = ['Negative', 'Positive']
for t, l in zip(ll._legend.texts, new_labels): t.set_text(l);
    
plt.title('bow model #2 with perplexity = 60, n_iter = 1000')
plt.show()


# **Observation:** Increasing the n_iter to 1000 has increased the stability of the plot.

# In[ ]:


# bow model 3, perplexity = 30, n_iter = 5000
bow_model_3 = TSNE(n_components=2, perplexity=30, n_iter=5000, random_state = 507)
bow_data_3 = bow_model_3.fit_transform(bow_standardized_data)
bow_final_data_3 = np.vstack((bow_data_3.T,final['Score'])).T
bow_final_data_3 = pd.DataFrame(bow_final_data_3,columns=('Dim_1', 'Dim_2', "Review"))


# In[ ]:


bow_final_data_3.head()


# In[ ]:


# Plotting the bow model #3
ll= sns.FacetGrid(bow_final_data_3,hue='Review',height=8).map(plt.scatter,'Dim_1','Dim_2').add_legend()

new_labels = ['Negative', 'Positive']
for t, l in zip(ll._legend.texts, new_labels): t.set_text(l);
    
plt.title('bow model #3 with perplexity = 30, n_iter = 5000')
plt.show()


# **Observation:** In this model, we have increased the perplexity as well as the n_iter and it can be seen that the points are more grouped as compared to the earlier model where the points were scattered.

# In[ ]:


# bow model 4, perplexity = 100, n_iter = 5000
bow_model_4 = TSNE(n_components=2, perplexity=100, n_iter=5000, random_state = 507)
bow_data_4 = bow_model_4.fit_transform(bow_standardized_data)
bow_final_data_4 = np.vstack((bow_data_4.T,final['Score'])).T
bow_final_data_4 = pd.DataFrame(bow_final_data_4,columns=('Dim_1', 'Dim_2', "Review"))


# In[ ]:


bow_final_data_4.head()


# In[ ]:


# Plotting the bow model #4
ll = sns.FacetGrid(bow_final_data_4,hue='Review',height=8).map(plt.scatter,'Dim_1','Dim_2').add_legend()

new_labels = ['Negative', 'Positive']
for t, l in zip(ll._legend.texts, new_labels): t.set_text(l);
    
plt.title('bow model #4 with perplexity = 100, n_iter = 5000')
plt.show()


# **Observation:** Not much difference can be observed with new values of perplexity and n_iter. The plot has already become stable with values from the last model.

# ## [5.2] Applying TNSE on Text TFIDF vectors

# In[ ]:


# Convert the sparse matrix to a dense matrix
final_tfidf_dense = final_tf_idf.todense()


# In[ ]:


# Standardize the data
tfidf_standardized_data = StandardScaler().fit_transform(final_tfidf_dense)


# In[ ]:


# tfidf model #1, perplexity = 5, n_iter = 1000
tfidf_model_1 = TSNE(n_components=2,perplexity=5,n_iter=1000, random_state = 507)


# In[ ]:


tfidf_data_1 = tfidf_model_1.fit_transform(tfidf_standardized_data)


# In[ ]:


tfidf_data_1.T


# In[ ]:


final['Score'][0:10]


# In[ ]:


tfidf_final_data_1 = np.vstack((tfidf_data_1.T,final['Score'])).T


# In[ ]:


tfidf_final_data_1 = pd.DataFrame(tfidf_final_data_1,columns=('Dim_1','Dim_2','Review'))


# In[ ]:


tfidf_final_data_1.head()


# In[ ]:


ll = sns.FacetGrid(tfidf_final_data_1,hue = 'Review',size = 6).map(plt.scatter,'Dim_1','Dim_2').add_legend()

new_labels = ['Negative', 'Positive']
for t, l in zip(ll._legend.texts, new_labels): t.set_text(l);

plt.title('tfidf model #1 with perplexity = 5, n_iter = 1000')
plt.show()


# **Observation:** The plot is not stable yet and is dispersed.

# In[ ]:


# tfidf model #2, perplexity = 30, n_iter = 5000
tfidf_model_2 = TSNE(n_components=2,perplexity=30,n_iter=5000, random_state = 507)
tfidf_data_2 = tfidf_model_2.fit_transform(tfidf_standardized_data)
tfidf_final_data_2 = np.vstack((tfidf_data_2.T,final['Score'])).T
tfidf_final_data_2 = pd.DataFrame(tfidf_final_data_2,columns=('Dim_1','Dim_2','Review'))


# In[ ]:


tfidf_final_data_2.head()


# In[ ]:


ll = sns.FacetGrid(tfidf_final_data_2,hue = 'Review',size = 6).map(plt.scatter,'Dim_1','Dim_2').add_legend()

new_labels = ['Negative', 'Positive']
for t, l in zip(ll._legend.texts, new_labels): t.set_text(l);
    
plt.title('tfidf model #2 with perplexity = 30, n_iter = 5000')
plt.show()


# **Observation:** The plot looks stable at the provided parameters. Due to class imbalance, we see only a few negative (blue) points. 

# In[ ]:


# tfidf model #3, perplexity = 100, n_iter = 5000
tfidf_model_3 = TSNE(n_components=2,perplexity=100,n_iter=5000, random_state = 507)
tfidf_data_3 = tfidf_model_1.fit_transform(tfidf_standardized_data)
tfidf_final_data_3 = np.vstack((tfidf_data_3.T,final['Score'])).T
tfidf_final_data_3 = pd.DataFrame(tfidf_final_data_3,columns=('Dim_1','Dim_2','Review'))


# In[ ]:


tfidf_final_data_3.head()


# In[ ]:


ll = sns.FacetGrid(tfidf_final_data_3,hue = 'Review',size = 6).map(plt.scatter,'Dim_1','Dim_2').add_legend()

new_labels = ['Negative', 'Positive']
for t, l in zip(ll._legend.texts, new_labels): t.set_text(l);
    
plt.title('tfidf model #3 with perplexity = 100, n_iter = 5000')
plt.show()


# **Observation:** Most of the points are clustered around the top right handside of the plot and is not stable.

# In[ ]:


# tfidf model #4, perplexity = 50, n_iter = 1000
tfidf_model_4 = TSNE(n_components=2,perplexity=50,n_iter=1000, random_state = 507)
tfidf_data_4 = tfidf_model_4.fit_transform(tfidf_standardized_data)
tfidf_final_data_4 = np.vstack((tfidf_data_4.T,final['Score'])).T
tfidf_final_data_4 = pd.DataFrame(tfidf_final_data_4,columns=('Dim_1','Dim_2','Review'))


# In[ ]:


tfidf_final_data_4.head()


# In[ ]:


ll = sns.FacetGrid(tfidf_final_data_1,hue = 'Review',size = 6).map(plt.scatter,'Dim_1','Dim_2').add_legend()

new_labels = ['Negative', 'Positive']
for t, l in zip(ll._legend.texts, new_labels): t.set_text(l);
    
plt.title('tfidf model #4 with perplexity = 50, n_iter = 1000')
plt.show()


# **Observation:** We have increased the perplexity and reduced the iterations but we cannot confidently say that the plot is stable.

# ## [5.3] Applying TNSE on Text Avg W2V vectors

# In[ ]:


# Convert the sparse matrix to a dense matrix
# We dont need to convert it to dense matrix as it already is a dense vector


# In[ ]:


# Standardize the data
sent_vectors_standardized = StandardScaler().fit_transform(sent_vectors)


# In[ ]:


# average word2vec model #1, perplexity = 5, n_iter = 5000
avgw2v_model_1 = TSNE(n_components=2,perplexity=5,n_iter = 5000)


# In[ ]:


avgw2v_data_1 = avgw2v_model_1.fit_transform(sent_vectors_standardized)


# In[ ]:


avgw2v_data_1.T


# In[ ]:


final['Score'][0:10]


# In[ ]:


avgw2v_final_data_1 = np.vstack((avgw2v_data_1.T,final['Score'])).T

avgw2v_final_data_1 = pd.DataFrame(avgw2v_final_data_1,columns=('Dim_1','Dim_2','Review'))

avgw2v_final_data_1.head()


# In[ ]:


ll = sns.FacetGrid(avgw2v_final_data_1,hue = 'Review',size = 8).map(plt.scatter,'Dim_1','Dim_2').add_legend()

new_labels = ['Negative', 'Positive']
for t, l in zip(ll._legend.texts, new_labels): t.set_text(l);
    
plt.title('Avg word2vec model #1 with Perplexity = 5 and n_iter = 5000')
plt.show()


# In[ ]:


# average word2vec model #2 with perplexity = 30, n_iter = 5000
avgw2v_model_2 = TSNE(n_components=2,perplexity=30,n_iter = 5000)
avgw2v_data_2 = avgw2v_model_2.fit_transform(sent_vectors_standardized)
avgw2v_final_data_2 = np.vstack((avgw2v_data_2.T,final['Score'])).T
avgw2v_final_data_2 = pd.DataFrame(avgw2v_final_data_2,columns=('Dim_1','Dim_2','Review'))


# In[ ]:


avgw2v_final_data_2.head()


# In[ ]:


ll = sns.FacetGrid(avgw2v_final_data_2,hue = 'Review',size = 8).map(plt.scatter,'Dim_1','Dim_2').add_legend()

new_labels = ['Negative', 'Positive']
for t, l in zip(ll._legend.texts, new_labels): t.set_text(l);
    
plt.title('Avg word2vec model #2 with Perplexity = 30 and n_iter = 5000')
plt.show()


# In[ ]:


# average word2vec model #3 with perplexity = 60, n_iter = 5000
avgw2v_model_3 = TSNE(n_components=2,perplexity=60,n_iter = 5000)
avgw2v_data_3 = avgw2v_model_3.fit_transform(sent_vectors_standardized)
avgw2v_final_data_3 = np.vstack((avgw2v_data_3.T,final['Score'])).T
avgw2v_final_data_3 = pd.DataFrame(avgw2v_final_data_3,columns=('Dim_1','Dim_2','Review'))


# In[ ]:


avgw2v_final_data_3.head()


# In[ ]:


ll = sns.FacetGrid(avgw2v_final_data_3,hue = 'Review',size = 8).map(plt.scatter,'Dim_1','Dim_2').add_legend()

new_labels = ['Negative', 'Positive']
for t, l in zip(ll._legend.texts, new_labels): t.set_text(l);
    
plt.title('Avg word2vec model #3 with Perplexity = 60 and n_iter = 5000')
plt.show()


# In[ ]:


# average word2vec model #4 with perplexity = 100, n_iter = 2500
avgw2v_model_4 = TSNE(n_components=2,perplexity=100,n_iter = 2500)
avgw2v_data_4 = avgw2v_model_4.fit_transform(sent_vectors_standardized)
avgw2v_final_data_4 = np.vstack((avgw2v_data_4.T,final['Score'])).T
avgw2v_final_data_4 = pd.DataFrame(avgw2v_final_data_4,columns=('Dim_1','Dim_2','Review'))


# In[ ]:


avgw2v_final_data_4.head()


# In[ ]:


ll = sns.FacetGrid(avgw2v_final_data_4,hue = 'Review',size = 8).map(plt.scatter,'Dim_1','Dim_2').add_legend()

new_labels = ['Negative', 'Positive']
for t, l in zip(ll._legend.texts, new_labels): t.set_text(l);
    
plt.title('Avg word2vec model #4 with Perplexity = 100 and n_iter = 2500')
plt.show()


# ## [5.4] Applying TNSE on Text TFIDF weighted W2V vectors

# In[ ]:


# Convert the sparse matrix to a dense matrix
# No need as it is already in its dense form


# In[ ]:


# Standardize the data
tfidf_sent_vectors_standardized_data  = StandardScaler().fit_transform(tfidf_sent_vectors)


# In[ ]:


# tfidf-ww2v model 1, perplexity = 5, n_iter = 5000
tfidf_ww2v_model_1 = TSNE(n_components=2,perplexity=5,n_iter = 5000)


# In[ ]:


tfidf_ww2v_data_1 = tfidf_ww2v_model_1.fit_transform(tfidf_sent_vectors_standardized_data)


# In[ ]:


tfidf_ww2v_data_1.T


# In[ ]:


final['Score'][0:10]


# In[ ]:


tfidf_ww2v_final_data_1 = np.vstack((tfidf_ww2v_data_1.T,final['Score'])).T

tfidf_ww2v_final_data_1 = pd.DataFrame(tfidf_ww2v_final_data_1,columns=('Dim_1','Dim_2','Review'))

tfidf_ww2v_final_data_1.head()


# In[ ]:


ll= sns.FacetGrid(tfidf_ww2v_final_data_1,hue = 'Review',size = 8).map(plt.scatter,'Dim_1','Dim_2').add_legend()

new_labels = ['Negative', 'Positive']
for t, l in zip(ll._legend.texts, new_labels): t.set_text(l);
    
plt.title('tfidf weighted word2vec model #1 with perplexity = 5 and n_iter = 5000')
plt.show()


# In[ ]:


# tfidf-ww2v model 2, perplexity = 30, n_iter = 3000
tfidf_ww2v_model_2 = TSNE(n_components=2,perplexity=30,n_iter = 3000)
tfidf_ww2v_data_2 = tfidf_ww2v_model_2.fit_transform(tfidf_sent_vectors_standardized_data)
tfidf_ww2v_final_data_2 = np.vstack((tfidf_ww2v_data_2.T,final['Score'])).T
tfidf_ww2v_final_data_2 = pd.DataFrame(tfidf_ww2v_final_data_2,columns=('Dim_1','Dim_2','Review'))


# In[ ]:


tfidf_ww2v_final_data_2.head()


# In[ ]:


ll = sns.FacetGrid(tfidf_ww2v_final_data_2,hue = 'Review',size = 8).map(plt.scatter,'Dim_1','Dim_2').add_legend()

new_labels = ['Negative', 'Positive']
for t, l in zip(ll._legend.texts, new_labels): t.set_text(l);
    
plt.title('tfidf weighted word2vec model #2 with perplexity = 30 and n_iter = 3000')
plt.show()


# In[ ]:


# tfidf-ww2v model 3, perplexity = 60, n_iter = 5000
tfidf_ww2v_model_3 = TSNE(n_components=2,perplexity=60,n_iter = 5000)
tfidf_ww2v_data_3 = tfidf_ww2v_model_3.fit_transform(tfidf_sent_vectors_standardized_data)
tfidf_ww2v_final_data_3 = np.vstack((tfidf_ww2v_data_3.T,final['Score'])).T
tfidf_ww2v_final_data_3 = pd.DataFrame(tfidf_ww2v_final_data_3,columns=('Dim_1','Dim_2','Review'))


# In[ ]:


tfidf_ww2v_final_data_3.head()


# In[ ]:


ll = sns.FacetGrid(tfidf_ww2v_final_data_3,hue = 'Review',size = 8).map(plt.scatter,'Dim_1','Dim_2').add_legend()

new_labels = ['Negative', 'Positive']
for t, l in zip(ll._legend.texts, new_labels): t.set_text(l);
    
plt.title('tfidf weighted word2vec model #3 with perplexity = 60 and n_iter = 5000')
plt.show()


# In[ ]:


# tfidf-ww2v model 4, perplexity = 100, n_iter = 3000
tfidf_ww2v_model_4 = TSNE(n_components=2,perplexity=100,n_iter = 3000)
tfidf_ww2v_data_4 = tfidf_ww2v_model_4.fit_transform(tfidf_sent_vectors_standardized_data)
tfidf_ww2v_final_data_4 = np.vstack((tfidf_ww2v_data_4.T,final['Score'])).T
tfidf_ww2v_final_data_4 = pd.DataFrame(tfidf_ww2v_final_data_4,columns=('Dim_1','Dim_2','Review'))


# In[ ]:


tfidf_ww2v_final_data_4.head()


# In[ ]:


ll = sns.FacetGrid(tfidf_ww2v_final_data_4,hue = 'Review',size = 8).map(plt.scatter,'Dim_1','Dim_2').add_legend()

new_labels = ['Negative', 'Positive']
for t, l in zip(ll._legend.texts, new_labels): t.set_text(l);
    
plt.title('tfidf weighted word2vec model #4 with perplexity = 100 and n_iter = 3000')
plt.show()


# # [6] Conclusions

# In[ ]:


# Write few sentance about the results that you got and observation that you did from the analysis


# 1. There's no such value which we can call correct and it satifies all the different models. We have to **experiment with the values** of perplexity and n_iter until the t-sne plot becomes stable.
# 2. The **more the number of iternations the better.** As we saw in one of the bow model, if we used n_iter = 250, the visualization on the review type was not well sorted. The plotting of the model  may be stable for a value less than 5000 but it is always better to keep the value of n_iter = 5000, as keeping it low may not yield expected results.
# 3. The positive and the negative **reviews seem to be overlap** each other. This means there are **certain words which occur in both the reviews and as a result positve and negative review classes are not easily separable.**
# 4. The **class imbalance** also played a role, as most of the review text was largely positive. We need to address this imbalance class of Review text so that our analysis can fetch better results.

# 
