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
# Given a review, determine whether the review is positive (rating of 4 or 5) or negative (rating of 1 or 2).
# 
# <br>
# [Q] How to determine if a review is positive or negative?<br>
# <br> 
# [Ans] We could use Score/Rating. A rating of 4 or 5 can be cosnidered as a positive review. A rating of 1 or 2 can be considered as negative one. A review of rating 3 is considered nuetral and such reviews are ignored from our analysis. This is an approximate and proxy way of determining the polarity (positivity/negativity) of a review.
# 
# 
# 

# # [1]. Reading Data

# ## [1.1] Loading the data
# 
# The dataset is available in two forms
# 1. .csv file
# 2. SQLite Database
# 
# In order to load the data, We have used the SQLITE dataset as it is easier to query the data and visualise the data efficiently.
# <br> 
# 
# Here as we only want to get the global sentiment of the recommendations (positive or negative), we will purposefully ignore all Scores equal to 3. If the score is above 3, then the recommendation wil be set to "positive". Otherwise, it will be set to "negative".

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


# In[ ]:


# using SQLite Table to read data.
con = sqlite3.connect('../input/database.sqlite') 

# filtering only positive and negative reviews i.e. 
# not taking into consideration those reviews with Score=3
# SELECT * FROM Reviews WHERE Score != 3 LIMIT 500000, will give top 500000 data points
# you can change the number to any other number based on your computing power

# filtered_data = pd.read_sql_query(""" SELECT * FROM Reviews WHERE Score != 3 LIMIT 500000""", con) 
# for tsne assignment you can take 5k data points

filtered_data = pd.read_sql_query(""" SELECT * FROM Reviews WHERE Score != 3 """, con) 

# Give reviews with Score>3 a positive rating(1), and reviews with a score<3 a negative rating(0).
def partition(x):
    if x < 3:
        return 0
    return 1

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


# #  [2] Exploratory Data Analysis

# ## [2.1] Data Cleaning: Deduplication
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


# As it can be seen above that same user has multiple reviews with same values for HelpfulnessNumerator, HelpfulnessDenominator, Score, Time, Summary and Text and on doing analysis it was found that <br>
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


# #  [3] Preprocessing

# ## [3.1].  Preprocessing Review Text
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


# In[ ]:


final['preprocessed_reviews']=preprocessed_reviews #adding a column of CleanedText which displays the data after pre-processing of the review 
final.head(3)


# <h2><font color='red'>[3.2] Preprocessing Review Summary</font></h2>

# In[ ]:


## Similartly you can do preprocessing for review summary also.
from tqdm import tqdm
preprocessed_reviews_Summary = []
# tqdm is for printing the status bar
for sentance in tqdm(final['Summary'].values):
    sentance = re.sub(r"http\S+", "", sentance)
    #sentance=BeautifulSoup(sentance,'lxml').get_text()
    sentance = decontracted(sentance)
    sentance = re.sub("\S*\d\S*", "", sentance).strip()
    sentance = re.sub('[^A-Za-z]+', ' ', sentance)
    # https://gist.github.com/sebleier/554280
    sentance = ' '.join(e.lower() for e in sentance.split() if e.lower() not in stopwords)
    preprocessed_reviews_Summary.append(sentance.strip())


# In[ ]:


final[' preprocessed_reviews_Summary']= preprocessed_reviews_Summary #adding a column of CleanedText which displays the data after pre-processing of the review 
final.head(3)


# In[ ]:


# store final table into an SQlLite table for future.
conn = sqlite3.connect('final.sqlite')
c=conn.cursor()
conn.text_factory = str
final.to_sql('Reviews', conn, schema=None, if_exists='replace', index=True, index_label=None, chunksize=None, dtype=None)


# In[ ]:


import sqlite3
con = sqlite3.connect("final.sqlite")
cleaned_data = pd.read_sql_query("select * from Reviews", con)
cleaned_data.head(3)
#cleaned_data.shape


# # [4] Featurization

# In[ ]:


# Sampling positive and negative reviews
positive_points = cleaned_data[cleaned_data['Score'] == 1].sample(
    n=50000, random_state=0)
negative_points = cleaned_data[cleaned_data['Score'] == 0].sample(
    n=50000, random_state=0)
total_points = pd.concat([positive_points, negative_points])

# Sorting based on time
total_points['Time'] = pd.to_datetime(
    total_points['Time'], origin='unix', unit='s')
total_points = total_points.sort_values('Time')


# In[ ]:


X = total_points['preprocessed_reviews']
Y = total_points['Score']


# In[ ]:


# https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html
from sklearn.model_selection import train_test_split

# X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.33, shuffle=Flase)# this is for time series split
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.30) # this is random splitting
X_train, X_cv, y_train, y_cv = train_test_split(X_train, y_train, test_size=0.30) # this is random splitting


print(X_train.shape, y_train.shape)
print(X_cv.shape, y_cv.shape)
print(X_test.shape, y_test.shape)

print("="*100)

from sklearn.feature_extraction.text import CountVectorizer
vectorizer = CountVectorizer()
vectorizer.fit(X_train) # fit has to happen only on train data

# we use the fitted CountVectorizer to convert the text to vector
X_train_bow = vectorizer.transform(X_train)
X_cv_bow = vectorizer.transform(X_cv)
X_test_bow = vectorizer.transform(X_test)

print("After vectorizations")
print(X_train_bow.shape, y_train.shape)
print(X_cv_bow.shape, y_cv.shape)
print(X_test_bow.shape, y_test.shape)
print("="*100)


# ## [5.1] Applying Naive Bayes on BOW,<font color='red'> SET 1</font>

# In[ ]:


from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt

train_auc = []
cv_auc = []
K = np.arange(0.00001, 1, .001)
for i in K:
    neigh = MultinomialNB(alpha= i)
    neigh.fit(X_train_bow, y_train)
    # roc_auc_score(y_true, y_score) the 2nd parameter should be probability estimates of the positive class
    # not the predicted outputs
    y_train_pred =  neigh.predict_proba(X_train_bow)[:,1]
    y_cv_pred =  neigh.predict_proba(X_cv_bow)[:,1]
    
    train_auc.append(roc_auc_score(y_train,y_train_pred))
    cv_auc.append(roc_auc_score(y_cv, y_cv_pred))

plt.plot(K, train_auc, label='Train AUC')
plt.plot(K, cv_auc, label='CV AUC')
plt.legend()
plt.xlabel("K: hyperparameter")
plt.ylabel("AUC")
plt.title("ERROR PLOTS")
plt.show()


# In[ ]:


# https://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_curve.html#sklearn.metrics.roc_curve
from sklearn.metrics import roc_curve, auc


neigh = MultinomialNB(alpha= .1)
neigh.fit(X_train_bow, y_train)
# roc_auc_score(y_true, y_score) the 2nd parameter should be probability estimates of the positive class
# not the predicted outputs

train_fpr, train_tpr, thresholds = roc_curve(y_train, neigh.predict_proba(X_train_bow)[:,1])
test_fpr, test_tpr, thresholds = roc_curve(y_test, neigh.predict_proba(X_test_bow)[:,1])

plt.plot(train_fpr, train_tpr, label="train AUC ="+str(auc(train_fpr, train_tpr)))
plt.plot(test_fpr, test_tpr, label="test AUC ="+str(auc(test_fpr, test_tpr)))
plt.legend()
plt.xlabel("K: hyperparameter")
plt.ylabel("AUC")
plt.title("ERROR PLOTS")
plt.show()


# ### [5.1.1] Top 10 important features of positive class from<font color='red'> SET 1</font>

# In[ ]:


# To get all the features name 

bow_features = vectorizer.get_feature_names()

# To count feature for each class while fitting the model
# Number of samples encountered for each (class, feature) during fitting

feat_count = neigh.feature_count_
feat_count.shape


# In[ ]:


# Number of samples encountered for each class during fitting

neigh.class_count_

feature_prob = pd.DataFrame(neigh.feature_log_prob_, columns = bow_features)
feature_prob_tr = feature_prob.T
feature_prob_tr.shape


# In[ ]:


# To show top 10 feature from Positive class
# Feature Importance

print("\n\n Top 10 Positive Features:-\n",feature_prob_tr[1].sort_values(ascending = False)[0:10])


# ### [5.1.1] Top 10 important features of negative class from<font color='red'> SET 1</font>

# In[ ]:


# To show top 10 feature from negative class
# Feature Importance
print("Top 10 Negative Features:-\n",feature_prob_tr[0].sort_values(ascending = False)[0:10])


# In[ ]:


from sklearn.metrics import confusion_matrix
print("Train confusion matrix")
Train_CF= confusion_matrix(y_train, neigh.predict(X_train_bow))
print(Train_CF)
print("Test confusion matrix")
Test_CF=confusion_matrix(y_test, neigh.predict(X_test_bow))
print(Test_CF)


# In[ ]:


# plot confusion matrix to describe the performance of classifier.
import seaborn as sns
class_label = ["negative", "positive"]
df_cm = pd.DataFrame(Test_CF, index = class_label, columns = class_label)
sns.heatmap(df_cm, annot = True, fmt = "d")
plt.title(" Confusiion Matrix")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.show()


# ## [4.3] TF-IDF

# In[ ]:


from sklearn.feature_extraction.text import TfidfVectorizer

tf_idf_vect = TfidfVectorizer(ngram_range=(1,2),min_df=10)
X_train_tfidf = tf_idf_vect.fit_transform(X_train)

X_CV_tfidf = tf_idf_vect.transform(X_cv)
# Convert test text data to its vectorizor
X_test_tfidf = tf_idf_vect.transform(X_test)

print("After vectorizations")
print(X_train_tfidf.shape, y_train.shape)
print(X_CV_tfidf.shape, y_cv.shape)
print(X_test_tfidf.shape, y_test.shape)
print("="*100)


# ## [5.2] Applying Naive Bayes on TFIDF,<font color='red'> SET 2</font>

# In[ ]:


from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt

train_auc = []
cv_auc = []
K = np.arange(0.00001, 1, .001)
for i in K:
    neigh_tfidf = MultinomialNB(alpha= i)
    neigh_tfidf.fit(X_train_tfidf, y_train)
    # roc_auc_score(y_true, y_score) the 2nd parameter should be probability estimates of the positive class
    # not the predicted outputs
    y_train_pred =  neigh_tfidf.predict_proba(X_train_tfidf)[:,1]
    y_cv_pred =  neigh_tfidf.predict_proba(X_CV_tfidf)[:,1]
    
    train_auc.append(roc_auc_score(y_train,y_train_pred))
    cv_auc.append(roc_auc_score(y_cv, y_cv_pred))

plt.plot(K, train_auc, label='Train AUC')
plt.plot(K, cv_auc, label='CV AUC')
plt.legend()
plt.xlabel("K: hyperparameter")
plt.ylabel("AUC")
plt.title("ERROR PLOTS")
plt.show()


# In[ ]:


# https://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_curve.html#sklearn.metrics.roc_curve
from sklearn.metrics import roc_curve, auc


neigh_tfidf = MultinomialNB(alpha= .1)
neigh_tfidf.fit(X_train_tfidf, y_train)
# roc_auc_score(y_true, y_score) the 2nd parameter should be probability estimates of the positive class
# not the predicted outputs

train_fpr, train_tpr, thresholds = roc_curve(y_train, neigh_tfidf.predict_proba(X_train_tfidf)[:,1])
test_fpr, test_tpr, thresholds = roc_curve(y_test, neigh_tfidf.predict_proba(X_test_tfidf)[:,1])

plt.plot(train_fpr, train_tpr, label="train AUC ="+str(auc(train_fpr, train_tpr)))
plt.plot(test_fpr, test_tpr, label="test AUC ="+str(auc(test_fpr, test_tpr)))
plt.legend()
plt.xlabel("K: hyperparameter")
plt.ylabel("AUC")
plt.title("ERROR PLOTS")
plt.show()

print("="*100)


# ### [5.2.1] Top 10 important features of positive class from<font color='red'> SET 2</font>

# In[ ]:


# To get all the features name 

tfidf_features = tf_idf_vect.get_feature_names()

# To count feature for each class while fitting the model
# Number of samples encountered for each (class, feature) during fitting

feat_count = neigh_tfidf.feature_count_
feat_count.shape


# In[ ]:


neigh_tfidf.class_count_


# In[ ]:


feature_prob_tfidf = pd.DataFrame(neigh_tfidf.feature_log_prob_,columns = tfidf_features)
feature_prob_tr_tfidf = feature_prob.T
feature_prob_tr_tfidf.shape


# In[ ]:


# To show top 10 feature from Positive class
# Feature Importance

print("\n\n Top 10 Positive Features:-\n",feature_prob_tr_tfidf[1].sort_values(ascending = False)[0:10])


# ### [5.2.2] Top 10 important features of negative class from<font color='red'> SET 2</font>

# In[ ]:


# To show top 10 feature from negative class
# Feature Importance
print("Top 10 Negative Features:-\n",feature_prob_tr_tfidf[0].sort_values(ascending = False)[0:10])


# In[ ]:


from sklearn.metrics import confusion_matrix
print("Train confusion matrix")
Train_CF= confusion_matrix(y_train, neigh_tfidf.predict(X_train_tfidf))
print(Train_CF)
print("Test confusion matrix")
Test_CF=confusion_matrix(y_test, neigh_tfidf.predict(X_test_tfidf))
print(Test_CF)


# In[ ]:


# plot confusion matrix to describe the performance of classifier.
import seaborn as sns
class_label = ["negative", "positive"]
df_cm = pd.DataFrame(Test_CF, index = class_label, columns = class_label)
sns.heatmap(df_cm, annot = True, fmt = "d")
plt.title(" Confusiion Matrix")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.show()


# In[ ]:


from prettytable import PrettyTable
    
xp = PrettyTable()

xp.field_names = ["vectorizer", "model", "hyper parameter", "Auc"]

xp.add_row(["BOW","Naive Bayes", 0.1,91.7])
xp.add_row(["TFIDF","Naive Bayes", 0.1,95.3])
print(xp)


# # [6] Conclusions

# 1)As in "naive baiyes with tfidf" when alpha = .1 the accuracy is quite good than bow.                                
# 
# 2)This model works well with unseen data and also have high accuracy .
# 
# 3)By comparing accuracy of models it's clear that both the model have almost equal accuracy more than 90%.

# In[ ]:




