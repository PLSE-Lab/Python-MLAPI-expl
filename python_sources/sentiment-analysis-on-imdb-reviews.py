#!/usr/bin/env python
# coding: utf-8

# # Sentiment Analysis on Movie Reviews

# In this notebook Sentiment Analysis is performed on movie reviews.
# 
# ---

# <h1>Content<span class="tocSkip"></span></h1>
# <div class="toc"><ul class="toc-item"><li><span><a href="#Data-Import" data-toc-modified-id="Data-Import-1"><span class="toc-item-num">1&nbsp;&nbsp;</span>Data Import</a></span></li><li><span><a href="#Data-Preprocessing" data-toc-modified-id="Data-Preprocessing-2"><span class="toc-item-num">2&nbsp;&nbsp;</span>Data Preprocessing</a></span></li><li><span><a href="#Feature-Engineering" data-toc-modified-id="Feature-Engineering-3"><span class="toc-item-num">3&nbsp;&nbsp;</span>Feature Engineering</a></span></li><li><span><a href="#Feature-Selection" data-toc-modified-id="Feature-Selection-4"><span class="toc-item-num">4&nbsp;&nbsp;</span>Feature Selection</a></span></li><li><span><a href="#Model-Architecture" data-toc-modified-id="Model-Architecture-5"><span class="toc-item-num">5&nbsp;&nbsp;</span>Model Architecture</a></span></li><li><span><a href="#Model-Training" data-toc-modified-id="Model-Training-6"><span class="toc-item-num">6&nbsp;&nbsp;</span>Model Training</a></span></li><li><span><a href="#Model-Evaluation" data-toc-modified-id="Model-Evaluation-7"><span class="toc-item-num">7&nbsp;&nbsp;</span>Model Evaluation</a></span><ul class="toc-item"><li><span><a href="#Accuracy-&amp;-Loss" data-toc-modified-id="Accuracy-&amp;-Loss-7.1"><span class="toc-item-num">7.1&nbsp;&nbsp;</span>Accuracy &amp; Loss</a></span></li><li><span><a href="#Error-Analysis" data-toc-modified-id="Error-Analysis-7.2"><span class="toc-item-num">7.2&nbsp;&nbsp;</span>Error Analysis</a></span></li></ul></li><li><span><a href="#Model-Application" data-toc-modified-id="Model-Application-8"><span class="toc-item-num">8&nbsp;&nbsp;</span>Model Application</a></span><ul class="toc-item"><li><span><a href="#Test-Predictions" data-toc-modified-id="Test-Predictions-8.1"><span class="toc-item-num">8.1&nbsp;&nbsp;</span>Test Predictions</a></span></li><li><span><a href="#Custom-Reviews" data-toc-modified-id="Custom-Reviews-8.2"><span class="toc-item-num">8.2&nbsp;&nbsp;</span>Custom Reviews</a></span></li></ul></li></ul></div>

# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')
import warnings
warnings.filterwarnings("ignore")
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
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.stem.wordnet import WordNetLemmatizer
from gensim.models import Word2Vec
from gensim.models import KeyedVectors
import pickle
from tqdm import tqdm
import os


# ---
# 
# ## Data Import
# First, we need to import the data.

# In[ ]:


path = "/kaggle/input/aclimdb/aclImdb/"
positiveFiles = [x for x in os.listdir(path+"train/pos/") if x.endswith(".txt")]
negativeFiles = [x for x in os.listdir(path+"train/neg/") if x.endswith(".txt")]
testFiles = [x for x in os.listdir(path+"test/") if x.endswith(".txt")]


# In[ ]:


positiveReviews, negativeReviews= [], []
for pfile in positiveFiles:
    with open(path+"train/pos/"+pfile, encoding="latin1") as f:
        positiveReviews.append(f.read())
for nfile in negativeFiles:
    with open(path+"train/neg/"+nfile, encoding="latin1") as f:
        negativeReviews.append(f.read())


# In[ ]:


reviews = pd.concat([
    pd.DataFrame({"review":positiveReviews, "label":1, "file":positiveFiles}),
    pd.DataFrame({"review":negativeReviews, "label":0, "file":negativeFiles}),
], ignore_index=True).sample(frac=1, random_state=1)
reviews.head()


# **With everything centralized in 1 dataframe, we now perform train, validation and test set splits.

# In[ ]:


# https://stackoverflow.com/a/47091490/4084039
import re

def decontracted(phrase):
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


stopwords= set(['br', 'the', 'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're", "you've",            "you'll", "you'd", 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself',             'she', "she's", 'her', 'hers', 'herself', 'it', "it's", 'its', 'itself', 'they', 'them', 'their',            'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', "that'll", 'these', 'those',             'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does',             'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of',             'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through', 'during', 'before', 'after',            'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further',            'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more',            'most', 'other', 'some', 'such', 'only', 'own', 'same', 'so', 'than', 'too', 'very',             's', 't', 'can', 'will', 'just', 'don', "don't", 'should', "should've", 'now', 'd', 'll', 'm', 'o', 're',             've', 'y', 'ain', 'aren', "aren't", 'couldn', "couldn't", 'didn', "didn't", 'doesn', "doesn't", 'hadn',            "hadn't", 'hasn', "hasn't", 'haven', "haven't", 'isn', "isn't", 'ma', 'mightn', "mightn't", 'mustn',            "mustn't", 'needn', "needn't", 'shan', "shan't", 'shouldn', "shouldn't", 'wasn', "wasn't", 'weren', "weren't",             'won', "won't", 'wouldn', "wouldn't"])


# In[ ]:


from bs4 import BeautifulSoup
i = 0
from tqdm import tqdm
preprocessed_reviews = []
positive_word = []
negative_word = []


# In[ ]:


class Preprocess_text():
    
    
    def decontracted(self,phrase):
        self.phrase =phrase
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

    def fit(self,text):
        self.text = text
        for sentance in tqdm(text):
            sentance = re.sub(r"http\S+", "", sentance)
            sentance = BeautifulSoup(sentance, 'lxml').get_text()
            sentence = decontracted(sentance)
            sentance = re.sub("\S*\d\S*", "", sentance).strip()
            sentance = re.sub('[^A-Za-z]+', ' ', sentance)
            # https://gist.github.com/sebleier/554280
            sentance = ' '.join(e.lower() for e in sentance.split() if e.lower() not in stopwords)
            preprocessed_reviews.append(sentance.strip())
       
        return preprocessed_reviews

    
    
       


# In[ ]:


preprocess = Preprocess_text()


# ---
# 
# ## Data Preprocessing
# The next step is data preprocessing. The following class behaves like your typical SKLearn vectorizer.
# 
# It can perform the following operations.
# * Discard non alpha-numeric characters
# * Set everything to lower case
# * Stems all words using PorterStemmer, and change the stems back to the most occurring existent word.
# * Discard non-Egnlish words (not by default).

# In[ ]:


preprocess_review = preprocess.fit(reviews['review'].values)


# 

# # [4.1] BAG OF WORDS
# 

# In[ ]:


#BoW
count_vect = CountVectorizer() #in scikit-learn
count_vect.fit(preprocess_review)


print("some feature names ", count_vect.get_feature_names()[:10])
print('='*50)

final_counts = count_vect.transform(preprocess_review)
print("the type of count vectorizer ",type(final_counts))
print("the shape of out text BOW vectorizer ",final_counts.get_shape())
print("the number of unique words ", final_counts.get_shape()[1])


# #  Bi-Grams and n-Grams.

# In[ ]:


#bi-gram, tri-gram and n-gram

#removing stop words like "not" should be avoided before building n-grams
# count_vect = CountVectorizer(ngram_range=(1,2))
# please do read the CountVectorizer documentation http://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.CountVectorizer.html

# you can choose these numebrs min_df=10, max_features=5000, of your choice

# min_df means minimum document freq will be 10 means at least 10 times if word present thne only consider
count_vect = CountVectorizer(ngram_range=(1,2), min_df=10)
final_bigram_counts = count_vect.fit_transform(preprocess_review)
print("the type of count vectorizer ",type(final_bigram_counts))
print("the shape of out text BOW vectorizer ",final_bigram_counts.get_shape())
print("the number of unique words including both unigrams and bigrams ", final_bigram_counts.get_shape()[1])


# #  Split our data in train test 
# 

# In[ ]:


x = final_bigram_counts

y = reviews['label'].values

from sklearn.model_selection import train_test_split

#split dataset into train and test data
X_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=1, stratify=y)


# In[ ]:


print(preprocess.uniqueStems.shape)
preprocess.uniqueStems[preprocess.uniqueStems.word.str.contains("disappoint")]


# #  Applying KNN brute force on BOW
# 
# So for knn we need to find best K then only we can fit our model so to find best K we need to do Cross validation 
# Below code is cross validation to find best K for given data 

# In[ ]:



from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
def k_classifier_brute(X_train, y_train):
    # creating odd list of K for KNN
    myList = list(range(0,40))
    neighbors = list(filter(lambda x: x % 2 != 0, myList))

    # empty list that will hold cv scores
    cv_scores = []

    # perform 10-fold cross validation
    for k in tqdm(neighbors):
        knn = KNeighborsClassifier(n_neighbors=k, algorithm = "brute")
        scores = cross_val_score(knn, X_train, y_train, cv=10, scoring='accuracy')
        cv_scores.append(scores.mean())

    # changing to misclassification error
    MSE = [1 - x for x in cv_scores]

    # determining best k
    optimal_k = neighbors[MSE.index(min(MSE))]
    print('\nThe optimal number of neighbors is %d.' % optimal_k)

    # plot misclassification error vs k 
    plt.plot(neighbors, MSE)

    for xy in zip(neighbors, np.round(MSE,3)):
        plt.annotate('(%s, %s)' % xy, xy=xy, textcoords='data')
    plt.title("Misclassification Error vs K")
    plt.xlabel('Number of Neighbors K')
    plt.ylabel('Misclassification Error')
    plt.show()

    print("the misclassification error for each k value is : ", np.round(MSE,3))
    return optimal_k


# In[ ]:


optimal_k_bow = k_classifier_brute(X_train, y_train)


# In[ ]:


knn_optimal = KNeighborsClassifier(n_neighbors=optimal_k_bow)

# fitting the model
knn_optimal.fit(X_train, y_train)
#knn_optimal.fit(bow_data, y_train)

# predict the response
pred = knn_optimal.predict(x_test)

print(X_train.get_shape())


# In[ ]:


train_acc_bow = knn_optimal.score(X_train, y_train)
print("Train accuracy", train_acc_bow)

from sklearn.metrics import accuracy_score
acc_bow = accuracy_score(y_test, pred) * 100
print('\nThe accuracy of the knn classifier on test data for k = %d is %f%%' % (optimal_k_bow, acc_bow))


# #  See on test data we get very bad accuracy on BOW 
# 
# Lets try to predict our own movie review lets what our model will predict .

# In[ ]:


N = ["I saw movie it was very Great movie i am very happy must watch movie"]
test_vectors = count_vect.transform(N)

test_vectors.get_shape()

prediction = knn_optimal.predict(test_vectors)


y_score = knn_optimal.predict_proba(test_vectors)

neg_prob = str(y_score[0][0]*100)
pos_prob = str(y_score[0][1]*100)

if prediction == 0:
    print("Negative review with PRobability : "+neg_prob)
else:
    print("Positive review with probability : "+pos_prob)


# In[ ]:





# # See our model predicted correctly but probability is very less 48% only model confident that review is positive  

# # Lets Try TF-TDF 

# In[ ]:


tf_idf_vect = TfidfVectorizer(ngram_range=(1,2), min_df=10)
tf_idf_vect.fit(preprocess_review)
print("some sample features(unique words in the corpus)",tf_idf_vect.get_feature_names()[0:10])
print('='*50)

final_tf_idf = tf_idf_vect.transform(preprocess_review)
print("the type of count vectorizer ",type(final_tf_idf))
print("the shape of out text TFIDF vectorizer ",final_tf_idf.get_shape())
print("the number of unique words including both unigrams and bigrams ", final_tf_idf.get_shape()[1])


# In[ ]:


x = final_tf_idf


from sklearn.model_selection import train_test_split

#split dataset into train and test data
X_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=1, stratify=y)


# # We already find best k for BOW so we will use k=39 and try to fit model with tf-idf
# 

# In[ ]:


# Please write all the code with proper documentation
knn_optimal = KNeighborsClassifier(n_neighbors=optimal_k_bow)

# fitting the model
knn_optimal.fit(X_train, y_train)
#knn_optimal.fit(bow_data, y_train)

# predict the response
pred = knn_optimal.predict(x_test)

print(X_train.get_shape())



train_acc_bow = knn_optimal.score(X_train, y_train)
print("Train accuracy", train_acc_bow)

from sklearn.metrics import accuracy_score
acc_bow = accuracy_score(y_test, pred) * 100
print('\nThe accuracy of the knn classifier for test data with k = %d is %f%%' % (optimal_k_bow, acc_bow))


# #  See TF-IDF give 80% acc and BOW gives 66% acc Hence we can surely say TF-IDF works far better than BOW 

# In[ ]:


N = ["I saw movie it was very Great movie i am very happy must watch movie, acting was fantastic"]
test_vectors = count_vect.transform(N)

test_vectors.get_shape()

prediction = knn_optimal.predict(test_vectors)
y_score = knn_optimal.predict_proba(test_vectors)

print(y_score)
neg_prob = str(y_score[0][0]*100)
pos_prob = str(y_score[0][1]*100)

if prediction == 0:
    print("Negative review with PRobability : "+neg_prob)
else:
    print("Positive review with probability : "+pos_prob)


# # Here we get little improvment in model performance 
# 
# <h4>So we can say TF-IDF works better than BOW</h4>

# 

# In[ ]:





# In[ ]:





# 

# In[ ]:





# In[ ]:





# In[ ]:





# 

# 

# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# 

# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# 

# In[ ]:





# ---
# 
# ## Model Application

# 

# In[ ]:





# In[ ]:





# In[ ]:




