#!/usr/bin/env python
# coding: utf-8

# # Nafisur Rahman
# nafisur21@gmail.com<br>
# https://www.linkedin.com/in/nafisur-rahman

# # Sentiment Analysis
# Finding the sentiment (positive or negative) from IMDB movie reviews.

# ## About this Project
# This is a kaggle project based on kaggle dataset of "Bag of Words Meets Bags of Popcorn". Original dataset can be found from stanford website http://ai.stanford.edu/~amaas/data/sentiment/.<br>
# The labeled data set consists of 50,000 IMDB movie reviews, specially selected for sentiment analysis. The sentiment of reviews is binary, meaning the IMDB rating < 5 results in a sentiment score of 0, and rating >=7 have a sentiment score of 1. No individual movie has more than 30 reviews. The 25,000 review labeled training set does not include any of the same movies as the 25,000 review test set. <br>
# * id - Unique ID of each review
# * sentiment - Sentiment of the review; 1 for positive reviews and 0 for negative reviews
# * review - Text of the review

# ## A. Loading libraries and Dataset

# ### Importing Packages

# In[1]:


import os
print(os.listdir("../input"))


# In[2]:


import nltk
import re
import numpy as np # linear algebra
import pandas as pd # data processing
import random
import matplotlib.pyplot as plt

from bs4 import BeautifulSoup
from nltk.corpus import stopwords
from string import punctuation
from nltk.stem import SnowballStemmer
stemmer=SnowballStemmer('english')

from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer

from nltk.tokenize import word_tokenize

get_ipython().run_line_magic('matplotlib', 'inline')


# ### Loading the dataset

# In[3]:


raw_data_train=pd.read_csv('../input/labeledTrainData.tsv',sep='\t')
raw_data_test=pd.read_csv('../input/testData.tsv',sep='\t')


# ### Basic visualization of dataset

# In[4]:


print(raw_data_train.shape)
print(raw_data_test.shape)


# In[5]:


raw_data_train.info()


# In[6]:


raw_data_test.info()


# In[7]:


raw_data_train.head()


# In[8]:


raw_data_test.head()


# In[9]:


raw_data_train['review'][0]


# ## B. Data Cleaning and Text Preprocessing

# Removing tags and markup

# In[10]:


from bs4 import BeautifulSoup
soup=BeautifulSoup(raw_data_train['review'][0],'lxml').text
soup


# Removing non-letters

# In[11]:


import re
re.sub('[^a-zA-Z]',' ',raw_data_train['review'][0])


# Word tokenization

# In[12]:


from nltk.tokenize import word_tokenize
word_tokenize((raw_data_train['review'][0]).lower())[0:20]


# Removing stopwords

# In[13]:


from nltk.corpus import stopwords
from string import punctuation
Cstopwords=set(stopwords.words('english')+list(punctuation))


# In[14]:


[w for w in word_tokenize(raw_data_train['review'][0]) if w not in Cstopwords][0:20]


# ### Defining a function that will perform the preprocessing task at one go

# In[15]:


from bs4 import BeautifulSoup
import re
from nltk.tokenize import word_tokenize
from nltk.stem import SnowballStemmer
stemmer=SnowballStemmer('english')
from nltk.stem import WordNetLemmatizer
lemma=WordNetLemmatizer()
from nltk.corpus import stopwords
from string import punctuation
Cstopwords=set(stopwords.words('english')+list(punctuation))
def clean_review(df):
    review_corpus=[]
    for i in range(0,len(df)):
        review=df[i]
        review=BeautifulSoup(review,'lxml').text
        review=re.sub('[^a-zA-Z]',' ',review)
        review=str(review).lower()
        review=word_tokenize(review)
        #review=[stemmer.stem(w) for w in word_tokenize(str(review).lower()) if w not in Cstopwords]
        review=[lemma.lemmatize(w) for w in review ]
        review=' '.join(review)
        review_corpus.append(review)
    return review_corpus


# In[16]:


df=raw_data_train['review']
clean_train_review_corpus=clean_review(df)
clean_train_review_corpus[0]


# In[17]:


df1=raw_data_test['review']
clean_test_review_corpus=clean_review(df1)
clean_test_review_corpus[0]


# In[18]:


df=raw_data_train
df['clean_review']=clean_train_review_corpus
df.head()


# ## C. Creating Features
# 1. Bag of Words (CountVectorizer)
# 2. tf
# 3. tfidf

# ### 1. Bag of Words model

# In[19]:


from sklearn.feature_extraction.text import CountVectorizer


# To limit the size of the feature vectors, we should choose some maximum vocabulary size. Below, we use the 5000 most frequent words (remembering that stop words have already been removed).

# In[20]:


cv=CountVectorizer(max_features=20000,min_df=5,ngram_range=(1,2))


# In[21]:


X1=cv.fit_transform(df['clean_review'])
X1.shape


# In[22]:


train_data_features = X1.toarray()


# In[23]:


y=df['sentiment'].values
y.shape


# ## D. Machine Learning

# #### Splitting data into Training and Test set

# In[24]:


X=train_data_features


# In[25]:


# train test split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
print(X_train.shape, y_train.shape)
print(X_test.shape, y_test.shape)


# In[26]:


# average positive reviews in train and test
print('mean positive review in train : {0:.3f}'.format(np.mean(y_train)))
print('mean positive review in test : {0:.3f}'.format(np.mean(y_test)))


# ### 1. Naive Bayes Classifier

# In[27]:


from sklearn.naive_bayes import MultinomialNB
model_nb=MultinomialNB()
model_nb.fit(X_train,y_train)
y_pred_nb=model_nb.predict(X_test)
print('accuracy for Naive Bayes Classifier :',accuracy_score(y_test,y_pred_nb))
print('confusion matrix for Naive Bayes Classifier:\n',confusion_matrix(y_test,y_pred_nb))


# In[28]:


# get the feature names as numpy array
feature_names = np.array(cv.get_feature_names())

# Sort the coefficients from the model
sorted_coef_index = model_nb.coef_[0].argsort()

# Find the 10 smallest and 10 largest coefficients
# The 10 largest coefficients are being indexed using [:-11:-1] 
# so the list returned is in order of largest to smallest
print('Smallest Coefs:\n{}\n'.format(feature_names[sorted_coef_index[:10]]))
print('Largest Coefs: \n{}'.format(feature_names[sorted_coef_index[:-11:-1]]))


# ### 2. Random Forest

# In[29]:


from sklearn.ensemble import RandomForestClassifier


# In[30]:


model_rf=RandomForestClassifier(random_state=0)


# In[38]:


# %%time
# from sklearn.model_selection import GridSearchCV
# parameters = {'n_estimators':[100,200],'criterion':['entropy','gini'],
#               'min_samples_leaf':[2,5,7],
#               'max_depth':[5,6,7]
#                }
# grid_search = GridSearchCV(estimator = model_rf,
#                            param_grid = parameters,
#                            scoring = 'accuracy',
#                            cv = 10,
#                            n_jobs = -1)
# grid_search = grid_search.fit(X_train, y_train)
# best_accuracy = grid_search.best_score_
# best_parameters = grid_search.best_params_
# print('Best Accuracy :',best_accuracy)
# print('Best parameters:\n',best_parameters)


# In[32]:


get_ipython().run_cell_magic('time', '', "model_rf=RandomForestClassifier()\nmodel_rf.fit(X_train,y_train)\ny_pred_rf=model_rf.predict(X_test)\nprint('accuracy for Random Forest Classifier :',accuracy_score(y_test,y_pred_rf))\nprint('confusion matrix for Random Forest Classifier:\\n',confusion_matrix(y_test,y_pred_rf))")


# ### 3. Logistic Regression

# In[33]:


from sklearn.linear_model import LogisticRegression as lr


# In[34]:


model_lr=lr(random_state=0)


# In[36]:


get_ipython().run_cell_magic('time', '', "model_lr=lr(penalty='l2',C=1.0,random_state=0)\nmodel_lr.fit(X_train,y_train)\ny_pred_lr=model_lr.predict(X_test)\nprint('accuracy for Logistic Regression :',accuracy_score(y_test,y_pred_lr))\nprint('confusion matrix for Logistic Regression:\\n',confusion_matrix(y_test,y_pred_lr))\nprint('F1 score for Logistic Regression :',f1_score(y_test,y_pred_lr))\nprint('Precision score for Logistic Regression :',precision_score(y_test,y_pred_lr))\nprint('recall score for Logistic Regression :',recall_score(y_test,y_pred_lr))\nprint('AUC: ', roc_auc_score(y_test, y_pred_lr))")


# In[37]:


# get the feature names as numpy array
feature_names = np.array(cv.get_feature_names())

# Sort the coefficients from the model
sorted_coef_index = model_lr.coef_[0].argsort()

# Find the 10 smallest and 10 largest coefficients
# The 10 largest coefficients are being indexed using [:-11:-1] 
# so the list returned is in order of largest to smallest
print('Smallest Coefs:\n{}\n'.format(feature_names[sorted_coef_index[:10]]))
print('Largest Coefs: \n{}'.format(feature_names[sorted_coef_index[:-11:-1]]))


# In[ ]:




