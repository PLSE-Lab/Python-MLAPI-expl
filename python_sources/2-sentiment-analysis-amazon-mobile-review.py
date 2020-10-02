#!/usr/bin/env python
# coding: utf-8

# # Nafisur Rahman
# nafisur21@gmail.com
# https://www.linkedin.com/in/nafisur-rahman

# ## Sentiment Analysis on Amazon Reviews: Unlocked Mobile Phones
# PromptCloud extracted 400 thousand reviews of unlocked mobile phones sold on Amazon.com to find out insights with respect to reviews, ratings, price and their relationships.

# ## Sentiment Analysis
# Finding the sentiment (positive or negative) from Amazon reviews.

# ## A. Loading Libraries and Dataset

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


#import pandas_profiling

get_ipython().run_line_magic('matplotlib', 'inline')


# In[3]:


raw_dataset=pd.read_csv('../input/Amazon_Unlocked_Mobile.csv')
raw_dataset.head()


# ### Basic visualization of dataset

# In[4]:


df=raw_dataset
df.info()


# In[5]:


#pandas_profiling.ProfileReport(df)


# selecting only two columns that is Reviews and Rating

# In[6]:


df.describe()


# In[7]:


df=df[['Reviews','Rating']]


# In[8]:


df.head()


# In[9]:


df.info()


# Removing rows with missing values

# In[10]:


df=df.dropna()
df.info()


# Removing rows with rating=3 that is neutral sentiment

# In[11]:


df=df[df['Rating']!=3]
df.info()


# In[12]:


df=df.reset_index(drop=True)
df.info()


# In[13]:


df['sentiment']=np.where(df['Rating'] > 3, 1, 0)
df.head()


# In[14]:


df.tail()


# ## B. Data Cleaning and Text Preprocessing

# In[15]:


Cstopwords=set(stopwords.words('english')+list(punctuation))
from nltk.stem import WordNetLemmatizer
lemma=WordNetLemmatizer()
def clean_review(review_column):
    review_corpus=[]
    for i in range(0,len(review_column)):
        review=review_column[i]
        #review=BeautifulSoup(review,'lxml').text
        review=re.sub('[^a-zA-Z]',' ',review)
        review=str(review).lower()
        review=word_tokenize(review)
        #review=[stemmer.stem(w) for w in review if w not in Cstopwords]
        review=[lemma.lemmatize(w) for w in review ]
        review=' '.join(review)
        review_corpus.append(review)
    return review_corpus


# In[16]:


review_column=df['Reviews']
review_corpus=clean_review(review_column)


# In[17]:


df['clean_review']=review_corpus
df.tail(20)


# ## C. Creating Features

# ### 1. Bag of words model
# * CountVectorizer

# In[18]:


from sklearn.feature_extraction.text import CountVectorizer


# In[19]:


cv=CountVectorizer(max_features=20000,min_df=5,ngram_range=(1,2))


# In[20]:


X1=cv.fit_transform(df['clean_review'])
X1.shape


# ### 2. Tfidf 

# In[21]:


from sklearn.feature_extraction.text import TfidfVectorizer


# In[22]:


tfidf=TfidfVectorizer(min_df=5, max_df=0.95, max_features = 20000, ngram_range = ( 1, 2 ),
                              sublinear_tf = True)


# In[23]:


tfidf=tfidf.fit(df['clean_review'])


# In[24]:


X2=tfidf.transform(df['clean_review'])
X2.shape


# In[25]:


y=df['sentiment'].values
y.shape


# ## D. Machine Learning

# #### Splitting data into Training and Test set

# In[26]:


X=X2 #X1 for bag of words model and X2 for Tfidf model


# In[27]:


# train test split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
print(X_train.shape, y_train.shape)
print(X_test.shape, y_test.shape)


# In[28]:


# average positive reviews in train and test
print('mean positive review in train : {0:.3f}'.format(np.mean(y_train)))
print('mean positive review in test : {0:.3f}'.format(np.mean(y_test)))


# ### 1. Logistic Regression

# In[29]:


from sklearn.linear_model import LogisticRegression as lr


# In[30]:


model_lr=lr(random_state=0)


# In[31]:


# %%time
# from sklearn.model_selection import GridSearchCV
# parameters = {'C':[0.5,1.0, 10.0], 'penalty' : ['l1','l2']}
# grid_search = GridSearchCV(estimator = model_lr,
#                            param_grid = parameters,
#                            scoring = 'accuracy',
#                            cv = 10,
#                            n_jobs = -1)
# grid_search = grid_search.fit(X_train, y_train)
# best_accuracy = grid_search.best_score_
# best_parameters = grid_search.best_params_
# print('Best Accuracy :',best_accuracy)
# print('Best parameters:\n',best_parameters)


# In[33]:


get_ipython().run_cell_magic('time', '', "model_lr=lr(penalty='l2',C=1.0,random_state=0)\nmodel_lr.fit(X_train,y_train)\ny_pred_lr=model_lr.predict(X_test)\nprint('accuracy for Logistic Regression :',accuracy_score(y_test,y_pred_lr))\nprint('confusion matrix for Logistic Regression:\\n',confusion_matrix(y_test,y_pred_lr))\nprint('F1 score for Logistic Regression :',f1_score(y_test,y_pred_lr))\nprint('Precision score for Logistic Regression :',precision_score(y_test,y_pred_lr))\nprint('recall score for Logistic Regression :',recall_score(y_test,y_pred_lr))\nprint('AUC: ', roc_auc_score(y_test, y_pred_lr))")


# In[34]:


# get the feature names as numpy array
feature_names = np.array(cv.get_feature_names())

# Sort the coefficients from the model
sorted_coef_index = model_lr.coef_[0].argsort()

# Find the 10 smallest and 10 largest coefficients
# The 10 largest coefficients are being indexed using [:-11:-1] 
# so the list returned is in order of largest to smallest
print('Smallest Coefs:\n{}\n'.format(feature_names[sorted_coef_index[:10]]))
print('Largest Coefs: \n{}'.format(feature_names[sorted_coef_index[:-11:-1]]))


# ### 2. Naive Bayes Classifier

# In[35]:


from sklearn.naive_bayes import MultinomialNB
model_nb=MultinomialNB()
model_nb.fit(X_train,y_train)
y_pred_nb=model_nb.predict(X_test)
print('accuracy for Naive Bayes Classifier :',accuracy_score(y_test,y_pred_nb))
print('confusion matrix for Naive Bayes Classifier:\n',confusion_matrix(y_test,y_pred_nb))
print('F1 score for Logistic Regression :',f1_score(y_test,y_pred_nb))
print('Precision score for Logistic Regression :',precision_score(y_test,y_pred_nb))
print('recall score for Logistic Regression :',recall_score(y_test,y_pred_nb))
print('AUC: ', roc_auc_score(y_test, y_pred_nb))


# In[36]:


# get the feature names as numpy array
feature_names = np.array(cv.get_feature_names())

# Sort the coefficients from the model
sorted_coef_index = model_nb.coef_[0].argsort()

# Find the 10 smallest and 10 largest coefficients
# The 10 largest coefficients are being indexed using [:-11:-1] 
# so the list returned is in order of largest to smallest
print('Smallest Coefs:\n{}\n'.format(feature_names[sorted_coef_index[:10]]))
print('Largest Coefs: \n{}'.format(feature_names[sorted_coef_index[:-11:-1]]))


# ### 3. Random Forest

# In[37]:


from sklearn.ensemble import RandomForestClassifier


# In[38]:


get_ipython().run_cell_magic('time', '', "model_rf=RandomForestClassifier()\nmodel_rf.fit(X_train,y_train)\ny_pred_rf=model_rf.predict(X_test)\nprint('accuracy for Random Forest Classifier :',accuracy_score(y_test,y_pred_rf))\nprint('confusion matrix for Random Forest Classifier:\\n',confusion_matrix(y_test,y_pred_rf))")


# In[39]:


# get the feature names as numpy array
feature_names = np.array(cv.get_feature_names())

# Sort the coefficients from the model
sorted_coef_index = model_rf.feature_importances_.argsort()

# Find the 10 smallest and 10 largest coefficients
# The 10 largest coefficients are being indexed using [:-11:-1] 
# so the list returned is in order of largest to smallest
print('Smallest Coefs:\n{}\n'.format(feature_names[sorted_coef_index[:10]]))
print('Largest Coefs: \n{}'.format(feature_names[sorted_coef_index[:-11:-1]]))


# In[ ]:




