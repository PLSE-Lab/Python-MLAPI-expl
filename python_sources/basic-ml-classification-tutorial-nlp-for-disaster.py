#!/usr/bin/env python
# coding: utf-8

# # Basic ML Classification for Beginners
# ## Introduction
# In this simple tutorial, I am going to show you how to do ML classification with simple classification methods, like Naive Bayes.
# 
# First of all, we need to load all the files needed for this process. Kaggle provided 3 files in this competition, but we are going to use only the train.csv file, which has been labelled, to test the data.
# 
# If you want to learn how to use all training data in the classifier and use the testing data for submission, you can skip ahead to the **Create Submission** section.

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.model_selection import train_test_split
from nltk import word_tokenize

import os, re
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# ### Load train.csv file

# In[ ]:


train = pd.read_csv('/kaggle/input/nlp-getting-started/train.csv')
train.head()


# # Preprocessing Text
# The first process that we need to perform is to clean the text, such as lowering all the letters, removing punctuations & links, and removing unnecessary words (stop words). You can use the stop words list already provided by NLTK library, but here I am going to write one myself. Any word contained in the stop words list will be removed from the text. You can also modify the list that I have created and add the words yourself.

# In[ ]:


### function to clean text (remove punctuation, links, lowercase all letters, etc)
def clean_text(text):
    temp = text.lower()
    temp = re.sub('\n', " ", temp)
    temp = re.sub('\'', "", temp)
    temp = re.sub('-', " ", temp)
    temp = re.sub(r"(http|https|pic.)\S+"," ",temp)
    temp = re.sub(r'[^\w\s]',' ',temp)
    
    return temp

### list of stop words that need to be removed
stop_words = ['as', 'in', 'of', 'is', 'are', 'were', 'was', 'it', 'for', 'to', 'from', 'into', 'onto', 
              'this', 'that', 'being', 'the','those', 'these', 'such', 'a', 'an']
### function to remove unnecessary words
def remove_stopwords(text):
    tokenized_words = word_tokenize(text)
    temp = [word for word in tokenized_words if word not in stop_words]
    temp = ' '.join(temp)
    return temp

### We save the cleaned and normalized texts in the new column, called 'clean'
train['clean'] = train['text'].apply(clean_text)
train['clean'] = train['clean'].apply(remove_stopwords)
train['clean']


# ## Combining data with keywords and location attributes
# You can skip this part and go straight to the next one if you just want to use the text as features. In this part, we are going to combine the text with location and keyword attributes to enrich the features.

# In[ ]:


def combine_attributes(text, location, keyword):
    var_list = [text, location, keyword]
    combined = ' '.join(x for x in var_list if x)
    return combined

train.fillna('', inplace=True)
train['combine'] = train.apply(lambda x: combine_attributes(x['clean'], x['location'], x['keyword']), axis=1)


# ### Split train data into train_data and test_data for testing purpose
# In order to test the classifier that we are going to use, we need to split our train data into training and testing. Training data are used to train the classifier, just like what it's called. Meanwhile, testing data are used to test how good the classifier is working.
# 
# If you decided not to combine the text, location, and keyword as I explained above, you should replace the `X = train['combine']` to `X = train['clean']`, so that it will use the cleaned data.

# In[ ]:


X = train['combine']
y = train['target']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=10)


# ### Vectorize the text using TFIDF
# Since computer cannot actually read words, we have to convert those words into readable format, which is in number. So, text vectorization is used to map words or phrases to a corresponding vector of real numbers. One of the most popular technique of text vectorization is TFIDF.

# In[ ]:


from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer = TfidfVectorizer()

X_train_vect = vectorizer.fit_transform(X_train)
X_test_vect = vectorizer.transform(X_test)


# ### Build the Classifier
# We are using Multinomial Naive Bayes for our classifier. It's important to note that only the training data that have been vectorized used to fit the classifier. The testing data that have been vectorized are used for prediction.

# In[ ]:


from sklearn.naive_bayes import MultinomialNB
clf = MultinomialNB()
clf.fit(X_train_vect, y_train)

y_pred = clf.predict(X_test_vect)


# ### Accuracy Score
# The accuracy score attained in this experiment is shown below. It can be improved by using other classification methods, such as SVM, Logistic Regression, Decision Tree, etc.

# In[ ]:


from sklearn.metrics import accuracy_score

accuracy_score(y_test, y_pred)


# ### Confusion Matrix
# Next, we will figure out the distribution of prediction result using confusion matrix. A *good* confusion matrix would consist of many **true positives**, as shown in the diagonal from the top left to bottom right.

# In[ ]:


from sklearn.metrics import confusion_matrix

confusion_matrix(y_test, y_pred)


# # Create Submission File
# After testing out the data and gaining the accuracy using train data only, now it's time to build classifier using all train data to get prediction for the test data. We cannot get accuracy for this experiment, because the result will go straight for submission. Here I will show you how to create submission file for the competition.
# 
# ## Load files
# To submit into the competition, we will use the train data for the training and apply the prediction on the test data. The result of the prediction on the test data are what we are going to submit for the competition. So, first of all, we have to load both files.

# In[ ]:


train = pd.read_csv('/kaggle/input/nlp-getting-started/train.csv')
test = pd.read_csv('/kaggle/input/nlp-getting-started/test.csv')
test.head()


# ## Preprocessing
# Since we already defined the functions above, we can just call the functions for this step to work.

# In[ ]:


### apply preprocessing on train data
train['clean'] = train['text'].apply(clean_text)
train['clean'] = train['clean'].apply(remove_stopwords)

### apply preprocessing on test data
test['clean'] = test['text'].apply(clean_text)
test['clean'] = test['clean'].apply(remove_stopwords)


# In[ ]:


train.fillna('', inplace=True)
train['combine'] = train.apply(lambda x: combine_attributes(x['clean'], x['location'], x['keyword']), axis=1)

test.fillna('', inplace=True)
test['combine'] = test.apply(lambda x: combine_attributes(x['clean'], x['location'], x['keyword']), axis=1)


# In[ ]:


X_train = train['combine']
y_train = train['target']

X_test = test['combine']


# ### Text Vectorization using TF-IDF
# It is important to note that we have to define the object for vectorizer, so don't use the vectorizer object that we have used above.

# In[ ]:


vectorizer = TfidfVectorizer()

X_train_vect = vectorizer.fit_transform(X_train)
X_test_vect = vectorizer.transform(X_test)


# In[ ]:


clf = MultinomialNB()
clf.fit(X_train_vect, y_train)

y_pred = clf.predict(X_test_vect)


# In[ ]:


result = pd.DataFrame({'id':test['id'], 'target':y_pred})
result


# ## Submission File
# Here is the file that we need to submit for submission. It contains the columns id and target.

# In[ ]:


result.to_csv('mnb_submission.csv', index=False)


# Thank you for reading my beginner tutorial on NLP and ML. Feel free to correct my kernel or give any suggestion on the comment below.
# 
# Lia
