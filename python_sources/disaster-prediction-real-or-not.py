#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# # Exploring the data

# I start by reading the input data. The train data set would be the one on which I will train our model. The test dataset would be the one on which I will do my predictions. Finally the sample dataset will be used for storing my final predicted target values against the id number.

# In[ ]:


train = pd.read_csv('/kaggle/input/nlp-getting-started/train.csv')
test = pd.read_csv('/kaggle/input/nlp-getting-started/test.csv')
sample = pd.read_csv('/kaggle/input/nlp-getting-started/sample_submission.csv')


# Next I checked for null values in all the columns of the train dataset. It can be seen from the below result that the columns **keyword** has **61 null values** and **location** has **2533 null values**. Probably it would be better if we drop these columns for now.

# In[ ]:


train.isnull().sum()


# I then created a new dataframe by droping the columns location, keyword and id columns from the train dataset.

# In[ ]:


df = train.drop('location', axis=1)
df = df.drop('keyword', axis=1)
df = df.drop('id', axis=1)


# Next I displayed the first 5 rows of the new dataframe df. It can be seen now the dataframe df has two columns **text** and **target**.

# In[ ]:


df.head()


# I repeated the same steps for the test dataset and created a new df_test dataframe out of it and displayed the first five rows of the dataframe.

# In[ ]:


df_test = test.drop('location', axis=1)
df_test = df_test.drop('keyword', axis=1)
df_test = df_test.drop('id', axis=1)


# In[ ]:


df_test.head()


# Now I checked the number of rows present in the df dataframe. It can be seen that the df dataframe has 7613 rows.

# In[ ]:


len(df)


# Sometimes the texts column may not have null values but could be represented as a blank string. Such texts won't be of any use in our prediction. Hence, I will next filter out such texts if any. 

# In[ ]:


blanks = []

for i,tx,tg in df.itertuples():
    if type(tx) == str:
        if tx.isspace():
            blanks.append(i)
print(len(blanks), "blanks", blanks)


# From the above results it seems that the text column under dataframe has no blank tweets. But, if there were any the below piece of code will filter them out.

# In[ ]:


df.drop(blanks, inplace=True)

len(df)


# Since, there were no blank spaces the length of the dataframe df remained the same even after executing the above code.

# Next I repeated the same for the df_test dataframe.

# In[ ]:


blanks = []

for i,tx in df_test.itertuples():
    if type(tx) == str:
        if tx.isspace():
            blanks.append(i)
print(len(blanks), "blanks", blanks)


# In[ ]:


df_test.drop(blanks, inplace=True)

len(df_test)


# Now I made a count of how many different types of target values that is present in the dataframe df. It seems the dataframe df has **4342 tweets** which don't actually signify a disaster and **3271 tweets** which signify some kind of disaster.

# In[ ]:


df['target'].value_counts()


# Next I made a countplot on the target values of the dataframe df.

# In[ ]:


import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


sns.countplot(df['target'])


# From the above plot it is clear there is higher number of tweets with a target value of 0. 

# # DATA PREPROCESSING

# I started the data preprocessing by splitting the df dataframe into training set set and test set. 

# In[ ]:


from sklearn.model_selection import train_test_split
X = df['text']
y = df['target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=101)


# Next I imported the Spacy library for text preprocessing. This would include the activities like tokenizing, lemmatizing, NER, lower case conversion, stop words removal, punctuation removal, etc as shown below.

# In[ ]:


import string
import spacy
from spacy.lang.en.stop_words import STOP_WORDS
from spacy.lang.en import English

# Create our list of punctuation marks
punctuations = string.punctuation

# Create our list of stopwords
nlp = spacy.load('en')
stop_words = spacy.lang.en.stop_words.STOP_WORDS

# Load English tokenizer, tagger, parser, NER and word vectors
parser = English()

# Creating our tokenizer function
def spacy_tokenizer(sentence):
    # Creating our token object, which is used to create documents with linguistic annotations.
    mytokens = parser(sentence)

    # Lemmatizing each token and converting each token into lowercase
    mytokens = [ word.lemma_.lower().strip() if word.lemma_ != "-PRON-" else word.lower_ for word in mytokens ]

    # Removing stop words
    mytokens = [ word for word in mytokens if word not in stop_words and word not in punctuations ]

    # return preprocessed list of tokens
    return mytokens


# In[ ]:


from sklearn.base import TransformerMixin
# Custom transformer using spaCy
class predictors(TransformerMixin):
    def transform(self, X, **transform_params):
        # Cleaning Text
        return [clean_text(text) for text in X]

    def fit(self, X, y=None, **fit_params):
        return self

    def get_params(self, deep=True):
        return {}

# Basic function to clean the text
def clean_text(text):
    # Removing spaces and converting text into lowercase
    return text.strip().lower()


# # Building the model

# I first imported the necessary libraries as shown below.

# In[ ]:


from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC,SVC
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.model_selection import GridSearchCV
from sklearn import metrics


# I next made a pipeline object performing the activities like Cleaning the text, TfIdf vectorizing, build the model using SVC, do Grid Search for finding the best parameters for the model and then finally fit the model on the training set. After the model has been fit I will do the predictions on the X_test set.

# In[ ]:


model = SVC()
kernel = ['poly', 'rbf', 'sigmoid']
C = [50, 10, 1.0, 0.1, 0.01]
gamma = ['scale']
# define grid search
grid = dict(kernel=kernel,C=C,gamma=gamma)
cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
classifier = GridSearchCV(estimator=model, param_grid=grid, n_jobs=-1, cv=cv, scoring='accuracy',error_score=0)
text_clf_model = Pipeline([("cleaner", predictors()),
                           ('tfidf', TfidfVectorizer(tokenizer = spacy_tokenizer)),
                           ('clf',classifier)])
#                              ('clf', GridSearchCV(estimator=model, param_grid=grid, n_jobs=-1, cv=cv, scoring='accuracy',error_score=0)),
#                             ])
text_clf_model.fit(X_train, y_train)
predictions = text_clf_model.predict(X_test)


# I next printed down the Confusion matrix, classification report and accuracy score. It seems my model gave an accuracy 0.80 which is not bad as a start.

# In[ ]:


sns.heatmap(metrics.confusion_matrix(y_test,predictions),annot=True,cmap='Blues', fmt='g')


# In[ ]:


print("Classification Report:")
print(metrics.classification_report(y_test,predictions))
print("\n")
print(f"Accuracy score is {metrics.accuracy_score(y_test,predictions)}")
   


# Now I will do a predict on the df_test based on our model and save the predictions on a CSV file.

# In[ ]:


X_predict = df_test['text']


# In[ ]:


df_test_predictions = text_clf_model.predict(X_predict)


# In[ ]:


sample_predicted_values = pd.DataFrame(df_test_predictions, columns=['predicted_target'])


# In[ ]:


sample = sample.join(sample_predicted_values)


# In[ ]:


sample = sample.drop('target', axis=1)


# In[ ]:


sample


# In[ ]:


sample.to_csv('Submissions2.csv', index=False)

