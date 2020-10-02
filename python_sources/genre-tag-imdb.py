#!/usr/bin/env python
# coding: utf-8

# # Genre Labeler of IMDB descriptions
# 
# Here I have created a model for labeling the genre of an input description based on the IMDB dataset. The current version of the model only has an accuracy of 48%.  When tested with data that it has not yet seen (2018 movies) the results are pretty convincing. For now, it is able to retrieve a sample description from the user as the input and will output the likeliest Genre tag for that description. Right now this is using the pipeline `model = Pipeline([('vectorizer',CountVectorizer()),('tfidf',TfidfTransformer()),('clf',OneVsRestClassifier(LinearSVC(class_weight='balanced')))])` for its prediction. What it basically does it look for frequency of the words from the description and create its mapping from there. From that mapping, the new descriptions can then be classified based on the frequency of words as well.
# 
# This could still be improved with the use of Deep Learning, with the use of LSTM or Conv1D (in wavenet configuration) we could possibly make the model think of the context of the description based on the arrangement of the words instead of just the frequency.

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from collections import Counter
import matplotlib.pyplot as plt
import string

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[ ]:


import warnings
warnings.filterwarnings('ignore')


# In[ ]:


# Read the dataset
data = pd.read_csv('../input/IMDB-Movie-Data.csv')


# In[ ]:


# Basic check of the data
data.info()


# In[ ]:


# Checking the data
data.head(10)


# ## Some Preprocessing
# 
# Here I did some preprocessing of the data. We will remove the punctuation marks for the description and make them all lower case. We also make the Genre lower case for uniformity as well. If we do not do this then we could end up with multiple labels as the code will see the different cased genre as unique even though they point to the same genre.

# In[ ]:


# Pre-processing: We want to at do a lower case of Genre and Description columns.
# Also added removal of punctuation.
translator = str.maketrans('','',string.punctuation)

data['Description']= data['Description'].str.lower().str.translate(translator)
data['Genre']= data['Genre'].str.lower()
data.head()


# ## Splitting up the Genre into the individual components
# 
# If we look at the `Genre` column we can see that there are movies that have more than one genre. Since we want to come up with a genre tag/labeler we must consider the each of the items listed under the `Genre` entry of a data set to be unique. Otherwise we would be labeling them based on the grouping (ex. Action,Sci-fi) instead of (Action __AND__ Sci-fi).<br><br>
# __To Do:__ Come up with a way that we would be able to spilt the genre into each individual components but still retain the same description. For example __[`genre`: `Sci-fi, Action` , `description`:`This is a good movie, lots of action and adventure...`]__ would become two entries, one for Action and one for Sci-fi.<br>
#  __[`genre`: `Action` , `description`:`This is a good movie, lots of action and adventure...`]__ <br>
#   __[`genre`: `Sci-fi` , `description`:`This is a good movie, lots of action and adventure...`]__ 
# 

# In[ ]:


# This is for splitting the individual grouped genre into individual genre
new_data = pd.DataFrame(columns = ['Title','Genre','Description'])
for i in range(len(data['Genre'])):  # GO over the Genre
    for word in data['Genre'][i].split(","): # We will split the Genre
        new_data = new_data.append({'Title':data['Title'][i],'Genre':word,'Description':data['Description'][i]}, ignore_index = 1)
# Checking the new data created
new_data.info()


# In[ ]:


new_data


# In[ ]:


# This is for splitting the individual grouped genre into individual genre
new_data = pd.DataFrame(columns = ['Title','Genre','Description'])
for i in range(len(data['Genre'])):  # GO over the Genre
    for word in data['Genre'][i].split(","): # We will split the Genre
        new_data = new_data.append({'Title':data['Title'][i],'Genre':word,'Description':data['Description'][i]}, ignore_index = 1)
# Checking the new data created
new_data.info()


# In[ ]:


new_data.head(5)


# In[ ]:


Genre_count = Counter(new_data['Genre'])
Genre_count


# ## Trimming up less frequent genre
# 
# Reviewing the dataset, we can see that there are genre that appear less frequent than others. For example we have `musical` which only appears 5 times or `western` which only has 7 entries. To avoid under-representation I trimmed up the genre and made one new genre `others` to cover all the genres that have less than 100 entries.

# In[ ]:


# Aggregate all Genres with less that 100 items as 'others'
others = ['animation','family','music','history','western','war','musical','sport','biography']


# In[ ]:


for i in range(len(new_data['Genre'])):
    if new_data['Genre'][i] in others:
        new_data.iloc[i]['Genre'] = 'others'


# In[ ]:


new_data[new_data['Genre']=='others']


# ## Cleaning up the description (Again)
# 
# This time we are going after the spaces, new lines, and digits (0-9) in the data.

# In[ ]:


import re

def cleanup(string):
    '''
    Helper Function:
    Will clean up the input string (for description) in this case.
    '''
    string = re.sub(r"\n",'',string)
    string = re.sub(r"\n",'',string)
    string = re.sub(r"[0-9]",'digit',string) # We do not care for the specific number
    string = re.sub(r"\''",'',string)
    string = re.sub(r'\"','',string)
    return string.strip().lower()
X = []

for item in range(new_data.shape[0]):
    X.append(cleanup(new_data.iloc[item][2]))
y = np.array(new_data['Genre'])


# In[ ]:


X


# ## Splitting the data
# 
# Here we split up the data into training data and testing data with a 70-30 split.

# In[ ]:


from sklearn.model_selection import train_test_split
X_train,X_test, y_train, y_test = train_test_split(X,y, test_size = 0.3, random_state = 5)


# ## Creating the pipeline
# 
# I am thinking this is similar to the way we define models in keras/tensorflow (?). We are having the countvectorizer for this model then we use that for the tfidf transformer and finally the classifier.

# In[ ]:


# Pipeline
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer, TfidfVectorizer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC
model = Pipeline([('vectorizer',CountVectorizer()),('tfidf',TfidfTransformer()),('clf',OneVsRestClassifier(LinearSVC(class_weight='balanced')))])


# ## Using Gridsearch to look for optimal parameters
# 
# Here we are simply going over some parameters for our model to see which one has the best score overall.

# In[ ]:


#Selecting the best parameter via gridsearch
from sklearn.grid_search import GridSearchCV
parameters = {'vectorizer__ngram_range':[(1,1), (1,2),(2,2)],
              'vectorizer__min_df':[0,0.001],
              'tfidf__use_idf':('True','False')}
gs_clf_svm = GridSearchCV(model, parameters, n_jobs= -1)
gs_clf_svm = gs_clf_svm.fit(X,y)
print(gs_clf_svm.best_score_)
print(gs_clf_svm.best_params_)


# In[ ]:


model = Pipeline([('vectorizer',CountVectorizer(ngram_range = (1,2),min_df=0)),
                  ('tfidf',TfidfTransformer(use_idf=True)),('clf',OneVsRestClassifier(LinearSVC(class_weight='balanced')))])


# In[ ]:


model.fit(X,y)
pred = model.predict(X_test)
from sklearn.metrics import confusion_matrix, accuracy_score
confusion_matrix(pred,y_test)


# In[ ]:


model.score(X,y)


# In[ ]:


model.predict(['Robert McCall serves an unflinching justice for the exploited and oppressed, but how far will he go when that is someone he loves?'])[0]

## Description is from 'The Equalizer' Genre: Action, Crime, Thriller 


# ## Balancing the dataset

# In[ ]:


a = Counter(new_data['Genre'])
s = set(a)
s


# In[ ]:


trimmed = pd.DataFrame(columns=new_data.columns)


# In[ ]:


for genre in s:
    trimmed=trimmed.append(new_data[new_data['Genre'] == genre][:100])


# In[ ]:


trimmed.info()


# In[ ]:


trimmed[trimmed['Genre']=='action'].head()


# In[ ]:


import re

def cleanup(string):
    '''
    Helper Function:
    Will clean up the input string (for description) in this case.
    '''
    string = re.sub(r"\n",'',string)
    string = re.sub(r"\n",'',string)
    string = re.sub(r"[0-9]",'digit',string) # We do not care for the specific number
    string = re.sub(r"\''",'',string)
    string = re.sub(r'\"','',string)
    return string.strip().lower()
X = []

for item in range(trimmed.shape[0]):
    X.append(cleanup(trimmed.iloc[item][2]))
y = np.array(trimmed['Genre'])


# In[ ]:


from sklearn.model_selection import train_test_split
X_train,X_test, y_train, y_test = train_test_split(X,y, test_size = 0.3, random_state = 5)


# In[ ]:


# Pipeline
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer, TfidfVectorizer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC
model = Pipeline([('vectorizer',CountVectorizer()),('tfidf',TfidfTransformer()),('clf',OneVsRestClassifier(LinearSVC(class_weight='balanced')))])


# In[ ]:


#Selecting the best parameter via gridsearch
from sklearn.grid_search import GridSearchCV
parameters = {'vectorizer__ngram_range':[(1,1), (1,2),(2,2)],
              'vectorizer__min_df':[0,0.001],
              'tfidf__use_idf':('True','False')}
gs_clf_svm = GridSearchCV(model, parameters, n_jobs= -1)
gs_clf_svm = gs_clf_svm.fit(X,y)
print(gs_clf_svm.best_score_)
print(gs_clf_svm.best_params_)


# In[ ]:


model = Pipeline([('vectorizer',CountVectorizer(ngram_range = (1,2),min_df=0)),
                  ('tfidf',TfidfTransformer(use_idf=True)),('clf',OneVsRestClassifier(LinearSVC(class_weight='balanced')))])


# In[ ]:


model.fit(X_train,y_train)
pred = model.predict(X_test)
from sklearn.metrics import confusion_matrix, accuracy_score
confusion_matrix(pred,y_test)


# In[ ]:


model.score(X,y)


# In[ ]:


Description_=input()
model.predict([Description_])[0]


# ## Saving the model
# 
# source: https://machinelearningmastery.com/save-load-machine-learning-models-python-scikit-learn/

# In[ ]:


import pickle
filename = "model-genre-classifier.sav"
pickle.dump(model, open(filename,'wb'))


# In[ ]:


loaded_model = pickle.load(open(filename,'rb'))
loaded_model.score(X,y)


# ## Summary
# 
# For now we have created a model that can classify an input description to a genre. We can still improve on the accuracy of the model, so far we only have a score of 0.495 although it can convincingly ouput a genre that is within the actual genre classification of the target.
# 
# Some improvements for this would be:
# 
# * The use of an ANN (for improved classification but still frequency based) or the use of RNN-LSTM or Conv1D (WaveNet configuration) for analysis on the arrangement of the words and not just the frequency.
# 
# * A possible feature for this would be to output the top 3 Genre for the given description. One possbile way to do this would be the use of cosine similarity although I have to figure out first how to add it to the pipeline.

# In[ ]:


### STOP HERE


# In[ ]:




