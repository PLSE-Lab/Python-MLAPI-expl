#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
#%matplotlib inline
import re
from sklearn.naive_bayes import MultinomialNB # Naive Bayes model
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
data = pd.read_csv("../input/gender-classifier-DFE-791531.csv", encoding='latin1')
data.head(2)


# In[ ]:


data.info() # There are a lot of NaN values in the description column. Later we will see that combining the profile
#description will increase the accuracy score


# In[ ]:


data.description = data.description.fillna('') # the missing values in the description columns were Nan floats; 
# they would throw an error when we use our cleaning function that's why we use this code.


# In[ ]:


data['gender:confidence'].hist() # by training on the profiles that have high gender confidence, the accuracy will go up. 


# In[ ]:


# code adapted from top kernels
# https://www.kaggle.com/crowdflower/twitter-user-gender-classification/kernels
def cleaning(s):
    s = str(s) #s.encode('utf-8').strip()
    s = s.lower()
    s = re.sub('\s\W',' ',s) 
    s = re.sub('\W,\s',' ',s) 
    s = re.sub(r'[^\w]', ' ', s)
    s = re.sub("\d+", " ", s)
    s = re.sub('\s+',' ',s)
    s = re.sub('[!@#$_]', ' ', s) 
    s = s.replace("co"," ")
    s = s.replace("https"," ")
    s = s.replace(",","")
    s = s.replace("[\w*"," ")
    return s

data['text_norm'] = [cleaning(s) for s in data['text']]
data['description_norm'] = [cleaning(s) for s in data['description']]


# In[ ]:


# 50 rows that are in the golden standard
data['_golden'].value_counts()


# In[ ]:


df = data[data['gender:confidence']==1]
df.shape


# In[ ]:


df['gender'].value_counts()


# In[ ]:


# we take out the unknown classifier
df=df[df['gender'].isin(['female', 'male', 'brand'])]
df['gender'].value_counts()


# In[ ]:


df['gender'][0:10]


# In[ ]:


encoder = LabelEncoder()
y = encoder.fit_transform(df['gender'])
y[0:10]


# In[ ]:


male=df[df['gender']=='male']
cvec = CountVectorizer(stop_words='english')
cvec.fit(male['text_norm'])
X_train = pd.DataFrame(cvec.transform(male['text_norm']).todense(),
                       columns=cvec.get_feature_names())
word_counts = X_train.sum(axis=0)
#word_counts.sort_values(ascending = False).head(20)
word_counts.sort_values(ascending = False).head(10).plot.barh()


# In[ ]:


female=df[df['gender']=='female']
cvec = CountVectorizer(stop_words='english')
cvec.fit(female['text_norm'])
X_train = pd.DataFrame(cvec.transform(female['text_norm']).todense(),
                       columns=cvec.get_feature_names())
word_counts = X_train.sum(axis=0)
word_counts.sort_values(ascending = False).head(20)
word_counts.sort_values(ascending = False).head(10).plot.barh()


# In[ ]:


brand=df[df['gender']=='brand']
cvec = CountVectorizer(stop_words='english')
cvec.fit(brand['text_norm'])
X_train = pd.DataFrame(cvec.transform(brand['text_norm']).todense(),
                       columns=cvec.get_feature_names())
word_counts = X_train.sum(axis=0)
word_counts.sort_values(ascending = False).head(20)
word_counts.sort_values(ascending = False).head(10).plot.barh()


# In[ ]:


data.info() # There are a lot of NaN values in the description column. Later we will see that combining the profile
#description will increase the accuracy score


# In[ ]:


data.description = data.description.fillna('') # the missing values in the description columns were Nan floats; 
# they would throw an error when we use our cleaning function that's why we use this code.


# In[ ]:


data['gender:confidence'].hist() # by just training on the profiles that have high gender confidence, the accuracy went up. see below


# In[ ]:


# code adapted from top kernels
# https://www.kaggle.com/crowdflower/twitter-user-gender-classification/kernels
def cleaning(s):
    s = str(s)#s.encode('utf-8').strip()
    s = s.lower()
    s = re.sub('\s\W',' ',s) 
    s = re.sub('\W,\s',' ',s) 
    s = re.sub(r'[^\w]', ' ', s)
    s = re.sub("\d+", " ", s)
    s = re.sub('\s+',' ',s)
    s = re.sub('[!@#$_]', ' ', s) 
    s = s.replace("co"," ")
    s = s.replace("https"," ")
    s = s.replace(",","")
    s = s.replace("[\w*"," ")
    return s

data['text_norm'] = [cleaning(s) for s in data['text']]
data['description_norm'] = [cleaning(s) for s in data['description']]


# In[ ]:


# there are 50 observations in the golden standard
data['_golden'].value_counts()


# In[ ]:


data['gender:confidence'].value_counts();


# In[ ]:


# Training with the profiles that have number 1 confidence will yield to higher accuracy
df = data[data['gender:confidence']==1]
df.shape


# In[ ]:


df['gender'].value_counts()


# In[ ]:


df=df[df['gender'].isin(['female', 'male', 'brand'])]


# In[ ]:


# We take out the unknown class
df['gender'].value_counts()


# In[ ]:


df['gender'][0:10]


# In[ ]:


encoder = LabelEncoder()
y = encoder.fit_transform(df['gender'])
y[0:10]


# In[ ]:


male=df[df['gender']=='male']
cvec = CountVectorizer(stop_words='english')
cvec.fit(male['text_norm'])
X_train = pd.DataFrame(cvec.transform(male['text_norm']).todense(),
                       columns=cvec.get_feature_names())
word_counts = X_train.sum(axis=0)
word_counts.sort_values(ascending = False).head(20)
word_counts.sort_values(ascending = False).head(10).plot.barh()


# In[ ]:


female=df[df['gender']=='female']
cvec = CountVectorizer(stop_words='english')
cvec.fit(female['text_norm'])
X_train = pd.DataFrame(cvec.transform(female['text_norm']).todense(),
                       columns=cvec.get_feature_names())
word_counts = X_train.sum(axis=0)
word_counts.sort_values(ascending = False).head(20)
word_counts.sort_values(ascending = False).head(10).plot.barh()


# In[ ]:


brand=df[df['gender']=='brand']
cvec = CountVectorizer(stop_words='english')
cvec.fit(brand['text_norm'])
X_train = pd.DataFrame(cvec.transform(brand['text_norm']).todense(),
                       columns=cvec.get_feature_names())
word_counts = X_train.sum(axis=0)
word_counts.sort_values(ascending = False).head(20)
word_counts.sort_values(ascending = False).head(10).plot.barh()


# In[ ]:


def docm(y_true, y_pred, labels=None):
    """ Returns a confusion matrix as a dataframe.
    Uses either passed in labels or integers as labels."""
    cm = confusion_matrix(y_true, y_pred)
    if labels is not None:
        cols = ['p_'+c for c in labels]
        df = pd.DataFrame(cm, index=labels, columns=cols)
    else:
        cols = ['p_'+str(i) for i in xrange(len(cm))]
        df = pd.DataFrame(cm, columns=cols)
    return df


# In[ ]:


from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.metrics import confusion_matrix, accuracy_score


# In[ ]:


models = [KNeighborsClassifier(),
          LogisticRegression(),
          DecisionTreeClassifier(),
          SVC(),
          RandomForestClassifier(),
          ExtraTreesClassifier()]

tvec = TfidfVectorizer()
#                        stop_words='english',
#                        sublinear_tf=True,
#                        max_df=0.5,
#                        min_df=2,
#                        max_features=1000)


X = tvec.fit_transform(df['text_norm'])

encoder = LabelEncoder()
y = encoder.fit_transform(df['gender'])

# split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)


# In[ ]:


names=['brand','female','male']


# In[ ]:


res = []

for model in models:
    print(model)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    score = accuracy_score(y_test, y_pred)
    print("\n")
    print(score)
    print("\n")
    cm = docm(y_test, y_pred,names)
    print("\n")
    print(cm)
    print("\n")
    res.append([model, score])


# In[ ]:


data['all_features'] = data['text_norm'].str.cat(data['description_norm'], sep=' ')
df = data[data['gender:confidence']==1]
# Naive Bayes, Countvectorizer, Description and tweet
cvec = CountVectorizer()
X = cvec.fit_transform(df['all_features'])

encoder = LabelEncoder()
y = encoder.fit_transform(df['gender'])

# split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# take a look at the shape of each of these
print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)
nb = MultinomialNB()
nb.fit(X_train, y_train)

print(nb.score(X_test, y_test))


# In[ ]:


The Best score is obtained from the Naive Bayes Classifier


# In[ ]:




