#!/usr/bin/env python
# coding: utf-8

# ![](https://i.imgur.com/sWyuy4Y.jpg)

# In this notebook, I'll use the dataset "StackSample: 10% of Stack Overflow Q&A", I'll only use the questions and the tags. 
# I will implement a tag suggestion system. I'll both try machine learning models and deep learning models like Word2Vec. I'll then compare the performance of both approaches. 
# 
# This notebook will be divided in 2 parts:
# * PART 1 : Cleaning data and EDA
# * PART 2 : Classical classifiers implemented (SGC classifier, MultiNomial Naive Bayes Classifier, Random Forest Classfier, ...
# 

# **PART 1: Cleaning Data and Exploratory Data Analysis**

# **1.1 Setting up the dataset for later training**

# Importing useful libraries at first

# In[ ]:


import pandas as pd
import numpy as np


import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import seaborn as sns

import warnings

import pickle
import time

import re
from bs4 import BeautifulSoup
import nltk
from nltk.tokenize import ToktokTokenizer
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import stopwords
from string import punctuation

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.model_selection import learning_curve
from sklearn.model_selection import ShuffleSplit
from sklearn.dummy import DummyClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC
from sklearn.linear_model import Perceptron
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn import model_selection
from sklearn.metrics import make_scorer
from sklearn.metrics import confusion_matrix
from sklearn.metrics import hamming_loss
from sklearn.cluster import KMeans


import logging

from scipy.sparse import hstack

warnings.filterwarnings("ignore")
plt.style.use('bmh')
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


# Setting a random seed in order to keep the same random results each time I run the notebook
np.random.seed(seed=11)


# In[ ]:


import os 
print(os.listdir("../input"))


# In[ ]:


# Importing the database 

df = pd.read_csv("../input/Questions.csv", encoding="ISO-8859-1")


# In[ ]:


df.head(5)


# In[ ]:


tags = pd.read_csv("../input/Tags.csv", encoding="ISO-8859-1", dtype={'Tag': str})


# In[ ]:


tags.head(5)


# In[ ]:


df.info()


# In[ ]:


tags.info()


# First, what I want to do is to merge both dataframes. In order to do that, I'll have to group tags by the id of the post since a post can have multiple tags. I'll just use the groupeby function and then merge the dataframes on the id. 

# In[ ]:


tags['Tag'] = tags['Tag'].astype(str)


# In[ ]:


grouped_tags = tags.groupby("Id")['Tag'].apply(lambda tags: ' '.join(tags))


# In[ ]:


grouped_tags.head(5)


# In[ ]:


grouped_tags.reset_index()


# In[ ]:


grouped_tags_final = pd.DataFrame({'Id':grouped_tags.index, 'Tags':grouped_tags.values})


# In[ ]:


grouped_tags_final.head(5)


# In[ ]:


df.drop(columns=['OwnerUserId', 'CreationDate', 'ClosedDate'], inplace=True)


# In[ ]:


df = df.merge(grouped_tags_final, on='Id')


# In[ ]:


df.head(5)


# Now, I'll take only quesions witha score greater than 5. I'm doing that for 2 reasons:
# * 1- I'll require less computational resources from kaggle.
# * 2- The posts will probably be with a better quality and will be better tagged since they have lots of upvotes. 
# 

# In[ ]:


new_df = df[df['Score']>5]


# **1.2 Cleaning Data**

# In[ ]:


plt.figure(figsize=(5, 5))
new_df.isnull().mean(axis=0).plot.barh()
plt.title("Ratio of missing values per columns")


# In[ ]:


print('Dupplicate entries: {}'.format(new_df.duplicated().sum()))
new_df.drop_duplicates(inplace = True)


# This is a very good dataset since there are no missing valeus or dupplicate values. 

# In[ ]:


new_df.drop(columns=['Id', 'Score'], inplace=True)


# Now we only need 3 columns: Body, Title and Tags. 

# **1.2.1 Tags**

# Let's do some cleaning on the tags' column. Furthermore, I decided to keep the 100 most popular tags because I'll be easier to predict the right tag from 100 words than from 14,000 and because we want to keep macro tags and not be too specific since it's only a recommendation for a post, the user can add more specific tags himself. 

# In[ ]:


new_df.head(5)


# In[ ]:


new_df['Tags'] = new_df['Tags'].apply(lambda x: x.split())


# In[ ]:


all_tags = [item for sublist in new_df['Tags'].values for item in sublist]


# In[ ]:


len(all_tags)


# In[ ]:


my_set = set(all_tags)
unique_tags = list(my_set)
len(unique_tags)


# In[ ]:


flat_list = [item for sublist in new_df['Tags'].values for item in sublist]

keywords = nltk.FreqDist(flat_list)

keywords = nltk.FreqDist(keywords)

frequencies_words = keywords.most_common(100)
tags_features = [word[0] for word in frequencies_words]


# In[ ]:


tags_features


# In[ ]:


fig, ax = plt.subplots(figsize=(15, 10))
keywords.plot(100, cumulative=False)


# In[ ]:


def most_common(tags):
    tags_filtered = []
    for i in range(0, len(tags)):
        if tags[i] in tags_features:
            tags_filtered.append(tags[i])
    return tags_filtered


# In[ ]:


new_df['Tags'] = new_df['Tags'].apply(lambda x: most_common(x))
new_df['Tags'] = new_df['Tags'].apply(lambda x: x if len(x)>0 else None)


# In[ ]:


new_df.shape


# In[ ]:


new_df.dropna(subset=['Tags'], inplace=True)


# In[ ]:


new_df.shape


# We are here loosing 10000 rows but the it's for the greater good. 

# **1.2.2 Body**

# In the next two columns: Body and Title, I'll use lots of text processing:
# * Removing html format 
# * Lowering text
# * Transforming abbreviations 
# * Removing punctuation (but keeping words like c# since it's the most popular tag)
# * Lemmatizing words
# * Removing stop words

# In[ ]:


# Converting html to text in the body

new_df['Body'] = new_df['Body'].apply(lambda x: BeautifulSoup(x).get_text()) 


# In[ ]:


def clean_text(text):
    text = text.lower()
    text = re.sub(r"what's", "what is ", text)
    text = re.sub(r"\'s", " ", text)
    text = re.sub(r"\'ve", " have ", text)
    text = re.sub(r"can't", "can not ", text)
    text = re.sub(r"n't", " not ", text)
    text = re.sub(r"i'm", "i am ", text)
    text = re.sub(r"\'re", " are ", text)
    text = re.sub(r"\'d", " would ", text)
    text = re.sub(r"\'ll", " will ", text)
    text = re.sub(r"\'scuse", " excuse ", text)
    text = re.sub(r"\'\n", " ", text)
    text = re.sub(r"\'\xa0", " ", text)
    text = re.sub('\s+', ' ', text)
    text = text.strip(' ')
    return text


# In[ ]:


new_df['Body'] = new_df['Body'].apply(lambda x: clean_text(x)) 


# In[ ]:


token=ToktokTokenizer()


# In[ ]:


punctuation


# In[ ]:


punct = '!"#$%&\'()*+,./:;<=>?@[\\]^_`{|}~'


# In[ ]:


def strip_list_noempty(mylist):
    newlist = (item.strip() if hasattr(item, 'strip') else item for item in mylist)
    return [item for item in newlist if item != '']


# In[ ]:


def clean_punct(text): 
    words=token.tokenize(text)
    punctuation_filtered = []
    regex = re.compile('[%s]' % re.escape(punct))
    remove_punctuation = str.maketrans(' ', ' ', punct)
    for w in words:
        if w in tags_features:
            punctuation_filtered.append(w)
        else:
            punctuation_filtered.append(regex.sub('', w))
  
    filtered_list = strip_list_noempty(punctuation_filtered)
        
    return ' '.join(map(str, filtered_list))



# In[ ]:


new_df['Body'] = new_df['Body'].apply(lambda x: clean_punct(x)) 


# In[ ]:


new_df['Body'][2]


# In[ ]:


lemma=WordNetLemmatizer()
stop_words = set(stopwords.words("english"))


# In[ ]:


def lemitizeWords(text):
    words=token.tokenize(text)
    listLemma=[]
    for w in words:
        x=lemma.lemmatize(w, pos="v")
        listLemma.append(x)
    return ' '.join(map(str, listLemma))

def stopWordsRemove(text):
    
    stop_words = set(stopwords.words("english"))
    
    words=token.tokenize(text)
    
    filtered = [w for w in words if not w in stop_words]
    
    return ' '.join(map(str, filtered))


# In[ ]:


new_df['Body'] = new_df['Body'].apply(lambda x: lemitizeWords(x)) 
new_df['Body'] = new_df['Body'].apply(lambda x: stopWordsRemove(x)) 


# **1.2.3 Title**

# In[ ]:


new_df['Title'] = new_df['Title'].apply(lambda x: str(x))
new_df['Title'] = new_df['Title'].apply(lambda x: clean_text(x)) 
new_df['Title'] = new_df['Title'].apply(lambda x: clean_punct(x)) 
new_df['Title'] = new_df['Title'].apply(lambda x: lemitizeWords(x)) 
new_df['Title'] = new_df['Title'].apply(lambda x: stopWordsRemove(x)) 


# **1.3 EDA**

# Here I'll just use some LDA to see if shows any paterns in words and the main topics.  

# In[ ]:


no_topics = 20


# In[ ]:


text = new_df['Body']


# In[ ]:


vectorizer_train = TfidfVectorizer(analyzer = 'word',
                                       min_df=0.0,
                                       max_df = 1.0,
                                       strip_accents = None,
                                       encoding = 'utf-8', 
                                       preprocessor=None,
                                       token_pattern=r"(?u)\S\S+", # Need to repeat token pattern
                                       max_features=1000)


# In[ ]:


TF_IDF_matrix = vectorizer_train.fit_transform(text)


# In[ ]:


lda = LatentDirichletAllocation(n_topics=no_topics, max_iter=5, learning_method='online', learning_offset=50,random_state=11).fit(TF_IDF_matrix)


# In[ ]:


def display_topics(model, feature_names, no_top_words):
    for topic_idx, topic in enumerate(model.components_):
        print("--------------------------------------------")
        print("Topic %d:" % (topic_idx))
        print(" ".join([feature_names[i]
                        for i in topic.argsort()[:-no_top_words - 1:-1]]))
        print("--------------------------------------------")
        

no_top_words = 10
display_topics(lda, vectorizer_train.get_feature_names(), no_top_words)


# It's a bit disappointing but I'm certain that it can be done better. 

# **PART 2: Classical classifiers**

# **2.1 Data preparation**

# Now our data is almost ready to be put into a classifier. I just need to:
# * Binarize the tags
# * Use a TFIDF for body and Title
# The parameters in the TFIDF are very important for the performance of our tags since we don't want him to delete words like c# or.net. To do that we need to use the following pattern : token_pattern=r"(?u)\S\S+"

# In[ ]:


X1 = new_df['Body']
X2 = new_df['Title']
y = new_df['Tags']


# In[ ]:


multilabel_binarizer = MultiLabelBinarizer()
y_bin = multilabel_binarizer.fit_transform(y)


# In[ ]:


vectorizer_X1 = TfidfVectorizer(analyzer = 'word',
                                       min_df=0.0,
                                       max_df = 1.0,
                                       strip_accents = None,
                                       encoding = 'utf-8', 
                                       preprocessor=None,
                                       token_pattern=r"(?u)\S\S+",
                                       max_features=1000)

vectorizer_X2 = TfidfVectorizer(analyzer = 'word',
                                       min_df=0.0,
                                       max_df = 1.0,
                                       strip_accents = None,
                                       encoding = 'utf-8', 
                                       preprocessor=None,
                                       token_pattern=r"(?u)\S\S+",
                                       max_features=1000)


# In[ ]:


X1_tfidf = vectorizer_X1.fit_transform(X1)
X2_tfidf = vectorizer_X2.fit_transform(X2)


# In[ ]:


X_tfidf = hstack([X1_tfidf,X2_tfidf])


# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(X_tfidf, y_bin, test_size = 0.2, random_state = 0) # Do 80/20 split


# Now it's finally ready. 

# **2.2 One vs Rest**

# To evaluate our models, I'll use the jacard score since it's the best fitted for multi label classification. 

# In[ ]:


def avg_jacard(y_true,y_pred):
    '''
    see https://en.wikipedia.org/wiki/Multi-label_classification#Statistics_and_evaluation_metrics
    '''
    jacard = np.minimum(y_true,y_pred).sum(axis=1) / np.maximum(y_true,y_pred).sum(axis=1)
    
    return jacard.mean()*100

def print_score(y_pred, clf):
    print("Clf: ", clf.__class__.__name__)
    print("Jacard score: {}".format(avg_jacard(y_test, y_pred)))
    print("Hamming loss: {}".format(hamming_loss(y_pred, y_test)*100))
    print("---")    


# In[ ]:


dummy = DummyClassifier()
sgd = SGDClassifier()
lr = LogisticRegression()
mn = MultinomialNB()
svc = LinearSVC()
perceptron = Perceptron()
pac = PassiveAggressiveClassifier()

for classifier in [dummy, sgd, lr, mn, svc, perceptron, pac]:
    clf = OneVsRestClassifier(classifier)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    print_score(y_pred, classifier)


# **2.3 MLP Classifier**

# In[ ]:


mlpc = MLPClassifier()
mlpc.fit(X_train, y_train)

y_pred = mlpc.predict(X_test)

print_score(y_pred, mlpc)


# **2.4 Random Forest**

# In[ ]:


rfc = RandomForestClassifier()
rfc.fit(X_train, y_train)

y_pred = rfc.predict(X_test)

print_score(y_pred, rfc)


# **2.5 GridSearch CV on the best classifier **

# In[ ]:


param_grid = {'estimator__C':[1,10,100,1000]
              }


# In[ ]:


svc = OneVsRestClassifier(LinearSVC())
CV_svc = model_selection.GridSearchCV(estimator=svc, param_grid=param_grid, cv= 5, verbose=10, scoring=make_scorer(avg_jacard,greater_is_better=True))
CV_svc.fit(X_train, y_train)


# In[ ]:


CV_svc.best_params_


# In[ ]:


best_model = CV_svc.best_estimator_


# In[ ]:


y_pred = best_model.predict(X_test)

print_score(y_pred, best_model)


# **2.6 Confusion matrix**

# In[ ]:


for i in range(y_train.shape[1]):
    print(multilabel_binarizer.classes_[i])
    print(confusion_matrix(y_test[:,i], y_pred[:,i]))
    print("")


# **2.7 Exctracting feature importance**

# In[ ]:


def print_top10(feature_names, clf, class_labels):
    """Prints features with the highest coefficient values, per class"""
    for i, class_label in enumerate(class_labels):
        top10 = np.argsort(clf.coef_[i])[-10:]
        print("--------------------------------------------")
        print("%s: %s" % (class_label,
              " ".join(feature_names[j] for j in top10)))
        print("--------------------------------------------")


# In[ ]:


feature_names = vectorizer_X1.get_feature_names() + vectorizer_X2.get_feature_names()


# In[ ]:


print_top10(feature_names, best_model, multilabel_binarizer.classes_)


# **If you have any comment or improvement I'm all ears. **
# ![](https://i.imgur.com/yO8v1sI.png)
