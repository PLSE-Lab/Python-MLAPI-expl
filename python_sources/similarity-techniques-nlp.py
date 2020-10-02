#!/usr/bin/env python
# coding: utf-8

# ##Predict Duplicate using basic ML + NLP techniques##
# 
# I am trying to predict the duplicate sentences using vector similarity calculations and NLP technique in this module.
# 
# Methods to be tried out
# 
#  - List item
# 
# - BOW/TFIDF + Cosine/Euclidean Similarity(other similarity techniques)
# - BOW/TFIDF + POS tagger + Cosine/Euclidean Similarity(other similarity techniques)
# - BOW/TFIDF + POS tagging + Dependency parsing + Cosine/Euclidean Similarity(other similarity techniques)
# - Word2Vec + Cosine/Euclidean Similarity(other similarity techniques)
# - Doc2Vec + Cosine/Euclidean Similarity(other similarity techniques)

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.


# **Reading train data, Cleaning**
# 
# *Reading Training Data ,
# Removing duplicates , 
# Removing NULL values*

# In[ ]:


def read_data():
    df = pd.read_csv("../input/train.csv")
    print ("Shape of base training File = ", df.shape)
    # Remove missing values and duplicates from training data
    df.drop_duplicates(inplace=True)
    df.dropna(inplace=True)
    print("Shape of base training data after cleaning = ", df.shape)
    return df

df_train = read_data()
print (df_train.head(2))


# **EDA**
# 
# Some EDA on the data to get a look and feel about the data.

# In[ ]:


from collections import Counter
import matplotlib.pyplot as plt
import operator
file = open("EDA.txt","wb")


def eda(df):
    eda_file = open("EDA.text","wb")
    print ("Duplicate Count = %s , Non Duplicate Count = %s" 
           %(df.is_duplicate.value_counts()[1],df.is_duplicate.value_counts()[0]))
    
    question_ids_combined = df.qid1.tolist() + df.qid2.tolist()
    print ("Unique Questions = %s" %(len(np.unique(question_ids_combined))))
    question_ids_counter = Counter(question_ids_combined)
    sorted_question_ids_counter = sorted(question_ids_counter.items(), key=operator.itemgetter(1))
    question_appearing_more_than_once = [i for i in question_ids_counter.values() if i > 1]
   
    print ("Count of Quesitons appearing more than once = %s" %(len(question_appearing_more_than_once)))
    
    
eda(df_train)


# ## Train Dictionary ##
# 
# Using gensims to train a dictionary of words available in the corpus

# In[ ]:


import re
import gensim
from gensim import corpora
from nltk.corpus import stopwords

words = re.compile(r"\w+",re.I)
stopword = stopwords.words('english')

def tokenize_questions(df):
    question_1_tokenized = []
    question_2_tokenized = []

    for q in df.question1.tolist():
        question_1_tokenized.append([i.lower() for i in words.findall(q) if i not in stopword])

    for q in df.question2.tolist():
        question_2_tokenized.append([i.lower() for i in words.findall(q) if i not in stopword])

    df["Question_1_tok"] = question_1_tokenized
    df["Question_2_tok"] = question_2_tokenized
    
    return df

def train_dictionary(df):
    
    questions_tokenized = df.Question_1_tok.tolist() + df.Question_2_tok.tolist()
    
    dictionary = corpora.Dictionary(questions_tokenized)
    dictionary.filter_extremes(no_below=5, no_above=0.5, keep_n=10000000)
    dictionary.compactify()
    
    return dictionary
    
df_train = tokenize_questions(df_train)
dictionary = train_dictionary(df_train)

print (df_train.columns)
print (df_train.shape)
print (len(dictionary.token2id))


# In[ ]:


def get_vectors(df, dictionary):
    
    question1_vec = [dictionary.doc2bow(text) for text in df.Question_1_tok.tolist()]
    question2_vec = [dictionary.doc2bow(text) for text in df.Question_2_tok.tolist()]
    
    question1_csc = gensim.matutils.corpus2csc(question1_vec, num_terms=len(dictionary.token2id))
    question2_csc = gensim.matutils.corpus2csc(question2_vec, num_terms=len(dictionary.token2id))
    
    return question1_csc.transpose(),question2_csc.transpose()


q1_csc, q2_csc = get_vectors(df_train, dictionary)

print (q1_csc.shape)
print (q2_csc.shape)


# In[ ]:


from sklearn.metrics.pairwise import cosine_similarity as cs

def get_cosine_similarity(q1_csc, q2_csc):
    cosine_sim = []
    for i,j in zip(q1_csc, q2_csc):
        sim = cs(i,j)
        cosine_sim.append(sim[0][0])
    
    return cosine_sim
    
cosine_sim = get_cosine_similarity(q1_csc, q2_csc)

print (len(cosine_sim))


# In[ ]:


from sklearn.ensemble import RandomForestClassifier as RFC
from sklearn.svm import SVC
from sklearn.ensemble import GradientBoostingClassifier as GBC
from sklearn.linear_model import LogisticRegression as LR
from sklearn.cross_validation import train_test_split
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import f1_score, confusion_matrix
from sklearn.pipeline import Pipeline

np.random.seed(10)

def train_rfc(X,y):
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.10, random_state=42)
    
    svm_models = [('svm', SVC(verbose=1, shrinking=False))]
    svm_pipeline = Pipeline(svm_models)
    svm_params = {'svm__kernel' : ['rbf'],
                  'svm__C' : [0.01,0.1,1],
                  'svm__gamma' :[0.1,0.2,0.4],
                  'svm__tol' :[0.001,0.01,0.1],
                  'svm__class_weight' : [{1:0.8,0:0.2}]}

    rfc_models = [('rfc', RFC())]
    rfc_pipeline = Pipeline(rfc_models)
    rfc_params = {'rfc__n_estimators' : [20],
                  'rfc__max_depth' : [10],
                  'rfc__min_samples_leaf' : [100]}

    lr_models = [('lr', LR(verbose=1))]
    lr_pipeline = Pipeline(lr_models)
    lr_params = {'lr__C': [0.1, 0.01],
                 'lr__tol': [0.001,0.01],
                 'lr__max_iter': [200,400],
                 'lr__class_weight' : [{1:0.8,0:0.2}]}

    gbc_models = [('gbc', GBC(verbose=1))]
    gbc_pipeline = Pipeline(gbc_models)
    gbc_params = {'gbc__n_estimators' : [100,200, 400, 800],
                  'gbc__max_depth' : [40, 80, 160, 320],
                  'gbc__learning_rate' : [0.01,0.1]}

    grid = zip([svm_pipeline, rfc_pipeline, lr_pipeline, gbc_pipeline],
               [svm_params, rfc_params, lr_params, gbc_params])

    grid = zip([rfc_pipeline],
               [rfc_params])

    best_clf = None

    for model_pipeline, param in grid:
        temp = GridSearchCV(model_pipeline, param_grid=param, cv=4, scoring='f1')
        temp.fit(X_train, y_train)

        if best_clf is None:
            best_clf = temp
        else:
            if temp.best_score_ > best_clf.best_score_:
                best_clf = temp
    
    model_details = {}
    model_details["CV Accuracy"] = best_clf.best_score_
    model_details["Model Parameters"] = best_clf.best_params_
    model_details["Test Data Score"] = best_clf.score(X_test, y_test)
    model_details["F1 score"] = f1_score(y_test, best_clf.predict(X_test))
    model_details["Confusion Matrix"] = str(confusion_matrix(y_test, best_clf.predict(X_test)))
    
    return best_clf, model_details

X = np.array(cosine_sim).reshape(-1,1)
y = df_train.is_duplicate

clf, model_details = train_rfc(X,y)

print (model_details)

