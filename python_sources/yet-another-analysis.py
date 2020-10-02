#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import datetime as dt
import re
import matplotlib as plt
import seaborn as sns
import string

import sklearn.model_selection as skcv
from sklearn import preprocessing as pp
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score

from nltk.stem import SnowballStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier


# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

def parser(x):
    t = re.findall(r"[\w]+",x)
    t[1] = t[1][:3]
    t.pop(0)
    t1 = " ".join(t)
    return dt.datetime.strptime(t1, "%b %d %Y")
    
botsDF = pd.read_csv("H:\Desktop\Python\Kaggle\Deep NLP\Sheet_1.csv", usecols=['response_id','class', 'response_text'])#, encoding="latin1", parse_dates=['Date'], date_parser = parser, dayfirst = True, index_col=[0], converters={'Longitute':np.float64})
resumeDF = pd.read_csv("H:\Desktop\Python\Kaggle\Deep NLP\Sheet_2.csv" , encoding="latin1") #, parse_dates=['Date'], date_parser = parser, dayfirst = True, index_col=[0], converters={'Longitute':np.float64})

stopwords = pd.read_csv("H:\Desktop\Python\edx AI\P5 - NLP\stopwords.en.txt", names=['words'])
stops = set(stopwords['words'].tolist())

def makeCorpus(txt):
    responses = txt.str.lower().str.split()
    
    refined = []
    
    for sentence in responses:
        refined.append(" ".join([word.strip(string.punctuation) for word in sentence if word not in stops]))
    
    sno = SnowballStemmer('english')
    r = [" ".join([sno.stem(x) for x in sentence.split(" ")]) for sentence in refined]
    return r

botsDF['corpus'] = makeCorpus(botsDF['response_text'])
resumeDF['corpus'] = makeCorpus(resumeDF['resume_text'])

botsDF.groupby('class').corpus.count().plot.bar()
      
cnt = CountVectorizer()
X = cnt.fit_transform(botsDF['corpus'])

X_train, X_test, Y_train, Y_test = skcv.train_test_split(X,botsDF['class'],test_size=0.4, random_state=1)
cv = skcv.KFold(n_splits=5, random_state=0)

def EvaluateModel(model, params, m_desc):
    """
    print("")
    print("Working for %s. Please wait ...." % m_desc)
    sTime = time.clock()
    """    
    gs = GridSearchCV(estimator=model, param_grid=params, cv=cv, n_jobs=1)
    
    gs.fit(X_train, Y_train)
    """
    print("Best parameters for %s is %r" % (m_desc, gs.best_params_))
    """
    print("Best parameters for %s are %r " % (m_desc, gs.best_params_))
    
    tr_score = gs.best_score_
    """
    print("Best training score is %f" % tr_score)
    """
    y_true, y_pred = Y_test, gs.predict(X_test)
    ts_score = accuracy_score(y_true, y_pred)
    """    
    print("Test score for %s is %f " %(m_desc, ts_score))
    eTime = time.clock() - sTime
    print("Paramter(s) tuned for %s in %d min(s) %d sec(s)" % (m_desc, int(eTime/60), int(eTime%60) ))   
    print("")
    """
    return m_desc, tr_score, ts_score

c = [0.1, 1, 3]
d = [4,5,6]
g = [0.1,1]

models = {}

c = [0.1, 0.5, 1, 5, 10, 50, 100]
m, tr, ts = EvaluateModel(SVC(kernel='linear'), dict(C=c),"svm_linear")    
models[m] = [m, tr, ts]


# SVC with Polynomial Kernel
# Params
c = [0.1, 1, 3]
d = [4,5,6]
g = [0.1,1]
m, tr, ts = EvaluateModel(SVC(kernel='poly'), dict(C=c,degree=d,gamma=g),"svm_polynomial")    
models[m] = [m, tr, ts]


# SVC with RBF Kernel
# Params
C = [0.1, 0.5, 1, 5, 10, 50, 100] 
g = [0.1, 0.5, 1, 3, 6, 10]
m, tr, ts = EvaluateModel(SVC(kernel='rbf'), dict(C=c,gamma=g),"svm_rbf")    
models[m] = [m, tr, ts]


#Logistic Regression
# Params
C = [0.1, 0.5, 1, 5, 10, 50, 100]
m, tr, ts = EvaluateModel(LogisticRegression(), dict(C=C),"logistic")    
models[m] = [m, tr, ts]


#K Nearest Neighbors
# Params
n_neighbors = np.linspace(1, 25, 1)
leaf_size = np.linspace(5, 60, 5)
m, tr, ts = EvaluateModel(KNeighborsClassifier(), dict(n_neighbors=n_neighbors,leaf_size=leaf_size),"knn")    
models[m] = [m, tr, ts]


#Decision Tree Classifier
# Params
max_depth = np.linspace(1, 50, 1)
min_samples_split = np.linspace(2, 10, 5, dtype=int)
m , tr, ts = EvaluateModel(DecisionTreeClassifier(), dict(max_depth=max_depth),"decision_tree")    
models[m] = [m, tr, ts]


#Random Forest Classifier
# Params
max_depth = np.linspace(1, 50, 50) 
max_features = np.linspace(5,15,10,dtype=int)
min_samples_split = np.linspace(2, 10, 5, dtype=int)
m , tr, ts = EvaluateModel(RandomForestClassifier(), dict(max_depth=max_depth, max_features=max_features, min_samples_split=min_samples_split),"random_forest")    
models[m] = [m, tr, ts]

