#!/usr/bin/env python
# coding: utf-8

# ## Preprocessing

# In[ ]:


import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import metrics
import seaborn as sns
sns.set()


# In[ ]:


import sqlite3

conn = sqlite3.connect('../input/database.sqlite')

data = pd.read_sql_query("select * from Reviews where Score !=3 order by Time", conn)
data.head()


# In[ ]:


# removing duplicates
data = data.drop_duplicates(subset={'ProductId', 'UserId', 'Score', 
                            'Text'}, keep='first')


# In[ ]:


positiveData = {
    "Text": [],
    "Sentiment": [],
    "Time": []
}

negativeData = {
    "Text": [],
    "Sentiment": [],
    "Time": []    
}

n_posRneg = 50000
n_negative = 0
n_positive = 0

for row in data.itertuples():
    
    if row[7] < 3:
        if n_negative < n_posRneg:
            positiveData['Text'].append(row[10])
            positiveData['Sentiment'].append(-1)
            positiveData['Time'].append(row[8])
            n_negative += 1
    else:
        if n_positive < n_posRneg:
            negativeData['Text'].append(row[10])
            negativeData['Sentiment'].append(1)
            negativeData['Time'].append(row[8])
            n_positive += 1

positiveData = pd.DataFrame(positiveData).sort_values(['Time'], axis=0)
negativeData = pd.DataFrame(negativeData).sort_values(['Time'], axis=0)


# In[ ]:


te = int(5000 * 7)
positiveTrain = positiveData[0:te]
positiveTest = positiveData[te:50000]
negativeTrain = negativeData[0:te]
negativeTest = negativeData[te:50000]


# In[ ]:


trainData = pd.concat([positiveTrain, negativeTrain], axis=0)
trainData = trainData.sort_values(['Time'], axis=0)
X_train = trainData['Text']
Y_train = trainData['Sentiment']

testData = pd.concat([positiveTest, negativeTest], axis=0)
testData = testData.sort_values(['Time'], axis=0)
X_test = testData['Text']
Y_test = testData['Sentiment']


# In[ ]:


from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
import re

snow = SnowballStemmer('english')
pre_processedTrain = []

i = 0
N = 70000
for sentence in X_train:    
    sentence = str(sentence)
    sentence = sentence.lower()
    clnr = re.compile('<.*?>') # for cleaning html tags
    sentence = re.sub(clnr, ' ', sentence)
    sentence = re.sub(r'[?|!|\'|"|#]', r'', sentence)
    sentence = re.sub(r'[.|,|)|(|\|/]', r' ', sentence) 
    
    words = [snow.stem(word) for word in sentence.split()              if word not in stopwords.words('english')]
    final_sentence = ''
    for word in words:
        final_sentence = final_sentence + word + ' '
    pre_processedTrain.append(final_sentence)
    print("{0:.2f} %".format(i/N*100), end='\r')
    i += 1
    
print(pre_processedTrain[0])


# In[ ]:



len(pre_processedTrain)


# ## Bag of words 

# In[ ]:


max_features = 5000

bow_model = CountVectorizer(max_features=max_features)
bow_dataTrain = bow_model.fit_transform(pre_processedTrain)
bow_dataTrain = bow_dataTrain.toarray()
bow_dataTrain[0]


# ### Training logistic Regression ( L1 - Regularization )

# #### Grid Search

# In[ ]:


from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
logistic_ = LogisticRegression(penalty='l1')

grid_parameters = {'C':[10**-2, 10**-1, 1, 10, 100]}
gridSearchModel = GridSearchCV(logistic_, grid_parameters, cv=5)
gridSearchModel.fit(bow_dataTrain, Y_train)
gridSearchModel.best_params_


# In[ ]:


classifier = LogisticRegression(penalty='l1', C=gridSearchModel.best_params_['C'])
classifier.fit(bow_dataTrain, Y_train)
tr_score = classifier.score(bow_dataTrain, Y_train)
print(tr_score)


# In[ ]:


pre_processedTest = []

i = 0
N = 30000
for sentence in X_test:    
    sentence = str(sentence)
    sentence = sentence.lower()
    clnr = re.compile('<.*?>') # for cleaning html tags
    sentence = re.sub(clnr, ' ', sentence)
    sentence = re.sub(r'[?|!|\'|"|#]', r'', sentence)
    sentence = re.sub(r'[.|,|)|(|\|/]', r' ', sentence) 
    
    words = [snow.stem(word) for word in sentence.split()              if word not in stopwords.words('english')]
    final_sentence = ''
    for word in words:
        final_sentence = final_sentence + word + ' '
    pre_processedTest.append(final_sentence)
    print("{0:.2f} %".format(i/N*100), end='\r')
    i += 1
    
print(pre_processedTest[0])

bow_dataTest = bow_model.transform(pre_processedTest).toarray()


# In[ ]:


len(pre_processedTest)


# In[ ]:


pred = classifier.predict(bow_dataTest)

accuracy = metrics.accuracy_score(Y_test, pred)
pre = metrics.precision_score(Y_test, pred)
rec = metrics.recall_score(Y_test, pred)
f1 = metrics.f1_score(Y_test, pred)


# In[ ]:


models_performence = {
    'Model':['Logistic Regression'],
    'Vectorizer': ['BoW'],
    'Regulizer': ['l1'],
    'best parameter Search Technique': ['Grid Search'],
    'Training Accuracy':[tr_score*100],
    'Test Accuracy':[accuracy*100], 
    'Precision':[pre*100],
    'Recall':[rec*100],
    'F1-Score':[f1*100]
}


# In[ ]:


columns = ["Model", "Vectorizer", "Regulizer", "Training Accuracy", "best parameter Search Technique", "Test Accuracy",
          "Precision", "Recall", "F1-Score"]
pd.DataFrame(models_performence, columns=columns)


# In[ ]:


from sklearn.metrics import confusion_matrix
cm = confusion_matrix(Y_test, pred)
print(cm)
sns.heatmap(cm)


# #### Random Search

# In[ ]:


from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import uniform

random_parameters = {'C':uniform()}
randomSearchModel = RandomizedSearchCV(logistic_, random_parameters, cv=5)
randomSearchModel.fit(bow_dataTrain, Y_train)
randomSearchModel.best_params_


# In[ ]:


classifier = LogisticRegression(penalty='l1', C=randomSearchModel.best_params_['C'])
classifier.fit(bow_dataTrain, Y_train)
tr_score = classifier.score(bow_dataTrain, Y_train)
print(tr_score)


# In[ ]:


pred = classifier.predict(bow_dataTest)

accuracy = metrics.accuracy_score(Y_test, pred)
pre = metrics.precision_score(Y_test, pred)
rec = metrics.recall_score(Y_test, pred)
f1 = metrics.f1_score(Y_test, pred)


# In[ ]:


models_performence['Model'].append('Logistic Regression')
models_performence['Vectorizer'].append('BoW')
models_performence['Regulizer'].append('l1')
models_performence['best parameter Search Technique'].append('Random Search')
models_performence['Training Accuracy'].append(tr_score*100)
models_performence['Test Accuracy'].append(accuracy*100)
models_performence['Precision'].append(pre*100)
models_performence['Recall'].append(rec*100)
models_performence['F1-Score'].append(f1*100)


# In[ ]:


columns = ["Model", "Vectorizer", "Regulizer", "Training Accuracy", "best parameter Search Technique", "Test Accuracy",
          "Precision", "Recall", "F1-Score"]
pd.DataFrame(models_performence, columns=columns)


# In[ ]:


from sklearn.metrics import confusion_matrix
cm = confusion_matrix(Y_test, pred)
print(cm)
sns.heatmap(cm, annot=True, fmt="d")


# ### Training Logistic Regression ( L2 - regularization)

# #### Grid Search

# In[ ]:


from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
logistic_ = LogisticRegression(penalty='l2', max_iter=130)

grid_parameters = {'C':[10**-2, 10**-1, 1, 10, 100]}
gridSearchModel = GridSearchCV(logistic_, grid_parameters, cv=5)
gridSearchModel.fit(bow_dataTrain, Y_train)
gridSearchModel.best_params_


# In[ ]:


classifier = LogisticRegression(penalty='l2', C=gridSearchModel.best_params_['C'])
classifier.fit(bow_dataTrain, Y_train)
tr_score = classifier.score(bow_dataTrain, Y_train)
print(tr_score)


# In[ ]:


pred = classifier.predict(bow_dataTest)

accuracy = metrics.accuracy_score(Y_test, pred)
pre = metrics.precision_score(Y_test, pred)
rec = metrics.recall_score(Y_test, pred)
f1 = metrics.f1_score(Y_test, pred)


# In[ ]:


models_performence['Model'].append('Logistic Regression')
models_performence['Vectorizer'].append('BoW')
models_performence['Regulizer'].append('l2')
models_performence['best parameter Search Technique'].append('Grid Search')
models_performence['Training Accuracy'].append(tr_score*100)
models_performence['Test Accuracy'].append(accuracy*100)
models_performence['Precision'].append(pre*100)
models_performence['Recall'].append(rec*100)
models_performence['F1-Score'].append(f1*100)


# In[ ]:


columns = ["Model", "Vectorizer", "Regulizer", "Training Accuracy", "best parameter Search Technique", "Test Accuracy",
          "Precision", "Recall", "F1-Score"]
pd.DataFrame(models_performence, columns=columns)


# In[ ]:


from sklearn.metrics import confusion_matrix
cm = confusion_matrix(Y_test, pred)
print(cm)
sns.heatmap(cm, annot=True, fmt="d")


# #### Random Search

# In[ ]:


random_parameters = {'C':uniform()}
randomSearchModel = RandomizedSearchCV(logistic_, random_parameters, cv=5)
randomSearchModel.fit(bow_dataTrain, Y_train)
randomSearchModel.best_params_


# In[ ]:


classifier = LogisticRegression(penalty='l2', C=randomSearchModel.best_params_['C'])
classifier.fit(bow_dataTrain, Y_train)
tr_score = classifier.score(bow_dataTrain, Y_train)
print(tr_score)


# In[ ]:


pred = classifier.predict(bow_dataTest)

accuracy = metrics.accuracy_score(Y_test, pred)
pre = metrics.precision_score(Y_test, pred)
rec = metrics.recall_score(Y_test, pred)
f1 = metrics.f1_score(Y_test, pred)


# In[ ]:


models_performence['Model'].append('Logistic Regression')
models_performence['Vectorizer'].append('BoW')
models_performence['Regulizer'].append('l2')
models_performence['best parameter Search Technique'].append('Random Search')
models_performence['Training Accuracy'].append(tr_score*100)
models_performence['Test Accuracy'].append(accuracy*100)
models_performence['Precision'].append(pre*100)
models_performence['Recall'].append(rec*100)
models_performence['F1-Score'].append(f1*100)


# In[ ]:


columns = ["Model", "Vectorizer", "Regulizer", "Training Accuracy", "best parameter Search Technique", "Test Accuracy",
          "Precision", "Recall", "F1-Score"]
pd.DataFrame(models_performence, columns=columns)


# In[ ]:


from sklearn.metrics import confusion_matrix
cm = confusion_matrix(Y_test, pred)
print(cm)
sns.heatmap(cm, annot=True, fmt="d")


#  ## Tf - IDF

# In[ ]:


max_features = 5000

tf_idf_model = TfidfVectorizer(max_features=max_features)
tf_idf_dataTrain = tf_idf_model.fit_transform(pre_processedTrain)
tf_idf_dataTrain = tf_idf_dataTrain.toarray()
tf_idf_dataTrain[0]


# ### Training Logistic Regression ( L1 - Regularizer )

# #### Grid Search

# In[ ]:


logistic_ = LogisticRegression(penalty='l1', max_iter=130)

grid_parameters = {'C':[10**-2, 10**-1, 1, 10, 100]}
gridSearchModel = GridSearchCV(logistic_, grid_parameters, cv=5)
gridSearchModel.fit(tf_idf_dataTrain, Y_train)
gridSearchModel.best_params_


# In[ ]:


classifier = LogisticRegression(penalty='l1', C=gridSearchModel.best_params_['C'])
classifier.fit(tf_idf_dataTrain, Y_train)
tr_score = classifier.score(tf_idf_dataTrain, Y_train)
print(tr_score)


# In[ ]:


tf_idf_dataTest = tf_idf_model.transform(pre_processedTest).toarray()


# In[ ]:


pred = classifier.predict(tf_idf_dataTest)

accuracy = metrics.accuracy_score(Y_test, pred)
pre = metrics.precision_score(Y_test, pred)
rec = metrics.recall_score(Y_test, pred)
f1 = metrics.f1_score(Y_test, pred)


# In[ ]:


models_performence['Model'].append('Logistic Regression')
models_performence['Vectorizer'].append('Tf-IDF')
models_performence['Regulizer'].append('l1')
models_performence['best parameter Search Technique'].append('Grid Search')
models_performence['Training Accuracy'].append(tr_score*100)
models_performence['Test Accuracy'].append(accuracy*100)
models_performence['Precision'].append(pre*100)
models_performence['Recall'].append(rec*100)
models_performence['F1-Score'].append(f1*100)


# In[ ]:


columns = ["Model", "Vectorizer", "Regulizer", "Training Accuracy", "best parameter Search Technique", "Test Accuracy",
          "Precision", "Recall", "F1-Score"]
pd.DataFrame(models_performence, columns=columns)


# In[ ]:


from sklearn.metrics import confusion_matrix
cm = confusion_matrix(Y_test, pred)
print(cm)
sns.heatmap(cm, annot=True, fmt="d")


# #### Random Search

# In[ ]:


random_parameters = {'C':uniform()}
randomSearchModel = RandomizedSearchCV(logistic_, random_parameters, cv=5)
randomSearchModel.fit(tf_idf_dataTrain, Y_train)
randomSearchModel.best_params_


# In[ ]:


classifier = LogisticRegression(penalty='l1', C=randomSearchModel.best_params_['C'])
classifier.fit(tf_idf_dataTrain, Y_train)
tr_score = classifier.score(tf_idf_dataTrain, Y_train)
print(tr_score)


# In[ ]:


pred = classifier.predict(tf_idf_dataTest)

accuracy = metrics.accuracy_score(Y_test, pred)
pre = metrics.precision_score(Y_test, pred)
rec = metrics.recall_score(Y_test, pred)
f1 = metrics.f1_score(Y_test, pred)


# In[ ]:


models_performence['Model'].append('Logistic Regression')
models_performence['Vectorizer'].append('Tf-IDF')
models_performence['Regulizer'].append('l1')
models_performence['best parameter Search Technique'].append('Random Search')
models_performence['Training Accuracy'].append(tr_score*100)
models_performence['Test Accuracy'].append(accuracy*100)
models_performence['Precision'].append(pre*100)
models_performence['Recall'].append(rec*100)
models_performence['F1-Score'].append(f1*100)


# In[ ]:


columns = ["Model", "Vectorizer", "Regulizer", "Training Accuracy", "best parameter Search Technique", "Test Accuracy",
          "Precision", "Recall", "F1-Score"]
pd.DataFrame(models_performence, columns=columns)


# In[ ]:


from sklearn.metrics import confusion_matrix
cm = confusion_matrix(Y_test, pred)
print(cm)
sns.heatmap(cm, annot=True, fmt="d")


# ### Training Logistic Regression ( L2 - regularizer )

# #### Grid Search

# In[ ]:


logistic_ = LogisticRegression(penalty='l2', max_iter=130)

grid_parameters = {'C':[10**-2, 10**-1, 1, 10, 100]}
gridSearchModel = GridSearchCV(logistic_, grid_parameters, cv=5)
gridSearchModel.fit(tf_idf_dataTrain, Y_train)
gridSearchModel.best_params_


# In[ ]:


classifier = LogisticRegression(penalty='l2', C=gridSearchModel.best_params_['C'])
classifier.fit(tf_idf_dataTrain, Y_train)
tr_score = classifier.score(tf_idf_dataTrain, Y_train)
print(tr_score)


# In[ ]:


pred = classifier.predict(tf_idf_dataTest)

accuracy = metrics.accuracy_score(Y_test, pred)
pre = metrics.precision_score(Y_test, pred)
rec = metrics.recall_score(Y_test, pred)
f1 = metrics.f1_score(Y_test, pred)


# In[ ]:


models_performence['Model'].append('Logistic Regression')
models_performence['Vectorizer'].append('Tf-IDF')
models_performence['Regulizer'].append('l2')
models_performence['best parameter Search Technique'].append('Grid Search')
models_performence['Training Accuracy'].append(tr_score*100)
models_performence['Test Accuracy'].append(accuracy*100)
models_performence['Precision'].append(pre*100)
models_performence['Recall'].append(rec*100)
models_performence['F1-Score'].append(f1*100)


# In[ ]:


columns = ["Model", "Vectorizer", "Regulizer", "Training Accuracy", "best parameter Search Technique", "Test Accuracy",
          "Precision", "Recall", "F1-Score"]
pd.DataFrame(models_performence, columns=columns)


# In[ ]:


from sklearn.metrics import confusion_matrix
cm = confusion_matrix(Y_test, pred)
print(cm)
sns.heatmap(cm, annot=True, fmt="d")


# #### Random Search

# In[ ]:


random_parameters = {'C':uniform()}
randomSearchModel = RandomizedSearchCV(logistic_, random_parameters, cv=5)
randomSearchModel.fit(tf_idf_dataTrain, Y_train)
randomSearchModel.best_params_


# In[ ]:


classifier = LogisticRegression(penalty='l2', C=randomSearchModel.best_params_['C'])
classifier.fit(tf_idf_dataTrain, Y_train)
tr_score = classifier.score(tf_idf_dataTrain, Y_train)
print(tr_score)


# In[ ]:


pred = classifier.predict(tf_idf_dataTest)

accuracy = metrics.accuracy_score(Y_test, pred)
pre = metrics.precision_score(Y_test, pred)
rec = metrics.recall_score(Y_test, pred)
f1 = metrics.f1_score(Y_test, pred)


# In[ ]:


models_performence['Model'].append('Logistic Regression')
models_performence['Vectorizer'].append('Tf-IDF')
models_performence['Regulizer'].append('l2')
models_performence['best parameter Search Technique'].append('Random Search')
models_performence['Training Accuracy'].append(tr_score*100)
models_performence['Test Accuracy'].append(accuracy*100)
models_performence['Precision'].append(pre*100)
models_performence['Recall'].append(rec*100)
models_performence['F1-Score'].append(f1*100)


# In[ ]:


columns = ["Model", "Vectorizer", "Regulizer", "Training Accuracy", "best parameter Search Technique", "Test Accuracy",
          "Precision", "Recall", "F1-Score"]
pd.DataFrame(models_performence, columns=columns)


# In[ ]:


from sklearn.metrics import confusion_matrix
cm = confusion_matrix(Y_test, pred)
print(cm)
sns.heatmap(cm, annot=True, fmt="d")


# # Summary

# In[ ]:


columns = ["Model", "Vectorizer", "Regulizer", "Training Accuracy", "best parameter Search Technique", "Test Accuracy",
          "Precision", "Recall", "F1-Score"]
pd.DataFrame(models_performence, columns=columns)


# In[ ]:




