#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import nltk
import seaborn as sns
from nltk.corpus import stopwords
from nltk import word_tokenize, ngrams
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.


# In[ ]:


test = "../input/test.csv"
train = "../input/train.csv"
testData = pd.read_csv(test,sep=",")
trainData = pd.read_csv(train,sep=",")
print(testData.info())


# In[ ]:


print(trainData.info())


# In[ ]:


print(trainData.groupby('is_duplicate').size())


# In[ ]:


print(trainData.head(10))


# In[ ]:



def get_feature_mat(train):
	#feature engineering in this funciton is applied to both test and train
	df 	= pd.read_csv("../input/"+train)
	return(df)

train, test = [get_feature_mat(train) for train in ['train.csv', 'test.csv']]

print(train.columns)
print(test.columns)

v = np.array(train['is_duplicate'])
mean = np.mean(v)
std = np.std(v)
print("mean", mean)
print("std", std)


# In[ ]:


trainData['is_duplicate'].hist()


# In[ ]:


import string
remove_punctuation_map = dict((ord(char), None) for char in string.punctuation)

sw = stopwords.words("english")
print(sw.index('what'))


# In[ ]:


from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer(stop_words="english")
question2 = []
for s in trainData['question2']: 
    if (type(s)!=float):
        question2.append(s.lower())
    else:    
        print(s)
        question2.append("")


# In[ ]:


#Stemming
from nltk.stem.snowball import SnowballStemmer
stemmer = SnowballStemmer("english")
words =""
question1_stem =[]
question2_stem =[]

for q in trainData['question1']:
    q=q.translate(remove_punctuation_map)
    stemmed = [stemmer.stem(w) for w in q.split() if w not in sw]
    if stemmed not in sw:
        words = " ".join(stemmed)
        question1_stem.append(words)    
for q in question2:
    q=q.translate(remove_punctuation_map)
    stemmed = [stemmer.stem(w) for w in q.split()]
    if stemmed not in sw:
        words = " ".join(stemmed)
        question2_stem.append(words)    
    
print(len(question1_stem))
print(len(question2_stem))        


# In[ ]:


print(question1_stem[:5])


# In[ ]:


print(question2_stem[:5])


# In[ ]:


#for x in range (1,40):
print(question1_stem[9545])
print(question2_stem[9545])

#print(question1_stem[35641])
#print(question2_stem[35641])

print(question1_stem[42947])
print(question2_stem[42947])

print(question1_stem[44344])
print(question2_stem[44344])


# In[ ]:


calculated_values = []
cnt = -1
#for x in range(0,404290):
for q1,q2 in zip (question1_stem, question2_stem):
    cnt =cnt +1
    #print(trainData['is_duplicate'][cnt])
    try:
        tfidf = vectorizer.fit_transform([q1, q2])
        calculated_values.append(((tfidf * tfidf.T).A)[0,1])
    except ValueError:
        print('Index with empty values or only root values',cnt)
        calculated_values.append(trainData['is_duplicate'][cnt])

print(len(calculated_values))    
print(calculated_values[:6])


# In[ ]:


# Two columns
len(trainData['is_duplicate'])
new_train_data = pd.DataFrame({"calculated":calculated_values,"actual":trainData['is_duplicate']})

print(type(new_train_data))
print(len(new_train_data))


# In[ ]:


from sklearn import model_selection
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

X = new_train_data['calculated'].values.reshape(-1,1)
y = new_train_data['actual']


# In[ ]:


print(new_train_data.describe())


# In[ ]:


from pandas.tools.plotting import scatter_matrix
scatter_matrix(new_train_data)


# In[ ]:


#split = int(0.75*1000)
#X_train = X[0:split]
#X_test  = X[split:]
#y_train = y[0:split]
#y_test  = y[split:]

validation_size = 0.20
seed = 7
X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(X, y, test_size=validation_size, random_state=seed)


# In[ ]:


#clf = GaussianNB()
#clf.fit(X_train,y_train)

#Test
seed = 7
scoring = 'accuracy'

# Various algos
models = []
models.append(('LR', LogisticRegression()))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier()))
models.append(('NB', GaussianNB()))
#models.append(('SVM', SVC()))
# evaluate each model in turn
results = []
names = []
for name, model in models:
	kfold = model_selection.KFold(n_splits=10, random_state=seed)
	cv_results = model_selection.cross_val_score(model, X_train, Y_train, cv=kfold, scoring=scoring)
	results.append(cv_results)
	names.append(name)
	msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
	print(msg)


# In[ ]:


question1 = []
for s in testData['question1'][15:22]: 
    if (type(s)!=float):
        question1.append(s.lower())
    else:    
        print(s)
        question1.append("")
question2 = []
for s in testData['question2'][15:22]: 
    if (type(s)!=float):
        question2.append(s.lower())
    else:    
        print(s)
        question2.append("")


# In[ ]:


#print(clf.predict(0.67))
#print(clf.predict(0.17))
#print(clf.predict(0.47))
question1_stem =[]
question2_stem =[]

for q in question1:
    q=q.translate(remove_punctuation_map)
    stemmed = [stemmer.stem(w) for w in q.split() if w not in sw]
    if stemmed not in sw:
        words = " ".join(stemmed)
        question1_stem.append(words)    
for q in question2:
    q=q.translate(remove_punctuation_map)
    stemmed = [stemmer.stem(w) for w in q.split()]
    if stemmed not in sw:
        words = " ".join(stemmed)
        question2_stem.append(words)    
    
print(len(question1_stem))
print(len(question2_stem))        


# In[ ]:


calculated_values = []
cnt = -1

for q1,q2 in zip (question1_stem, question2_stem):
    cnt =cnt +1
    try:
        tfidf = vectorizer.fit_transform([q1, q2])
        calculated_values.append(((tfidf * tfidf.T).A)[0,1])
    except ValueError:
        print('Index with empty values or only root values',cnt)
        calculated_values.append(1.0)

print(len(calculated_values))    
print(calculated_values[:6])


# In[ ]:


print(testData.info())


# In[ ]:


clf = DecisionTreeClassifier()
clf.fit(X_train, Y_train)


# In[ ]:


clf = DecisionTreeClassifier()
clf.fit(X_train, Y_train)
test_id = testData['test_id'][15:22]
calc = pd.DataFrame(calculated_values)
preds = clf.predict(calc.values.reshape(-1,1))

print(len(preds))
print(preds[:10])
out_df = pd.DataFrame({"test_id":test_id, "is_duplicate":preds})
print(out_df[:15])
file = out_df.to_csv("isduplicate_predicted_final5.csv", index=False)

