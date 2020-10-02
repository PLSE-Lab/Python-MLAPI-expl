#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# # 01 Frame the Problem

# Problem statement link :https://www.kaggle.com/c/movie-review-sentiment-analysis-kernels-only/overview

# # 02 Obtain the data

# In[ ]:


import pandas as pd
import numpy as np


# In[ ]:


df1 = pd.read_csv('../input/movie-review-sentiment-analysis-kernels-only/sampleSubmission.csv', sep=",")
df2 = pd.read_csv('../input/movie-review-sentiment-analysis-kernels-only/train.tsv', sep="\t")
df3 = pd.read_csv('../input/movie-review-sentiment-analysis-kernels-only/test.tsv', sep="\t")


# # 03 Analyze The data

# In[ ]:


df1.head(5)


# In[ ]:


df2.head(5)


# In[ ]:


df3.head(5)


# In[ ]:


df2.info()


# In[ ]:


df3.info()


# In[ ]:


df2['Sentiment'].value_counts()


# In[ ]:


df2[df2['Sentiment']==1]['Phrase'].head(3)


# In[ ]:


df2.loc[df2.SentenceId ==2]


# In[ ]:


#regular expression 
import re 

#regular expression for the removal of name tags and the emoticons from tweets.
def process(Phrase):
    return " ".join(re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])", " ",Phrase.lower()).split())


# In[ ]:


#Applying the Process function to the given Train Data
df2['Newphrase']= df2['Phrase'].apply(process)


# In[ ]:


import matplotlib.pyplot as plt
import seaborn as sns


# In[ ]:


z = df2['Sentiment']
plt.hist(z, bins=5,color='Red')
plt.ylabel('count')
plt.xlabel('rating')
plt.show()


# # 04 Feature Enginnering

# In[ ]:


df2.head(5)


# In[ ]:


#df2.drop('Phrase',inplace=True,axis=1)


# In[ ]:


#df2.drop('PhraseId', inplace=True,axis=1)


# # 05 Model Selection

# In[ ]:


from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer, TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier


# In[ ]:


df2.info()


# # 1st model

# In[ ]:


count_vect = CountVectorizer(stop_words='english',ngram_range=(1,3),analyzer='word')
transformer = TfidfTransformer(norm='l2',sublinear_tf=True)


# In[ ]:


#splitting the data into random train and test subsets
x_train, x_test, y_train, y_test = train_test_split(df2["Newphrase"],df2["Sentiment"],
                                                    test_size = 0.2, random_state = 20)

x_train_counts = count_vect.fit_transform(x_train)
x_train_tfidf = transformer.fit_transform(x_train_counts)
x_test_counts = count_vect.transform(x_test)
x_test_tfidf = transformer.transform(x_test_counts)


# In[ ]:


print(x_train_counts.shape)
print(x_train_tfidf.shape)
print(x_test_counts.shape)
print(x_test_tfidf.shape)


# In[ ]:


from sklearn.linear_model import SGDClassifier

model = SGDClassifier(loss="modified_huber", penalty="l1")
model.fit(x_train_tfidf,y_train)
predictions = model.predict(x_test_tfidf)


# In[ ]:


from sklearn.metrics import precision_score,recall_score,f1_score, accuracy_score, confusion_matrix


# In[ ]:


accuracy_score(y_test,predictions)


# In[ ]:


f1_score(y_test,predictions, average ='micro')


# In[ ]:


recall_score(y_test,predictions, average = 'micro')


# In[ ]:


precision_score(y_test,predictions,average = 'micro')


# # 2nd model

# In[ ]:


#different classification modesls being used
from sklearn.svm import LinearSVC

model_svc = LinearSVC(C=2.0,max_iter=500,tol=0.0001,loss ='hinge')
model_svc.fit(x_train_counts,y_train)


# In[ ]:


predict_svc = model_svc.predict(x_test_counts)


# In[ ]:


f1_score(y_test,predict_svc, average = 'micro')


# In[ ]:


recall_score(y_test,predict_svc, average = 'micro')


# In[ ]:


accuracy_score(y_test,predict_svc)


# # 06 Tune the model

# In[ ]:


#optimizing parameters
from sklearn.model_selection import GridSearchCV


params = {"tfidf__ngram_range": [(1, 2), (1,3), (1,4)],
          "svc__C": [.01, .1, 1, 10, 100]}

clf = Pipeline([("tfidf", TfidfVectorizer(sublinear_tf=True)),
                ("svc", LinearSVC(loss='hinge'))])

gs = GridSearchCV(clf, params, verbose=4, n_jobs=-1)
gs.fit(x_train,y_train)
print("Best Estimator = ", gs.best_estimator_)
print("Best Score = ",gs.best_score_)


# In[ ]:


predicted = gs.predict(x_test)


# In[ ]:


predicted


# In[ ]:


f1_score(y_test,predicted, average = 'micro')


# In[ ]:


recall_score(y_test,predicted, average = 'micro')


# In[ ]:


precision_score(y_test,predicted, average = 'micro')


# In[ ]:


accuracy_score(y_test,predicted)


# # 07 Predict on new cases

# In[ ]:


df3['Newphrase']= df3['Phrase'].apply(process)


# In[ ]:


#df3.drop('Phrase',inplace=True,axis=1)


# In[ ]:


#df3.drop('PhraseId', inplace=True,axis=1)


# In[ ]:


df3.head(5)


# In[ ]:


predicted = gs.predict(df3['Newphrase'])


# In[ ]:


final_predict = pd.DataFrame(predicted,columns=['Sentiment'])
result = pd.DataFrame(df3['PhraseId'])
result = pd.concat([result,final_predict],axis=1)
result.to_csv('Submission.csv',index=False)


# In[ ]:


predicted


# In[ ]:


result


# In[ ]:


result['Sentiment'].value_counts()


# In[ ]:




