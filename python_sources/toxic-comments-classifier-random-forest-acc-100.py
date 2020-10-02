# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import pandas as pd
import matplotlib


train = pd.read_csv('C:\\Users\\Surya\\Desktop\\jigsaw-unintended-bias-in-toxicity-classification\\train.csv') 
train.head(5) 

test = pd.read_csv('C:\\Users\\Surya\\Desktop\\jigsaw-unintended-bias-in-toxicity-classification\\test.csv') 
test.head(5) 

train.shape
test.shape
df_n = pd.DataFrame(train)
df_n

df_new = pd.DataFrame(test)
df_new

data = df_n.append([df_new])
data.head()

traindata = data[['id', 'target', 'comment_text']]
traindata


comments = traindata[['comment_text']]
comments

comments.shape
traindata["target"].fillna(0, inplace=True)
var = traindata[traindata.id >= 7000000]
var
targetoutput = var["target"]
targetoutput.head()

targetcomment = var["comment_text"]
targetcomment.head()

from sklearn.model_selection import train_test_split

reviews = targetcomment

import numpy as np  
import re  
import nltk  
from sklearn.datasets import load_files  
nltk.download('stopwords')  
import pickle  
from nltk.corpus import stopwords  
from sklearn.feature_extraction.text import CountVectorizer  
vectorizer = CountVectorizer(max_features=45, min_df=5, max_df=0.7, stop_words=stopwords.words('english'))  
X = vectorizer.fit_transform(reviews).toarray()

from sklearn.feature_extraction.text import TfidfTransformer  
tfidfconverter = TfidfTransformer()  
X = tfidfconverter.fit_transform(X).toarray()

from sklearn.feature_extraction.text import TfidfVectorizer  
tfidfconverter = TfidfVectorizer(max_features=45, min_df=5, max_df=0.7, stop_words=stopwords.words('english'))  
X = tfidfconverter.fit_transform(reviews).toarray()  

y = targetoutput
y.head()

from sklearn.model_selection import train_test_split  
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

from sklearn.ensemble import RandomForestClassifier

classifier = RandomForestClassifier(n_estimators=1000, random_state=0)  
classifier.fit(X, y)

y_pred = classifier.predict(X_test) 

y_pred 
array([0., 0., 0., ..., 0., 0., 0.])


from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

print(confusion_matrix(y_test,y_pred))  
print(classification_report(y_test,y_pred))  
print(accuracy_score(y_test, y_pred))


[[19464]]
              precision    recall  f1-score   support

         0.0       1.00      1.00      1.00     19464

   micro avg       1.00      1.00      1.00     19464
   macro avg       1.00      1.00      1.00     19464
weighted avg       1.00      1.00      1.00     19464

1.0

Accuracy is 100% with random forest
and y_predictions are 0 denotes most of the data set contains less toxic comments 



