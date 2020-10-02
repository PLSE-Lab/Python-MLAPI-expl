from __future__ import division
import os
import pip
print (pip.__version__)
print (os.getcwd())
#Kaggle

#Spam Detection Challenge

#two columns 1. v1 and 2. v2

#v1 ~ flag (ham/spam)
#v2 ~ actual description/comment/sms

#data from
#https://www.kaggle.com/uciml/sms-spam-collection-dataset
#data contains invalid chars
#Use encoding argument for failsafe

from future.utils import iteritems
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer,CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from wordcloud import WordCloud,STOPWORDS
import matplotlib.pyplot as plt


data = pd.read_csv('../input/spam.csv',encoding='ISO-8859-1')
data.drop(['Unnamed: 2','Unnamed: 3','Unnamed: 4'],axis=1,inplace=True)
print (data.head(10))

data.columns = ['flags','messages']

#create binary labels

data['b_labels'] = data['flags'].map({'ham':0,'spam':1})
Y = data['b_labels'].as_matrix()

#Creating features
tfidf = TfidfVectorizer(decode_error='ignore')
X = tfidf.fit_transform(data['messages'])

countVectorizer = CountVectorizer(decode_error='ignore')
_X = countVectorizer.fit_transform(data['messages'])

#split the data
trainX,testX,trainY,testY = train_test_split(_X,Y,test_size=0.33)

#train the model, print scores

model = MultinomialNB()
model.fit(trainX,trainY)
print ("train score:", model.score(trainX,trainY))
print ("test score:", model.score(testX,testY))

#wordclouds
data['new_msgs'] = data['messages'].str.lower()

df_ham = data[data['b_labels']==0]['new_msgs']
corpus_ham = " ".join(c for c in df_ham)

df_spam = data[data['b_labels']==1]['new_msgs']
corpus_spam = " ".join(c for c in df_spam)


def wordcld(corpus):
    height = 600
    width = 800
    max_words = 300
    wc = WordCloud(width=width,height=height,max_words=max_words)
    wc.generate(corpus)
    plt.imshow(wc)
    plt.title('word cloud')
    plt.axis('off')
    plt.show()
    
wordcld(corpus_ham)
wordcld(corpus_spam)

data['predictions']=model.predict(_X)

#Check False-positives and False-negatives

#false-negatives
sneaky_spam = data[(data['predictions']==0) & (data['b_labels']==1)]['messages']
for msg in sneaky_spam:
    print (msg)
    
#false-postives
not_actually_spam = data[(data['predictions']==1) & (data['b_labels']==0)]['messages']
for msg in not_actually_spam:
    print (msg)
    
    



























