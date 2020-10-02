#!/usr/bin/env python
# coding: utf-8

# * Import Libraries 

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import pyplot
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')

# Warnings
import warnings
warnings.filterwarnings('ignore')

import sklearn 
import nltk
from nltk.tokenize import word_tokenize
from nltk.tokenize import sent_tokenize
from sklearn.model_selection import train_test_split
from nltk.corpus import stopwords
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import accuracy_score
from sklearn.feature_extraction.text import CountVectorizer
from nltk.tokenize import TreebankWordTokenizer
from sklearn.feature_extraction.text import TfidfTransformer 
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from nltk.stem import PorterStemmer
from collections import Counter
from wordcloud import WordCloud


# * Read in the Data and check first 5 elements

# In[ ]:


data = pd.read_csv("../input/sms-spam-collection-dataset/spam.csv",encoding='latin-1')
data.head()


# * Last columns seem to be unnecessary, check if there are values in there.

# In[ ]:


Sum = data.isnull().sum()
Percentage = (data.isnull().sum()/data.isnull().count())
values = pd.DataFrame([Sum,Percentage])
print(values)


# * More than 99% empty decide to remove columns 
# * Rename columns 
# * Other columns have no missing values 

# In[ ]:


data = pd.read_csv("../input/sms-spam-collection-dataset/spam.csv",encoding='latin-1')
data.drop(["Unnamed: 2", "Unnamed: 3", "Unnamed: 4"], axis=1, inplace = True)
data.columns =['label','text']

data.head()


# In[ ]:


colors = ['#ff9999','#66b3ff']
data["label"].value_counts().plot(kind = 'pie',colors = colors ,explode = (0.1,0),autopct = '%1.1f%%')


# 13.4% of dataset are spam messages, other 86.6% are not

# * Most popular "ham" text: 'Sorry, I'll call later', recurse 30 times.
# * Most popular "spam" text= 'Please call our customer service representative ...', recurse 4 times.

# TODO: Word Cloud 

# In[ ]:


stop_words = set(stopwords.words('english'))
spam_text = data.loc[data['label'] == 'spam']
ham_text = data.loc[data['label'] == 'ham']

count_Ham = Counter(" ".join(data[data['label']=='ham']["text"]).split()).most_common(100)
common_Ham = pd.DataFrame.from_dict(count_Ham)[0]
common_ham = common_Ham.str.cat(sep=' ')

count_Spam = Counter(" ".join(data[data['label']=='spam']["text"]).split()).most_common(100)
common_Spam = pd.DataFrame.from_dict(count_Spam)[0]
common_spam = common_Spam.str.cat(sep=' ')

wordcloud = WordCloud(stopwords=stop_words,background_color = "white")
wordcloud.generate(common_ham)
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()


wordcloud = WordCloud(stopwords=stop_words,background_color = "white")
wordcloud.generate(common_spam)
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()


# In[ ]:


data.groupby("label").describe()


# * Add variable length 
# * Change spam to dummy -> spam = 1, ham = 0

# In[ ]:


data['length'] = data['text'].apply(len)
data =pd.get_dummies(data, columns=['label'], prefix = 'Dummy' ,drop_first = True)


# remove stopwords 

# In[ ]:


# stop  = stopwords.words('english')
# data['text'].apply(lambda x: [item for item in x if item not in stop])
# data.head()


# In[ ]:





# In[ ]:


all_sent = []
for text in data.text:
    all_sent.append(text.lower())

common_sent = nltk.FreqDist(all_sent).most_common(10)
display(common_sent)


# In[ ]:


X=data['length'].values[:,None]
y= data['Dummy_spam']
X_train,X_test,y_train,y_test=train_test_split(X,y)


# In[ ]:


plt.style.use('seaborn-pastel')
data.hist(column='length',by='Dummy_spam',figsize=(10,5), bins=100, label = ("Ham","Spam") )
plt.xlim(-40,800)
plt.ioff()


# In[ ]:


models = []
models.append(['LR', LogisticRegression(solver='lbfgs')])
models.append(['SVM', svm.SVC(gamma='auto')])
models.append(['RF', RandomForestClassifier(n_estimators=1000, max_depth=10)])
models.append(['NN', MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(150, 10))])
models.append(['KNN', KNeighborsClassifier()])
models.append(['DTC', DecisionTreeClassifier()])
models.append(['MNB', MultinomialNB(alpha=0.2)])
models.append(['ABC', AdaBoostClassifier(n_estimators=100)])
print('Done')


# 1. Classification based on length of text

# In[ ]:


results = []
for name, model in models:
    wine_model = model
    wine_model.fit(X_train, y_train)
    pred = wine_model.predict(X_test)
    acc = accuracy_score(y_test, pred)
    precision = precision_score(y_test, pred, average=None)
    recall = recall_score(y_test, pred, average= None)
    error_Rate = 1- acc
    cm = pd.DataFrame(metrics.confusion_matrix(y_test,pred),index=['ham','spam'],columns=['ham','spam'])
    print('Model tested: {}'.format(name))
    print('Confusion Matrix')
    print(cm)
    print('Accuracy= {}'.format(acc))
    print('Error Rate= {}'.format(error_Rate))
    print('Recall Rate= {}'.format(recall))
    print("Precision Rate: {}".format(precision))
    print(metrics.classification_report(y_test,pred))
    print()
    results.append([name, precision])


# In[ ]:


spam_text = data[data["Dummy_spam"] == 1]["text"]
ham_text = data[data["Dummy_spam"] == 0]["text"]


# 2. Countvectorizer: 
#     - TF-IDF Vectorizer: 
#     - stopwords get removed 
#     - Words NOT Stemmed

# In[ ]:


X = data['text']
y = data['Dummy_spam']
X_train,X_test,y_train,y_test=train_test_split(X,y)
results2= []

for name, model in models:
    text_clf=Pipeline([('tfidf',TfidfVectorizer(stop_words='english')),(name,model)])
    text_clf.fit(X_train, y_train)
    pred = text_clf.predict(X_test)
    acc = accuracy_score(y_test, pred)
    precision = precision_score(y_test, pred, average=None)
    recall = recall_score(y_test, pred, average= None)
    error_Rate = 1- acc
    cm = pd.DataFrame(metrics.confusion_matrix(y_test,pred),index=['ham','spam'],columns=['ham','spam'])
    print('Model tested: {}'.format(name))
    print('Confusion Matrix')
    print(cm)
    print('Accuracy= {}'.format(acc))
    print('Error Rate= {}'.format(error_Rate))
    print('Recall Rate= {}'.format(recall))
    print("Precision Rate: {}".format(precision))
    print(metrics.classification_report(y_test,pred))
    print()
    results2.append([name,precision])


# In[ ]:


df1 = pd.DataFrame.from_items(results,orient='index', columns=['Accuracy length'])
df2 = pd.DataFrame.from_items(results2,orient='index', columns=['Accuracy words of bag'])
df = pd.concat([df1,df2],axis=1)
df.plot(kind='bar', figsize=(12,6), align='center')
plt.xticks(np.arange(9), df.index)
plt.ylabel('Accuracy Score')
plt.title('Accuracy by Classifier')
plt.legend(bbox_to_anchor=(1.05, 1), loc=2)


# occurrences to frequencies
# https://scikit-learn.org/stable/tutorial/text_analytics/working_with_text_data.html#from-occurrences-to-frequencies

# 3. Countvectorizer: 
#     - TF-IDF Vectorizer: 
#     - stopwords get removed 
#     - Words Stemmed

# In[ ]:


def get_final_text(stemmed_text):
    final_text=" ".join([word for word in stemmed_text])
    return final_text


# In[ ]:


stemmer = PorterStemmer()
data = pd.read_csv("../input/sms-spam-collection-dataset/spam.csv",encoding='latin-1')
data.drop(["Unnamed: 2", "Unnamed: 3", "Unnamed: 4"], axis=1, inplace = True)
data.columns =['label','text']
data.text = data['text'].str.split()
data['stemmed_text'] = data['text'].apply(lambda x: [stemmer.stem(y) for y in x])
data['final_text']=data.stemmed_text.apply(lambda row : get_final_text(row))
data.head()


# In[ ]:


X = data['final_text']
y = data['label']
X_train,X_test,y_train,y_test=train_test_split(X,y)
results3= []

for name, model in models:
    text_clf=Pipeline([('tfidf',TfidfVectorizer(stop_words='english')),(name,model)])
    text_clf.fit(X_train, y_train)
    pred = text_clf.predict(X_test)
    acc = accuracy_score(y_test, pred)
    precision = precision_score(y_test, pred, average=None)
    recall = recall_score(y_test, pred, average= None)
    error_Rate = 1- acc
    cm = pd.DataFrame(metrics.confusion_matrix(y_test,pred),index=['ham','spam'],columns=['ham','spam'])
    print('Model tested: {}'.format(name))
    print('Confusion Matrix')
    print(cm)
    print('Accuracy= {}'.format(acc))
    print('Error Rate= {}'.format(error_Rate))
    print('Recall Rate= {}'.format(recall))
    print("Precision Rate: {}".format(precision))
    print(metrics.classification_report(y_test,pred))
    print()
    results3.append([name,precision])

