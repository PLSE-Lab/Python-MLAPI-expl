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


# In[ ]:


#Importing Libraries
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns


# In[ ]:


#Data Acquistion
train_data=pd.read_csv('/kaggle/input/nlp-getting-started/train.csv')
test_data=pd.read_csv('/kaggle/input/nlp-getting-started/test.csv')


# In[ ]:


# First few rows of train data
train_data.head()


# In[ ]:


# First few rows of test data
test_data.head()


# In[ ]:


#Shape of train and test data
train_data.shape,test_data.shape


# In[ ]:


#Count of null values of individual feature of train data
train_data.isnull().sum()


# In[ ]:


#Count of null values of individual feature of test data
test_data.isnull().sum()


# In[ ]:


#Creating a column named target for test data with NA values
test_data['target']=np.nan


# In[ ]:


#Checking head of test data
test_data.head()


# In[ ]:


#Stacking train and test data set 
dataframe=pd.concat([train_data,test_data],ignore_index=True)


# In[ ]:


#Checking head of dataframe
dataframe.head()


# In[ ]:


#Shape of dataframe
dataframe.shape


# In[ ]:


dataframe.info()


# In[ ]:


#Checking Null Values for each column if any 
dataframe.isnull().any()


# In[ ]:


#Counting Null values of each column
null=dataframe.isnull().sum()
null


# In[ ]:


#Bar Plot of Features and their Missing Values
plt.figure(figsize=(16,4))
null.plot(kind='bar',color='blue')
plt.title('Features of Data and there Missing Values Count')


# In[ ]:


#top 10 keywords from tweets
dataframe['keyword'].value_counts(dropna=False).head(10)


# In[ ]:


#Top 20 keywords from Tweets that are not actually caused disaster
temp1=dataframe.groupby('target')['keyword'].value_counts()[0][:20]
temp1


# In[ ]:


#Top 20 keywords from tweets that have caused disaster
temp2=dataframe.groupby('target')['keyword'].value_counts()[1][:20]
temp2


# In[ ]:


fig,axes=plt.subplots(2,1,figsize=(18,16))
temp1.plot(kind='bar',color='red',ax=axes[0]).set_title('Top 20 Fake Disaster Keywords')
temp2.plot(kind='bar',color='violet',ax=axes[1]).set_title('Top 20 Real Disaster Keywords')


# In[ ]:


#Top 10 Locations of the Tweets
dataframe.location.value_counts(dropna=False).head(10)


# In[ ]:


#Top 10 locations from where fake Disaster tweets have received
temp3=dataframe.groupby('target')['location'].value_counts()[0][:10]
temp3


# In[ ]:


#Top 10 locations from where real Disaster tweets have received
temp4=dataframe.groupby('target')['location'].value_counts()[1][:10]
temp4


# In[ ]:


fig,axes=plt.subplots(2,1,figsize=(18,12))
temp3.plot(kind='bar',color='red',ax=axes[0]).set_title('Top 10 Fake Disaster Locations')
temp4.plot(kind='bar',color='violet',ax=axes[1]).set_title('Top 10 Real Disaster Locations')


# In[ ]:


#Checking Target Variable
dataframe.target.value_counts(dropna=False)


# In[ ]:


sns.set_style('darkgrid')
plt.figure(figsize=(8,4))
sns.countplot(dataframe['target'])


# In[ ]:


#Clear Report of th data using Pandas_Profile_Report 
from pandas_profiling import ProfileReport
profile = ProfileReport(dataframe)
profile


# **Text Preprocessing**

# In[ ]:


#Removing leading and ending spaces of string
dataframe['text']=dataframe['text'].apply(lambda x: x.strip())


# In[ ]:


#Text Lowercasing
dataframe['text']=dataframe['text'].apply(lambda x: " ".join(x.lower() for x in x.split()))
dataframe.head()


# In[ ]:


#Removing Punctuations
import string
punct_dict=dict((ord(punct),None) for punct in string.punctuation)
print(string.punctuation)
print(punct_dict)


# In[ ]:


for i in range(0,dataframe.shape[0]):
    dataframe['text'][i]=dataframe['text'][i].translate(punct_dict)
dataframe['text'].head()    


# In[ ]:


#Removing Stop Words
from nltk.corpus import stopwords
stop=stopwords.words('english')
len(stop)


# In[ ]:


dataframe['text']=dataframe['text'].apply(lambda x: " ".join(x for x in x.split() if x not in stop))
dataframe['text'].head()


# In[ ]:


#Removing Numbers
dataframe['text']=dataframe['text'].apply(lambda x: ''.join([x for x in x if not x.isdigit()]))
dataframe['text']


# In[ ]:


#Tokenization
from nltk.tokenize import word_tokenize
for i in range(0,len(dataframe['text'])):
    dataframe['text'][i]=word_tokenize(dataframe['text'][i])
dataframe['text']    


# In[ ]:


#Lemmatization
from nltk.stem import WordNetLemmatizer
lem=WordNetLemmatizer()
dataframe['text']=dataframe['text'].apply(lambda x: ' '.join(lem.lemmatize(term) for term in x))
dataframe['text']


# In[ ]:


#Converting datatype of target variable from float to int
dataframe['target']=dataframe['target'][:7613].astype('int')
dataframe.head()


# In[ ]:


#Removing URL's
dataframe['text'] = dataframe['text'].str.replace('http\S+|www.\S+','',case=False)
dataframe['text']


# In[ ]:


#List of Tokens
all_words=[]
for msg in dataframe['text']:
    words=word_tokenize(msg)
    for w in words:
        all_words.append(w)        


# In[ ]:


#Frequency of Most Common Words
import nltk
frequency_dist=nltk.FreqDist(all_words)
print('Length of the words',len(frequency_dist))
print('Most Common Words',frequency_dist.most_common(100))


# In[ ]:


#Frequency Plot for first 100 most frequently occuring words
plt.figure(figsize=(20,8))
frequency_dist.plot(100,cumulative=False)


# In[ ]:


#Disaster Tweets
disaster_tweets=dataframe[dataframe['target']==1]['text']
disaster_tweets


# In[ ]:


fake_disaster_tweets=dataframe[dataframe['target']==0]['text']
fake_disaster_tweets


# In[ ]:


#Word Cloud for Real Disaster Tweets
from wordcloud import WordCloud
plt.figure(figsize=(16,10))
wordcloud1=WordCloud(width=600,height=400).generate(' '.join(disaster_tweets))
plt.imshow(wordcloud1)
plt.axis('off')
plt.title('Disaster Tweets',fontsize=40)


# In[ ]:


#Word Cloud for Fake Disaster Tweets
plt.figure(figsize=(16,10))
wordcloud2=WordCloud(width=600,height=400).generate(' '.join(fake_disaster_tweets))
plt.imshow(wordcloud2)
plt.axis('off')
plt.title("Fake Disaster Tweets",fontsize=40)


# In[ ]:


#Feature extraction using Tfidf Vectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
tfidfVect=TfidfVectorizer(max_features=5000,stop_words='english')

tfidf=tfidfVect.fit_transform(dataframe['text'])


# In[ ]:


#Shape
tfidf.shape


# In[ ]:


#Splitting data into train and test
train=tfidf[:7613]
test=tfidf[7613:]


# In[ ]:


#splitting train data into training and validation sets 
X_train=train[:5330]
X_valid=train[5330:]
y_train=dataframe['target'][:5330]
y_valid=dataframe['target'][5330:7613]


# In[ ]:


#Importing Classification algorithms
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier


# In[ ]:


#Naive Bayes Model
naive=MultinomialNB()
naive_model=naive.fit(X_train,y_train)
naive_model


# In[ ]:


#Prediction
pred=naive_model.predict(X_valid)
pred


# In[ ]:


#Important Metrics to know the Performance of Model
from sklearn.metrics import classification_report,confusion_matrix,precision_score,recall_score,f1_score,accuracy_score
print('Classification Report',classification_report(y_valid,pred))
print('Confusion Matrix',confusion_matrix(y_valid,pred))
print('Accuracy Score',accuracy_score(y_valid,pred))
print('Precision Score',precision_score(y_valid,pred))
print('Recall Score',recall_score(y_valid,pred))
print('F1 Score',f1_score(y_valid,pred))


# In[ ]:


#Cross Validation
from sklearn.model_selection import cross_val_score
cross_val_score(naive_model,X=X_train,y=y_train,cv=5)


# In[ ]:


#Predicting final test data
final_naive_pred=naive_model.predict(test)
final_naive_pred=pd.Series(final_naive_pred)
final_naive_pred.value_counts()


# In[ ]:


#Logistic Regression
log=LogisticRegression()
log_model=log.fit(X_train,y_train)
log_model


# In[ ]:


#Prediction
from sklearn.preprocessing import binarize
pred2=log_model.predict_proba(X_valid)
pred3=log_model.predict(X_valid)


# In[ ]:


#Important Metrics used to know the performance of the model
print('Classification Report',classification_report(y_valid,pred3))
print('Confusion Matrix',confusion_matrix(y_valid,pred3))
print('Accuracy Score',accuracy_score(y_valid,pred3))
print('Precision Score',precision_score(y_valid,pred3))
print('Recall Score',recall_score(y_valid,pred3))
print('F1 Score',f1_score(y_valid,pred3))


# In[ ]:


#Cross Validation
from sklearn.model_selection import cross_val_score
cross_val_score(log_model,X=X_train,y=y_train,cv=5)


# In[ ]:


#ROC_AUC Curve 
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve



fpr,tpr,thresholds=roc_curve(y_valid,pred2[:,1])
plt.figure(figsize=(10,8))
plt.plot(fpr,tpr)
plt.xlim([0.0,1.0])
plt.ylim([0.0,1.0])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate ')
plt.legend()
plt.title('ROC Curve')


# In[ ]:


#Final Prediction of test data 
final_log_pred=log_model.predict(test)
final_log_pred=pd.Series(final_log_pred)
final_log_pred.value_counts()


# In[ ]:


# Fitting Decision Tree calssifier Model to the Training set
from sklearn.tree import DecisionTreeClassifier
dt = DecisionTreeClassifier(random_state=101)
dt_model=dt.fit(X_train,y_train)
dt_model


# In[ ]:


#prediction
pred4=dt_model.predict(X_valid)
pred4


# In[ ]:


#Cross Validation
from sklearn.model_selection import cross_val_score
cross_val_score(dt_model,X=X_train,y=y_train,cv=5)


# In[ ]:


#Important Metrics to know the Performance of Model
from sklearn.metrics import classification_report,confusion_matrix,precision_score,recall_score,f1_score,accuracy_score
print('Classification Report',classification_report(y_valid,pred4))
print('Confusion Matrix',confusion_matrix(y_valid,pred4))
print('Accuracy Score',accuracy_score(y_valid,pred4))
print('Precision Score',precision_score(y_valid,pred4))
print('Recall Score',recall_score(y_valid,pred4))
print('F1 Score',f1_score(y_valid,pred4))


# In[ ]:


#Predicting final test data
final_dt_pred=dt_model.predict(test)
final_dt_pred=pd.Series(final_dt_pred)
final_dt_pred.value_counts()


# In[ ]:


#Random Forest Model
rf=RandomForestClassifier(n_estimators=100,max_features='sqrt')
rf_model=rf.fit(X_train,y_train)
rf_model


# In[ ]:


#Prediction
pred5=rf_model.predict(X_valid)
pred5


# In[ ]:


#Important Metrics used to know the Performance of Model
print('Classification Report',classification_report(y_valid,pred5))
print('Confusion Matrix',confusion_matrix(y_valid,pred5))
print('Accuracy Score',accuracy_score(y_valid,pred5))
print('Precision Score',precision_score(y_valid,pred5))
print('Recall Score',recall_score(y_valid,pred5))
print('F1 Score',f1_score(y_valid,pred5))


# In[ ]:


#Predicting Final test data
final_rf_pred=rf_model.predict(test)
final_rf_pred=pd.Series(final_rf_pred)
final_rf_pred.value_counts()


# In[ ]:


#Final sumbmission
naive_pred=pd.DataFrame(final_naive_pred, columns=['target'])
naive_pred
test_data1=pd.concat([test_data['id'],naive_pred], axis=1)
final_sub1=test_data1.to_csv('final_naive_submission.csv', index=False, header=True)


# In[ ]:


#Final sumbmission
log_pred=pd.DataFrame(final_log_pred, columns=['target'])
log_pred
test_data2=pd.concat([test_data['id'],log_pred], axis=1)
final_sub2=test_data2.to_csv('final_log_submission.csv', index=False, header=True)


# In[ ]:


#Final sumbmission
dt_pred=pd.DataFrame(final_dt_pred, columns=['target'])
dt_pred
test_data3=pd.concat([test_data['id'],dt_pred], axis=1)
final_sub3=test_data3.to_csv('final_dt_submission.csv', index=False, header=True)


# In[ ]:


#Final sumbmission
rf_pred=pd.DataFrame(final_rf_pred, columns=['target'])
rf_pred
test_data4=pd.concat([test_data['id'],rf_pred], axis=1)
final_sub4=test_data4.to_csv('final_rf_submission.csv', index=False, header=True)


# Out of all the models, the best model is naive bayes with accuracy of 0.76

# In[ ]:




