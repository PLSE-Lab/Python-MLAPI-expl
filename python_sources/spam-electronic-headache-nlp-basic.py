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


# **Importing the data **

# In[ ]:


data= pd.read_csv('../input/sms-spam-collection-dataset/spam.csv',encoding='ISO-8859-1')


# In[ ]:


data.head()


# In[ ]:


data.columns


# We can see that there are there unnecessary columns in the data set.We can drop the unwanted columns 

# In[ ]:


data.drop(columns=['Unnamed: 2', 'Unnamed: 3','Unnamed: 4'],inplace=True)


# In[ ]:


data.head()


# Lets change the names of columns for convienence

# In[ ]:


data.rename({'v1': 'labels', 'v2': 'messages'}, axis=1, inplace=True)


# In[ ]:


data.head()


# In[ ]:


data.describe()


# In[ ]:


data.info()


# In[ ]:


data.groupby('labels').describe().T


# In[ ]:


data.isnull().sum()


# In[ ]:


len(data)


# In[ ]:


data['length']=data['messages'].apply(len)


# In[ ]:


data.head()


# In[ ]:


data['labels'].unique()


# In[ ]:


data['labels'].value_counts()


# **Plotting the histogram of Labels **

# In[ ]:


import matplotlib.pyplot as plt 
import seaborn as sns
#plt.style.use('fivethirtyeight')


# In[ ]:


data['length'].plot(bins=50,kind='hist')
plt.ioff()


# In[ ]:


plt.xscale('log')
bins=1.15**(np.arange(0,50))
plt.hist(data[data['labels']=='ham']['length'],bins=bins,alpha=0.8)
plt.hist(data[data['labels']=='spam']['length'],bins=bins,alpha=0.8)
plt.legend('ham','spam')
plt.show()


# Spam text messages are longer than ham text messages 

# In[ ]:


data.hist(column='length',by='labels',bins=50,figsize=(10,4))
plt.ioff()


# **Lets print out longest message**

# In[ ]:


data['length'].describe()


# In[ ]:


data[data['length']==910]['messages'].iloc[0]


# **Email Classification based on length of Mail**

# In[ ]:


from sklearn.model_selection import train_test_split


# **Creating the matrix features and target **

# In[ ]:


X=data['length'].values[:,None]
#X=data['length'].values
y=data['labels']


# **Splitting the data **

# In[ ]:


X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=42)


# In[ ]:


X_train.shape


# In[ ]:


#y_test


# **Using Logistic Regression**

# In[ ]:


from sklearn.linear_model import LogisticRegression


# In[ ]:


lr_model=LogisticRegression(solver='lbfgs')


# In[ ]:


lr_model.fit(X_train,y_train)


# In[ ]:


from sklearn import metrics


# In[ ]:


predictions=lr_model.predict(X_test)


# In[ ]:


predictions


# In[ ]:


#y_test


# In[ ]:


print(metrics.confusion_matrix(y_test,predictions))


# In[ ]:


df=pd.DataFrame(metrics.confusion_matrix(y_test,predictions),index=['ham','spam'],columns=['ham','spam'])
df


# In[ ]:


print(metrics.classification_report(y_test,predictions))


# In[ ]:


print(metrics.accuracy_score(y_test,predictions))


# **Using Naive Bayes **

# In[ ]:


from sklearn.naive_bayes import MultinomialNB
nb_model=MultinomialNB()
nb_model.fit(X_train,y_train)
predictions=nb_model.predict(X_test)
print(metrics.confusion_matrix(y_test,predictions))


# In[ ]:


print(metrics.classification_report(y_test,predictions))


# **Lets try Support Vector Machine **

# In[ ]:


from sklearn.svm import SVC
svc_model=SVC(gamma='auto')
svc_model.fit(X_train,y_train)
predictions=svc_model.predict(X_test)
print(metrics.confusion_matrix(y_test,predictions))


# In[ ]:


print(metrics.classification_report(y_test,predictions))


# **Extracting the features from text **

# In[ ]:


data.head()


# #### Check for missing values 

# In[ ]:


data.isnull().sum()


# In[ ]:


data['labels'].value_counts()


# In[ ]:


from sklearn.model_selection import train_test_split


# In[ ]:


X=data['messages']


# In[ ]:


y=data['labels']


# In[ ]:


X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.33,random_state=42)


# In[ ]:


from sklearn.feature_extraction.text import CountVectorizer 


# In[ ]:


count_vect=CountVectorizer()


# In[ ]:


# FIT Vectorizer to the data (build a vocab,count the number of words)
#count_vect.fit(X_train)
# Transform the original text to message --> Vector 
#X_train_counts=count_vect.transform(X_train)

X_train_counts=count_vect.fit_transform(X_train) # One step Fit and Transform


# In[ ]:


X_train_counts


# In[ ]:


X_train.shape


# In[ ]:


X_train_counts.shape


# In[ ]:


from sklearn.feature_extraction.text import TfidfTransformer 


# In[ ]:


tfidf_transformer=TfidfTransformer()


# In[ ]:


X_train_tfidf=tfidf_transformer.fit_transform(X_train_counts)


# In[ ]:


X_train_tfidf.shape


# **Combining the Count Vectorization and Tdidf Transformation **

# In[ ]:


from sklearn.feature_extraction.text import TfidfVectorizer 


# In[ ]:


vectorizer=TfidfVectorizer()


# In[ ]:


X_train_tfidf=vectorizer.fit_transform(X_train)


# **Training a classifier **

# In[ ]:


from sklearn.svm import LinearSVC


# In[ ]:


clf=LinearSVC()


# In[ ]:


clf.fit(X_train_tfidf,y_train)


# **Creating a single pipeline tfidf,Vectorizer and Classification**

# In[ ]:


from sklearn.pipeline import Pipeline


# In[ ]:


text_clf=Pipeline([('tfidf',TfidfVectorizer()),('clf',LinearSVC())])


# In[ ]:


text_clf.fit(X_train,y_train)


# In[ ]:


predictions=text_clf.predict(X_test)


# ![](http://)**Confusion Matrix & CLassification report**

# In[ ]:


from sklearn.metrics import confusion_matrix,classification_report


# In[ ]:


print(confusion_matrix(y_test,predictions))


# In[ ]:


print(classification_report(y_test,predictions))


# **Accuracy **

# In[ ]:


from sklearn import metrics 


# In[ ]:


metrics.accuracy_score(y_test,predictions)


# **Predicting on new dataset **

# In[ ]:


text_clf.predict(["Hi how are you doing today"])


# In[ ]:


text_clf.predict(["COngraluations you are lucky winner of bummer prize money"])


# In[ ]:




