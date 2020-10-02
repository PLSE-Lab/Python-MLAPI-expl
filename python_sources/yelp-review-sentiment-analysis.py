#!/usr/bin/env python
# coding: utf-8

# In this Kernel we will exlore the Data Set.Apply NLP techniques like Data Cleaning,Apply Count Vectoriser and Use Naive Bayes Algroithm to do a Sentiment Analysis.This Kernel is work in Process if you Like my Work please do vote.

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


# **Import Python Modules**

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# **Import the Dataset**

# In[ ]:


df=pd.read_csv('../input/yelp-csv/yelp_academic_dataset_review.csv')
df.head()


# **Summary of Data**

# In[ ]:


print('Rows     :',df.shape[0])
print('Columns  :',df.shape[1])
print('\nFeatures :\n     :',df.columns.tolist())
print('\nMissing values    :',df.isnull().values.sum())
print('\nUnique values :  \n',df.nunique())


# In[ ]:


df.isnull().sum()


# In[ ]:


df.dropna()


# We have dropped the row for which the review was missing.

# In[ ]:


df.describe()


# We can see that the mean of column stars is 3.73

# In[ ]:


df.info()


# No missing values in our Dataset

# In[ ]:


#df.drop(df.loc[10001:1125458].index, inplace=True)
#df.shape


# **Displaying the Reviews**

# In[ ]:


df['text'][0]


# **Visualise Data set **

# In[ ]:


#df['lenght']=df['text'].apply(len)
df['length']=df['text'].str.len()


# In[ ]:


df.head()


# We have calculated the length of each document and added the value to the Length Colum of the Dataframe.

# **Histogram**

# In[ ]:


df['length'].plot(bins=100,kind='hist');


# We can see that most of the revies are around 700 words

# In[ ]:


df.describe().T


# We can see that the maximum length of the text is 5000 words.Lets Display this review

# In[ ]:


df[df['length']==1]['text'].iloc[0]


# Smallest Review contains only A

# In[ ]:


df[df['length']==5000]['text'].iloc[0]


# Now thats a really very big review :)

# **Lets Look at the Start Columns**

# In[ ]:


sns.countplot(x='stars',data=df);
#sns.countplot(y='stars',data=df);


# We can see that there are more 5 start ratings in reviews.Thats Surprising.

# Plott**ing the histogram of all the stars**

# In[ ]:


g=sns.FacetGrid(data=df,col='stars',col_wrap=5)
g.map(plt.hist,'length',bins=20,color='r');


# Looking at the Histogram of Start 5 we can conclude that Generally ratings with 5 star are short

# **Segregating Reviews with 1 and 5 star**

# In[ ]:


df_1=df[df['stars']==1]
df_1.head()


# In[ ]:


df_5=df[df['stars']==5]
df_5.head()


# In[ ]:


df_1_5=pd.concat([df_1,df_5])
df_1_5.shape


# In[ ]:


df_1_5.info()


# In[ ]:


print('1-Star Review Percentage=',(len(df_1)/len(df_1_5))*100,'%')


# In[ ]:


print('5-Star Review Percentage=',(len(df_5)/len(df_1_5))*100,'%')


# In[ ]:


sns.countplot(x=df_1_5['stars'],label='Count');


# **Creating Testing and Training Data**

# **Exercise to remove Puncuation**

# In[ ]:


import string
string.punctuation


# In[ ]:


Test='Hello Mr. Future,I am so happy to be learning AI'


# In[ ]:


Test_punc_removed=[char  for char in Test if char not in string.punctuation]
Test_punc_removed


# In[ ]:


Test_punc_removed_join=''.join(Test_punc_removed)


# In[ ]:


Test_punc_removed_join


# **Exercise to remove STOPWORDS**

# In[ ]:


from nltk.corpus import stopwords
stopwords.words('english')


# Prited above are the list of Stop Words in the nltk library

# In[ ]:


Test_punc_removed_join_clean=[word  for word in Test_punc_removed_join.split() if word.lower() not in stopwords.words('english')]
Test_punc_removed_join_clean


# So we have managed to remove stop words from out list

# In[ ]:


mini_challenge='Here is a mini challenge,that will teach you how to remove stopwords and puncutations'


# In[ ]:


challenge=[char for char in mini_challenge if char not in string.punctuation  ]
challenge=''.join(challenge)
challenge=[word  for word in challenge.split() if word.lower() not in stopwords.words('english')]


# In[ ]:


challenge


# **Exercise Count Vectoriser**

# In[ ]:


sample_data=['This is the first document.','This is thesecond document.','This is the third document']
from sklearn.feature_extraction.text import CountVectorizer
vectorizer=CountVectorizer()
X=vectorizer.fit_transform(sample_data)


# In[ ]:


print(vectorizer.get_feature_names())


# In[ ]:


print(X.toarray())


# In[ ]:


mini_challenge=['Hello World','Hello Hello World','Hello World world world']
vectorizer_challenge=CountVectorizer()
X_challenge=vectorizer_challenge.fit_transform(mini_challenge)
print(X_challenge.toarray())


# R**emoving Puncuation,Stop words and Appling Count Vectorizer to the dataset**

# In[ ]:


df_1_5


# In[ ]:


df_1_5 = df_1_5.reset_index()
df_1_5.shape


# In[ ]:


#df_1_5.drop(df_1_5.loc[0:516818].index, inplace=True)   # Considering only first 10000 reviews
#df_1_5.shape


# In[ ]:


def message_cleaning(message):
    Test_punc_removed = [char for char in message if char not in string.punctuation ]
    Test_punc_removed_join=''.join(Test_punc_removed)
    Test_punc_removed_join_clean=[ word for word in Test_punc_removed_join.split() if word.lower() not in stopwords.words('english')]
    return Test_punc_removed_join_clean


# In[ ]:


df_clean=df_1_5['text'].apply(message_cleaning)


# In[ ]:


"""#test_strs = ['THIS IS A TEST!', 'another test', 'JUS!*(*UDFLJ)']
df = pd.DataFrame(df_1_5, columns=['text'])
df_clean = df.apply(lambda x: message_cleaning(x.text), axis=1)"""


# In[ ]:


"""#test_strs = ['THIS IS A TEST!', 'another test', 'JUS!*(*UDFLJ)']
df = pd.DataFrame(test_strs, columns=['text'])
df['new_text'] = df.apply(lambda x: clean(x.text), axis=1)"""


# In[ ]:


#df_clean=df_1_5['text'].apply(message_cleaning)


# In[ ]:


print(df_clean[0]) # cleaned up review 


# In[ ]:


print(df_1_5['text'][0]) # Original review


# **Applying Count Vectoriser to the data**

# In[ ]:


from sklearn.feature_extraction.text import CountVectorizer
vectorizer=CountVectorizer(analyzer=message_cleaning)
df_countvectorizer=vectorizer.fit_transform(df_1_5['text'])


# In[ ]:


print(vectorizer.get_feature_names())


# In[ ]:


print(df_countvectorizer.toarray())


# In[ ]:


df_countvectorizer.shape


# **Model Training **

# In[ ]:


from sklearn.naive_bayes import MultinomialNB
NB_classifier=MultinomialNB()
label=df_1_5['stars'].values


# In[ ]:


df_1_5['stars'].values


# In[ ]:


NB_classifier.fit(df_countvectorizer,label)


# In[ ]:


testing_sample=['amazing food! highly recommended']
testing_sample_countvectorizer=vectorizer.transform(testing_sample)
test_predict=NB_classifier.predict(testing_sample_countvectorizer)
test_predict


# **Dividing the data into test train split**

# In[ ]:


X=df_countvectorizer
X.shape


# In[ ]:


y=label
y.shape


# In[ ]:


from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test,y_test=train_test_split(X,y,test_size=0.2)


# In[ ]:


X_train.shape


# In[ ]:


y_train.shape


# In[ ]:


from sklearn.naive_bayes import MultinomialNB
NB_classifier=MultinomialNB()
NB_classifier.fit(X_train,y_train)


# **Evaluating the Model**

# In[ ]:


from sklearn.metrics import classification_report,confusion_matrix
y_predict_train=NB_classifier.predict(X_train)
y_predict_train


# In[ ]:


cm=confusion_matrix(y_train,y_predict_train)
sns.heatmap(cm,annot=True)


# In[ ]:


y_predict_test=NB_classifier.predict(X_test)
y_predict_test
cm=confusion_matrix(y_test,y_predict_test)
sns.heatmap(cm,annot=True)


# In[ ]:


print(classification_report(y_test,y_predict_test))


# In[ ]:





# In[ ]:




