#!/usr/bin/env python
# coding: utf-8

#  Stack Exchange Question Classification Using NLP and Scikit-learn

# The Training dataset is json file which consists of topic name,question and excerpt or description about the question. 

# We will use NLTK for text processing and different machine learing of Scikit-Learn library

# In[ ]:


#Using Pandas for Reading json/txt file
import pandas as pd
import numpy as np


# In[ ]:


#Library required for text processing
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer,LancasterStemmer,PorterStemmer


# In[ ]:


#Sklearn Library for prediction
from sklearn.pipeline import Pipeline
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import CountVectorizer,TfidfTransformer,TfidfVectorizer
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score


# ______________________________________________Training Data_______________________________________

# In[ ]:


ls ../input


# In[ ]:


df=pd.read_json(r'../input/training.json',lines=True,)


# Dropped 1st row as it consist of number of entries in files

# In[ ]:


df=df.drop(0,axis=0).reset_index(drop=True)


# Rename of column to Content

# In[ ]:


df.columns=['Content']


# Lets see short info about our dataset

# In[ ]:


df.info()
df.head(5)


# Lets see how each entry looks like

# In[ ]:


df['Content'][0]


# Lets break down the data into dictionary and create a new column for each key

# In[ ]:


df['Topic']=[dict(x)['topic'] for x in df['Content']]
df['Question']=[dict(x)['question'] for x in df['Content']]
df['Excerpt']=[dict(x)['excerpt'] for x in df['Content']]


# Thats how it finally looks

# In[ ]:


df.head(5)


# We will LabelEncoder to convert target i.e Topic into integer form

# In[ ]:


lb=LabelEncoder().fit(df['Topic'])
Y_train=lb.transform(df['Topic'])


# Let's create a function that will do the text processing

# We Will use RegexpTokenizer which will break the Excerpt into words, removing all the punctuations as well.
# Convert the Excerpt to lower case.

# Create a list of stopwords. Stopwords are the not so important words like is am are to in a an etc.
# We will refine the list of tokens,removing all the stopwords

# Now we have change the form of each word in our refined list.
# For this we can use Stemming or Lemmetization
# Stemming can be done using PorterStemmer or LancasterStemming.
# Stemming will reduce the word to its base form irrespective of correct english word where as Lemmetization will use dictionary to bring the word in its base form. Hence Lemmetization is slower process.

# In[ ]:


def textProcessing(text):
    #Tokenization
    tokenizer=RegexpTokenizer(r'\w+')
    word_token=tokenizer.tokenize(text.lower())
    #Remove Stop Words
    stopWordList=stopwords.words('english')
    wordListRefined=[]
    for word in word_token:
        if word not in stopWordList:
            wordListRefined.append(word)
    #print(word_token)
    #Lemmetization
    WordList=[]
    for word in wordListRefined:
        WordList.append(WordNetLemmatizer().lemmatize(word,pos='v'))
        #WordList.append(LancasterStemmer().stem(word))
    return " ".join(WordList)


# So we have finally created one more columns for processed text which we will use for training purpose.

# In[ ]:


df['Text']=[textProcessing(x) for x in df['Excerpt']]
X_train=df['Text']
df.head(5)


# Lets process the Testing Data same as done for training data

# In[ ]:


#______________________Testing Data_________________________________________#
df_test=pd.read_json('../input/input00.txt',lines=True)
df_test=df_test.drop(0,axis=0).reset_index(drop=True)
df_test.columns=['Content']
df_test.info()
df_test.head(5)


# In[ ]:


df_test['Question']=[dict(x)['question'] for x in df_test['Content']]
df_test['Excerpt']=[dict(x)['excerpt'] for x in df_test['Content']]
df_test['Topic']=pd.read_csv("../input/output00.txt",header=None)


# In[ ]:


Y_test=lb.transform(df_test['Topic'])


# In[ ]:


df_test['Text']=[textProcessing(x) for x in df_test['Excerpt']]
X_test=df_test['Text']
df_test.head(5)


# Here we will use CountVectorizer to create features and TfidfTransformer to apply Tfidf (Term Frequency-Inverse Document Frequency).
# 
# Here Term frequency summarizes how often a given word appears within a document. Inverse Document Frequency downscales words that appear a lot across documents.
# 
# CountVectorizer wil create the sparse matrix with count of each word.
# 
# TfidfTransformer will provide the most popular word across all document. The inverse document frequencies are calculated for each word. The lower the score more frequent the word observed.

# Here we will firstly use SVM with linear kernel

# Here,we will create a model without tuning hyper parameters. Lets see the initial performance and later on tune the htper parameters

# In[ ]:


cf_fit=CountVectorizer().fit(X_train)
cf_train=cf_fit.transform(X_train)
tf_fit=TfidfTransformer().fit(cf_train)
tf_train=tf_fit.transform(cf_train)
svm=SVC(kernel='linear')
svm.fit(tf_train,Y_train)
print("Training Data Accuracy-->",accuracy_score(Y_train,svm.predict(tf_train)))


# In[ ]:


cf_test=cf_fit.transform(X_test)
tf_test=tf_fit.transform(cf_test)
print("Testing Data Accuracy-->",accuracy_score(Y_test,svm.predict(tf_test)))


# Now we will tune the hyper parameters and few other tweeks in model.

# We can combine CountVectorizer and TfidfTransformer to TfidfVectorizer.
# 
# We will now introduce SVD for dimentionality reduction.
# 
# We can also use the pipeline from sklearn to combine all the above processes in a single line.

# In[ ]:


pipe_SVM=Pipeline([('tf',TfidfVectorizer(sublinear_tf=True)),('svm',SVC(kernel='linear',C=10))])


# In[ ]:


pipe_SVM.fit(X_train,Y_train)
accuracy_score(Y_train,pipe_SVM.predict(X_train))


# In[ ]:


accuracy_score(Y_test,pipe_SVM.predict(X_test))


# Now lets try Logistic Regression with few hyper parameter

# In[ ]:


pipe_log=Pipeline([('tf',TfidfVectorizer(sublinear_tf=True)),('svd',TruncatedSVD(n_components=500)),('lg',LogisticRegression(penalty='none',n_jobs=-1,solver='saga'))])


# In[ ]:


pipe_log.fit(X_train,Y_train)
accuracy_score(Y_train,pipe_log.predict(X_train))


# In[ ]:


accuracy_score(Y_test,pipe_log.predict(X_test))


# RESULTS:
# 
# Simple SVM with Linear Kernel ---> Training Acc.=96.71%
#                               ---> Testing Acc.=86.82%
# 
# SVM with tuned hyper parameters  ---> Training Acc.=99.98%
#                                  ---> Testing Acc.=84.62%
# 
# Logistic Regression with tuned hyper parameters ---> Training Acc.=88.83%
#                                                 ---> Testing Acc.=84.47%

# SVM give higher accuracy on training set as compared to Logistic Regression. But it is computaionally expensive approach. Logistic Regression is much faster.
