#!/usr/bin/env python
# coding: utf-8

# In[ ]:



import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import re # regular expression
import nltk

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# ### Reading the CSV and performing Exploratory data analytics

# In[ ]:


data=pd.read_csv("/kaggle/input/blog-authorship-corpus/blogtext.csv")


# In[ ]:


data.head(10)


# In[ ]:


data.isna().any()


# *No Null values present in the dataset*

# In[ ]:


data.shape


# > **There are 68,124 records and is huge to perform analysis and computation, hence we are going to take a subset and rerun with the entire data-set once all errors are fixed and optimization is done**

# In[ ]:


data=data.head(10000)


# In[ ]:


data.info()


# In[ ]:


data.drop(['id','date'], axis=1, inplace=True)


# Columns like ID and date are removed from the dateset as they do not provide much value

# In[ ]:


data.head()


# In[ ]:


data['age']=data['age'].astype('object')


# In[ ]:


data.info()


# Converted all the columns to object data-type

# ## **Data Wrangling for data['text'] column to remove all unwanted text from the column**

# In[ ]:


data['clean_data']=data['text'].apply(lambda x: re.sub(r'[^A-Za-z]+',' ',x))


# In[ ]:


data['clean_data']=data['clean_data'].apply(lambda x: x.lower())


# In[ ]:


data['clean_data']=data['clean_data'].apply(lambda x: x.strip())


# In[ ]:


print("Actual data=======> {}".format(data['text'][1]))


# In[ ]:


print("Cleaned data=======> {}".format(data['clean_data'][1]))


# ### Remove all stop words

# In[ ]:


from nltk.corpus import stopwords
stopwords=set(stopwords.words('english'))


# In[ ]:


data['clean_data']=data['clean_data'].apply(lambda x: ' '.join([words for words in x.split() if words not in stopwords]))


# In[ ]:


data['clean_data'][6]


# ### Merging all the other columns into labels columns

# In[ ]:


data['labels']=data.apply(lambda col: [col['gender'],str(col['age']),col['topic'],col['sign']], axis=1)


# In[ ]:


data.head()


# In[ ]:


data=data[['clean_data','labels']]


# In[ ]:


data.head()


# ### Splitting the data into X and Y

# In[ ]:


X=data['clean_data']


# In[ ]:


Y=data['labels']


# ### Lets perform count vectorizer with bi-grams and tri-grams to get the count vectors of the X data

# In[ ]:


from sklearn.feature_extraction.text import CountVectorizer


# In[ ]:


vectorizer=CountVectorizer(binary=True, ngram_range=(1,2))


# In[ ]:


X=vectorizer.fit_transform(X)


# In[ ]:


X[1]


# #### Let us see some feature names

# In[ ]:


vectorizer.get_feature_names()[:5]


# In[ ]:


label_counts=dict()

for labels in data.labels.values:
    for label in labels:
        if label in label_counts:
            label_counts[label]+=1
        else:
            label_counts[label]=1


# In[ ]:


label_counts


# ### Pre-processing the labels

# In[ ]:


from sklearn.preprocessing import MultiLabelBinarizer
binarizer=MultiLabelBinarizer(classes=sorted(label_counts.keys()))


# In[ ]:


Y=binarizer.fit_transform(data.labels)


# ### Splitting the data into 80% Train set :20% Test set 

# In[ ]:


from sklearn.model_selection import train_test_split


# In[ ]:


Xtrain,Xtest,Ytrain,Ytest=train_test_split(X,Y,test_size=0.2)


# In[ ]:


from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression


# In[ ]:


model=LogisticRegression(solver='lbfgs')


# In[ ]:


model=OneVsRestClassifier(model)


# In[ ]:


model.fit(Xtrain,Ytrain)


# In[ ]:


Ypred=model.predict(Xtest)


# In[ ]:


Ypred_inversed = binarizer.inverse_transform(Ypred)
y_test_inversed = binarizer.inverse_transform(Ytest)


# In[ ]:


for i in range(5):
    print('Text:\t{}\nTrue labels:\t{}\nPredicted labels:\t{}\n\n'.format(
        Xtest[i],
        ','.join(y_test_inversed[i]),
        ','.join(Ypred_inversed[i])
    ))


# In[ ]:


from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import average_precision_score
from sklearn.metrics import recall_score

def print_evaluation_scores(Ytest, Ypred):
    print('Accuracy score: ', accuracy_score(Ytest, Ypred))
    print('F1 score: ', f1_score(Ytest, Ypred, average='micro'))
    print('Average precision score: ', average_precision_score(Ytest, Ypred, average='micro'))
    print('Average recall score: ', recall_score(Ytest, Ypred, average='micro'))


# In[ ]:


print_evaluation_scores(Ytest, Ypred)


# In[ ]:




