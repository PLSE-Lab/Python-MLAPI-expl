#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import re # Regular expression


# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        path=(os.path.join(dirname, filename))
print(path)

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# ### Data Reading and EDA

# In[ ]:


data=pd.read_csv(path)


# In[ ]:


data.head()


# In[ ]:


data.info()


# In[ ]:


data.isna().any()


# In[ ]:


data['category'].value_counts()


# In[ ]:


import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns


# In[ ]:


sns.countplot(data['category'])


# *There is not much imbalance*

# ### Data wrangling
# 

# In[ ]:


data['text-cleaned']=data.text.apply(lambda x: re.sub(r'[^A-Za-z]+',' ',x))


# In[ ]:


data['text-cleaned']=data['text-cleaned'].apply(lambda x: x.lower())


# In[ ]:


data['text-cleaned']=data['text-cleaned'].apply(lambda x:x.strip())


# In[ ]:


import nltk
from nltk.corpus import stopwords


# In[ ]:


stopwords=stopwords.words("english")


# In[ ]:


data['text-cleaned']=data['text-cleaned'].apply(lambda x : ' '.join([words for words in x.split() if words not in stopwords]))


# In[ ]:


print("Text before cleaning===============> {}".format(data.text[1]))


# In[ ]:


print("Text after cleaning===============> {}".format(data['text-cleaned'][1]))


# ### Splitting data into independent and predicting features

# In[ ]:


X=data['text']
Y=data['category']


# In[ ]:


X.head()


# In[ ]:


Y.head()


# ### Pre-processing the data

# In[ ]:


from sklearn.feature_extraction.text import CountVectorizer


# In[ ]:


CountProcessor=CountVectorizer(binary=True, ngram_range=(1,3))
X=CountProcessor.fit_transform(X)


# In[ ]:


X[1].toarray()


# In[ ]:


from sklearn.preprocessing import LabelEncoder


# In[ ]:


Labelencoder=LabelEncoder()
Y=Labelencoder.fit_transform(Y)


# In[ ]:


from sklearn.model_selection import train_test_split
Xtrain,Xtest,Ytrain,Ytest=train_test_split(X,Y,test_size=0.2,random_state=42)


# In[ ]:


Ytrain[5]


# In[ ]:


from sklearn.ensemble import GradientBoostingClassifier


# In[ ]:


model=GradientBoostingClassifier()
model.fit(Xtrain,Ytrain)


# In[ ]:


Ypred=model.predict(Xtest)


# In[ ]:


print("The accuracy of Train set is {}".format(model.score(Xtrain,Ytrain)))


# In[ ]:


print("The accuracy of Test set is {}".format(model.score(Xtest,Ytest)))


# In[ ]:


from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
def print_evaluation_scores(y_val, predicted):
    print('Accuracy score: ', accuracy_score(y_val, predicted))
    print('F1 score: ', f1_score(y_val, predicted, average='micro'))
    print('Average precision score: ', precision_score(y_val, predicted, average='micro'))
    print('Average recall score: ', recall_score(y_val, predicted, average='micro'))


# In[ ]:


print_evaluation_scores(Ytest, Ypred)


# In[ ]:


pred_inversed = Labelencoder.inverse_transform(Ypred)
y_test_inversed = Labelencoder.inverse_transform(Ytest)
Xtest=CountProcessor.inverse_transform(Xtest)


# In[ ]:


for i in range(5):
    print('Title:\t{}\nTrue labels:\t{}\nPredicted labels:\t{}\n\n'.format(
        Xtest[i],
        ','.join(y_test_inversed[i]),
        ','.join(pred_inversed[i])
    ))


# In[ ]:




