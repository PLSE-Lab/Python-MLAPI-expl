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


#importing all the library
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import nltk
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelBinarizer
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize,sent_tokenize
from bs4 import BeautifulSoup
import re,string,unicodedata
from nltk.tokenize.toktok import ToktokTokenizer
from nltk.stem import LancasterStemmer,WordNetLemmatizer


# In[ ]:


train = pd.read_csv("/kaggle/input/hackerearth-effectiveness-of-std-drugs/dataset/train.csv")
test = pd.read_csv("/kaggle/input/hackerearth-effectiveness-of-std-drugs/dataset/train.csv")


# In[ ]:


train.head()


# In[ ]:


#checking for null
train.isna().sum()


# In[ ]:


# Taking the required columns
train_new=train[['patient_id','effectiveness_rating','number_of_times_prescribed','review_by_patient','base_score']]


# **PreProcessing of the REVIEW column using NATURAL LANGUAGE PROCESSING**

# In[ ]:


from nltk.corpus.reader.wordnet import WordNetError
from nltk.corpus import sentiwordnet as swn, wordnet


# In[ ]:


stop = set(stopwords.words('english'))
punctuation = list(string.punctuation)
stop.update(punctuation)


# In[ ]:


def get_simple_pos(tag):
    if tag.startswith('J'):
        return wordnet.ADJ
    elif tag.startswith('V'):
        return wordnet.VERB
    elif tag.startswith('N'):
        return wordnet.NOUN
    elif tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN


# In[ ]:


from nltk import pos_tag, word_tokenize


# In[ ]:


lemmatizer = WordNetLemmatizer()
def lemmatize_words(review_by_patient):
    final_text = []
    for i in review_by_patient.split():
        if i.strip().lower() not in stop:
            pos = pos_tag([i.strip()])
            word = lemmatizer.lemmatize(i.strip(),get_simple_pos(pos[0][1]))
            final_text.append(word.lower())
    return " ".join(final_text)


# In[ ]:


train_new.review_by_patient = train_new.review_by_patient.apply(lemmatize_words)


# In[ ]:


train_new.head()


# **NOW we will create a dataframe of rest all numeric column along with review column**

# In[ ]:


from sklearn.feature_extraction.text import TfidfVectorizer


# In[ ]:


from sklearn_pandas import DataFrameMapper, cross_val_score


# In[ ]:


data2 = train_new.copy()


# In[ ]:


data2 = data2.fillna('')


# DataFrameMapper is used to combined the dataset of different types
# 
# like here we have combine text data along with other columns

# In[ ]:


mapper = DataFrameMapper([
     ('patient_id',None),
     ('effectiveness_rating',None),
     ('number_of_times_prescribed', None),
     ('review_by_patient', TfidfVectorizer()),
 ])


# In[ ]:


features = mapper.fit_transform(data2)


# **Now we will prepare the data for train and test split**

# In[ ]:


features.shape


# In[ ]:


train_new.dtypes


# In[ ]:


pred_base_score = train_new['base_score']


# In[ ]:


from sklearn.model_selection import train_test_split


# In[ ]:


# Split the data between train and test
x_train, x_test, y_train, y_test = train_test_split(features,pred_base_score,test_size=0.2,train_size=0.8, random_state = 0)


# In[ ]:


x_train.shape


# In[ ]:


x_test.shape


# In[ ]:


y_train


# As the the y target data is float type we will convert it into integer type using label encoder

# In[ ]:


y_train_new=y_train


# In[ ]:


y_train_new.shape


# In[ ]:


from sklearn import preprocessing
from sklearn import utils

lab_enc = preprocessing.LabelEncoder()
encoded = lab_enc.fit_transform(y_train_new)


# **Importing the logistic model for prediction**
# 
# 
# 

# In[ ]:


from sklearn.linear_model import LogisticRegression


# In[ ]:


model=LogisticRegression()


# **AS DUE to memory issue of the kernel try to use google colab for faster execution **
# 
# As due to memory issue I have done the model fitting using only 500 rows but you should fit the model for all the rows of X_train.So that you can get better accuracy.
# 
# model.fit(x_train,encoded)
# 
# prediction=model.predict(x_test)

# In[ ]:


x_train.shape


# In[ ]:


x_hlf=x_train[0:500, :]


# In[ ]:


x_hlf.shape


# In[ ]:


encoded.shape


# In[ ]:


y_hlf=encoded[0:500]


# In[ ]:


y_hlf.shape


# In[ ]:


model.fit(x_hlf,y_hlf)


# In[ ]:


prediction=model.predict(x_test)


# In[ ]:


final_prediction=(prediction/100)


# In[ ]:


final_prediction


# AS for test dataset you have to follow the similar manner first preprocessing of the test dataset as you have done on train dataset
#  
#  And than do the prediction part and create a submission file.
