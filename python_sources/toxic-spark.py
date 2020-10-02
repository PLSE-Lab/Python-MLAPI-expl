#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[ ]:


pip install pyspark


# In[ ]:


from pyspark.sql import SQLContext
from pyspark import SparkContext
import pandas as pd
from pyspark.sql import SparkSession
import pyspark.sql.functions as F
import pyspark.sql.types as T
from pyspark.ml.feature import Tokenizer, HashingTF, IDF
from pyspark.ml.classification import LogisticRegression
#sc =SparkContext()


# In[ ]:


spark = SparkSession.builder.appName("TestCSV").getOrCreate()


# In[ ]:


train_df = spark.read.csv('../input/train.csv',header=True)
test_df = spark.read.csv('../input/test.csv',header=True)


# In[ ]:


train_df.take(5)


# In[ ]:


train_df.columns


# In[ ]:


train_df_old = train_df


# In[ ]:


out_cols = [i for i in train_df.columns if i not in ["id", "comment_text"]]


# In[ ]:


out_cols


# In[ ]:


train_df = train_df.drop(*out_cols)


# In[ ]:


train_df.take(5)


# In[ ]:


test_df.take(5)


# In[ ]:


all_data = train_df.union(test_df)


# In[ ]:


all_data.take(5)


# In[ ]:


from nltk import pos_tag 
from nltk.corpus import wordnet
def get_wordnet_pos(pos_tag):
    if pos_tag.startswith('J'):
        return wordnet.ADJ
    elif pos_tag.startswith('V'):
        return wordnet.VERB
    elif pos_tag.startswith('N'):
        return wordnet.NOUN
    elif pos_tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN


# In[ ]:


import string
from nltk import pos_tag
from nltk.corpus import stopwords
from nltk.tokenize import WhitespaceTokenizer
from nltk.stem import WordNetLemmatizer

def clean_text(text):
    # lower text
    text = text.lower()
    # tokenize text and remove puncutation
    text = [word.strip(string.punctuation) for word in text.split(" ")]
   # remove words that contain numbers
    text = [word for word in text if not any(c.isdigit() for c in word)]
   # remove stop words
    stop = stopwords.words('english')
    text = [x for x in text if x not in stop]
   # remove empty tokens
    text = [t for t in text if len(t) > 0]
   # pos tag text
    pos_tags = pos_tag(text)
   # lemmatize text
    text = [WordNetLemmatizer().lemmatize(t[0], get_wordnet_pos(t[1])) for t in pos_tags]
   # remove words with only one letter
    text = [t for t in text if len(t) > 1]
   # join all
    text = " ".join(text)
    return(text)


# In[ ]:


all_rdd = all_data.select("comment_text").rdd.flatMap(lambda x: x)
all_rdd.take(5)


# In[ ]:


all_rdd = all_rdd.filter(lambda x: x is not None).filter(lambda x: x != "")


# In[ ]:


#all_rdd.collect()


# In[ ]:


#all_rdd = all_data.map(lambda x : clean_text(x))


# In[ ]:


#cleaned_rdd.collect()


# In[ ]:


# from pyspark.sql.types import StringType 
# from pyspark.sql.functions import udf
# clean_udf = udf(clean_text,StringType())


# In[ ]:


all_rdd = all_rdd.map(lambda x : x.lower())
all_rdd.take(5)


# In[ ]:


all_rdd = all_rdd.map(lambda x : [word.strip(string.punctuation) for word in x.split(" ")])


# In[ ]:


all_rdd = all_rdd.map(lambda text : [word for word in text if not any(c.isdigit() for c in word)])
all_rdd.take(5)


# In[ ]:


stop = stopwords.words('english')
remove_stop = lambda text : [x for x in text if x not in stop]


# In[ ]:


all_rdd = all_rdd.map(remove_stop)
all_rdd.take(5)


# In[ ]:


remove_empty = lambda text : [t for t in text if len(t) > 0]
all_rdd = all_rdd.map(remove_empty)
all_rdd.take(5)


# In[ ]:


# pos_tags = lambda text:pos_tag(text)
# # def get_pos_tag(text):
# #     yield pos_tag(text)
# # def lemmatize(text):
# #     yiled
import nltk
nltk.download('wordnet')
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
# lemmatize text
def lemma(x):
    return lemmatizer.lemmatize(x)

lemmatize = lambda x : [lemma(i) for i in x]
#lemmatize = lambda post_tags : [WordNetLemmatizer().lemmatize(t[0], get_wordnet_pos(t[1])) for t in pos_tags]


# all_rdd = all_rdd.map(pos_tags)
all_rdd = all_rdd.map(lemmatize)


# In[ ]:


all_rdd.take(5)


# In[ ]:


rem_small = lambda text : [t for t in text if len(t) > 1]
all_rdd = all_rdd.map(rem_small)
all_rdd.take(5)


# In[ ]:


to_string = lambda text : " ".join(text)
all_rdd = all_rdd.map(to_string)
all_rdd.take(5)


# In[ ]:




