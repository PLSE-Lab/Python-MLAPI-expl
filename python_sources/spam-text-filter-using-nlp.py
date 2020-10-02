#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# In[ ]:


import nltk
#nltk.download_shell()
import matplotlib.pyplot as plt
import seaborn as sns


# In[ ]:


messages = [line.rstrip() for line  in open("/kaggle/input/spam-datasets/SMSSpamCollection")]


# In[ ]:


print(len(messages))


# In[ ]:


messages[0:5]


# In[ ]:


for mesg_num,message in enumerate(messages[0:5]):
    print(mesg_num, message)
    print("\n")
    
#tab seperated file \t


# In[ ]:


messages = pd.read_csv("/kaggle/input/spam-datasets/SMSSpamCollection",sep="\t",names=["label","message"])


# In[ ]:


messages.describe()


# In[ ]:


messages.groupby(by='label').describe()


# In[ ]:


messages["length"] = messages["message"].apply(len)


# In[ ]:


messages.head()


# In[ ]:


messages["length"].hist(bins=120)    


# In[ ]:


messages["length"].describe()


# In[ ]:


messages[messages["length"]==910]["message"].iloc[0]       #to show whole mesg


# In[ ]:


messages.hist(column="length",by="label",bins=60,figsize=(12,4))


# # **Spam messages seem to have more characters**

# In[ ]:


#Text preprocessing for classification
#remove stop words (general words)

import string
message = "sample text! with,& punctuation."
#remove punctuation  sring.punctuiation

non_punctuation = [c for c in message if c not in string.punctuation]
non_punctuation = "".join(non_punctuation)


# In[ ]:


from nltk.corpus import stopwords

stopwords.words("english")       #useless words fot classification


# In[ ]:


non_punctuation.split() #split into words


# In[ ]:


clean_message = [word for word in non_punctuation.split() if word.lower() not in stopwords.words("english") ]


# In[ ]:


clean_message


# # Building function with above functionality (Tokenization)
# 
# Tokenization is cleaning the text of punctuation and other common words to make the text more useful
# 

# In[ ]:


import string
from nltk.corpus import stopwords
def text_process(message):
    non_punctuation = [char for char in message if char not in string.punctuation]       # remove punctuation
    non_punctuation = "".join(non_punctuation)
    clean_message = [word for word in non_punctuation.split() if word.lower() not in stopwords.words("english") ]
    return clean_message    


# In[ ]:


messages.head()


# In[ ]:


messages["message"].head(5).apply(text_process)                #TOKENIZED


# Use stemming to club similar words wrt dictionary if needed
# STEMMING - not great for shortform messages need full words

# In[ ]:


messages.head()


# BAG OF WORDS MODEL

# In[ ]:


from sklearn.feature_extraction.text import CountVectorizer


# In[ ]:


bow_transformer = CountVectorizer(analyzer = text_process)
bow_transformer.fit(messages["message"])

# use our own custom function text_process defined above inside the count vectorizer


# In[ ]:


len(bow_transformer.vocabulary_)


# In[ ]:


messages_bow = bow_transformer.transform(messages["message"])


# In[ ]:


messages_bow.shape


# In[ ]:


messages_bow.nnz              #no of Non zeroes 


# In[ ]:


#sparsity

sparsity = 100*messages_bow.nnz/(messages_bow.shape[0]*messages_bow.shape[1])
format(sparsity)


# # Weights calculated by TF-IDF

# In[ ]:


from sklearn.feature_extraction.text import TfidfTransformer

tfidf_transformer = TfidfTransformer()
tfidf_transformer.fit_transform(messages_bow)
messages_tfidf = tfidf_transformer.transform(messages_bow)


# In[ ]:


print(tfidf_transformer.idf_[bow_transformer.vocabulary_['u']])
print(tfidf_transformer.idf_[bow_transformer.vocabulary_['university']])


# In[ ]:


from sklearn.naive_bayes import MultinomialNB
spam_detection = MultinomialNB().fit(messages_tfidf,messages["label"])


# In[ ]:


all_pred = spam_detection.predict(messages_tfidf)


# In[ ]:


all_pred


# In[ ]:


from sklearn.model_selection import train_test_split
msg_train,msg_test,label_train,label_test = train_test_split(messages["message"],messages["label"],test_size=0.3)


# In[ ]:


from sklearn.pipeline import Pipeline
pipeline =Pipeline([("bow",CountVectorizer(analyzer=text_process)),
                    ("tfidf",TfidfTransformer()),
                     ("clf",MultinomialNB())
                    ])


# In[ ]:


label_train


# In[ ]:


msg_train


# In[ ]:


pipeline.fit(msg_train,label_train)


# In[ ]:


predictions = pipeline.predict(msg_test)


# In[ ]:


from sklearn.metrics import classification_report
print(classification_report(label_test,predictions))


# In[ ]:




