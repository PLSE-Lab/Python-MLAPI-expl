#!/usr/bin/env python
# coding: utf-8

# **INTRODUCTION**
# 
# Hi, in this notebook, I'll just doing exercise of using nltk and naive bayes classification algorithm to do a *very simple Spam/Ham* Classification from SMS dataset from UCI

# In[ ]:


import nltk
from nltk.corpus import stopwords
import string
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report,confusion_matrix


# In[ ]:


messages = pd.read_csv('../input/Email.csv', encoding='latin-1')
messages.head()


# In[ ]:



messages.drop(['Unnamed: 2','Unnamed: 3','Unnamed: 4'],axis=1,inplace=True)
messages = messages.rename(columns={'v1': 'class','v2': 'text'})


# **PART 1: DATA PREPROCESSING**
# 
# since the dataset comes with additional unnamed, column, I need to drop them first

# In[ ]:


messages.head()


# In[ ]:


messages.groupby('class').describe()


# from above information, we know that:
# 1. only about 15% of the text messages is classified as a spam
# 2. there are some duplicate messages, since the number of unique values lower than the count values of the text
# 
# in the next part, lext check the length of each text messages to see whether it is correlated with the text classified as a spam or not.

# In[ ]:


messages['length'] = messages['text'].apply(len)


# In[ ]:


messages.hist(column='length',by='class',bins=60, figsize=(15,5))


# from above figure, we can see that most of ham (or not spam) messages only have length under 200 (100 to be exact) while spam messages tend to have higher lentgh above 130 or 140 approximately.

# **PART 2: CREATE TOKENIZER**

# In[ ]:


def process_text(text):
    '''
    What will be covered:
    1. Remove punctuation
    2. Remove stopwords
    3. Return list of clean text words
    '''
    
    #1
    nopunc = [char for char in text if char not in string.punctuation]
    nopunc = ''.join(nopunc)
    
    #2
    clean_words = [word for word in nopunc.split() if word.lower() not in stopwords.words('english')]
    
    #3
    return clean_words


# let's check what above code will produce

# In[ ]:


messages['text'].apply(process_text).head()


# **PART 3: SPLITTING DATASET**

# In[ ]:


msg_train, msg_test, class_train, class_test = train_test_split(messages['text'],messages['class'],test_size=0.2)


# **PART 4: DATA PREPROCESSING**
# 
# wait, we've just created the tokenizer isn't it? let the pipeline do the rest.

# **PART 5: MODEL CREATION**
# 
# here I'll just use pipeline in order to minimize effort on doing preprocessing, transforming then training data on both training dataset and test dataset. Using pipeline will handle them all in a few lines of codes.

# In[ ]:


pipeline = Pipeline([
    ('bow',CountVectorizer(analyzer=process_text)), # converts strings to integer counts
    ('tfidf',TfidfTransformer()), # converts integer counts to weighted TF-IDF scores
    ('classifier',MultinomialNB()) # train on TF-IDF vectors with Naive Bayes classifier
])


# **PART 6: TESTING**

# In[ ]:


pipeline.fit(msg_train,class_train)


# In[ ]:


predictions = pipeline.predict(msg_test)


# In[ ]:


print(classification_report(class_test,predictions))


# In[ ]:


import seaborn as sns
sns.heatmap(confusion_matrix(class_test,predictions),annot=True)


# **Notes:**
# * we got fairly high but not good enough prediction result here, maybe if the dataset gets higher, maybe naive bayes will do its work better
# 
# thanks :)

# In[ ]:


from sklearn.metrics import accuracy_score

print('Accuracy: %.5f' % accuracy_score(class_test,predictions))

