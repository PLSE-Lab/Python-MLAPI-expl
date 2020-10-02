#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import plotly
import sklearn
import tensorflow


# In[ ]:


df_train = pd.read_csv('../input/tweet-sentiment-extraction/train.csv')
df_test  = pd.read_csv('../input/tweet-sentiment-extraction/test.csv')


# In[ ]:


df_train.head()


# In[ ]:


df_test.head()


# In[ ]:


print(df_train.shape)
df_test.shape


# In[ ]:


import cufflinks as cf
from plotly.offline import download_plotlyjs,init_notebook_mode,plot,iplot
init_notebook_mode(connected=True)
cf.go_offline()


# In[ ]:


import nltk


# In[ ]:


nltk.download_shell()   

# Stopwords is already installed.


# In[ ]:


from nltk.corpus import stopwords


# In[ ]:


print(df_train.columns)
print(df_test.columns)


# In[ ]:


print(df_train.info())
df_test.info()


# In[ ]:



print(df_train.isna().sum())

# only 2 null values out of 24000 total values.
# let's drop them.


df_test.isna().sum()

# No null values in test dataset.


# In[ ]:


df_train.dropna(inplace=True)


# In[ ]:


df_train.isna().sum()

# No null values left.


# # Exploratory Data Analysis

# In[ ]:


# adding a column of text_length
df_train['text_length'] = df_train['text'].apply(lambda x : len(x))

df_test['text_length'] = df_test['text'].apply(lambda x : len(x))


# In[ ]:



sns.set_style(style='whitegrid')
plt.figure(figsize=(10,5))
sns.distplot(df_train['text_length'],color='green')

# normal distributed data


# In[ ]:


g = sns.FacetGrid(data=df_train,col='sentiment',height=4)
g.map(sns.distplot,'text_length')


# In[ ]:


df_train['sentiment'].value_counts().iplot(kind='bar',color='black')

# Maximum Neutral texts


# In[ ]:


df_test['sentiment'].value_counts().iplot(kind='bar',color='purple')


# In[ ]:


import string


# In[ ]:


print(df_train['text'][4])
df_train['selected_text'][4].split()


# In[ ]:



def sel_tex(i):
    split_text = i.split()
    return split_text


# In[ ]:


df_train['selected_text2'] = df_train['selected_text'].apply(sel_tex)


# In[ ]:


df_train.head()


# # Feature Engineering

# # OPTION 1

# ### Using selected_text column of the Train Dataset for predictions.
# 

# In[ ]:


# selected_text column of test dataset will bo on the basis of selected_text of Train dataset to 
#    predict better for types of messages.


select_text = pd.Series(df_train['selected_text'])


list1 = ' '.join(select_text)


list2 = list1.split()


# In[ ]:


def test_select(i):
    l  = [ ]
    for w in i.split():
        if w in list2:
            l.append(w)
    return(l)


# In[ ]:


df_test['selected_text'] = df_test['text'].apply(test_select)


# In[ ]:


df_test.head(6)


# In[ ]:


df_train.head(1)


# In[ ]:


from sklearn.feature_extraction.text import CountVectorizer


# # Fitting and Training the Model

# In[ ]:


bag_of_words = CountVectorizer(analyzer=test_select).fit(df_test['text'])


# In[ ]:


df_test_bow_trans = bag_of_words.transform(df_test['text'])


# In[ ]:


df_test_bow_trans


# In[ ]:


from sklearn.feature_extraction.text import TfidfTransformer


# In[ ]:


tfidf = TfidfTransformer().fit(df_test_bow_trans)


# In[ ]:


df_test_tfidf = tfidf.transform(df_test_bow_trans)


# In[ ]:


df_test_tfidf.shape


# In[ ]:


from sklearn.naive_bayes import MultinomialNB


# In[ ]:


sentiment_detect_model = MultinomialNB().fit(df_test_tfidf,df_test['sentiment'])


# In[ ]:


all_sentiments_predictions = sentiment_detect_model.predict(df_test_tfidf)


# In[ ]:


from sklearn.metrics import confusion_matrix,classification_report


# In[ ]:


print(confusion_matrix(all_sentiments_predictions,df_test['sentiment']))


# In[ ]:


print(classification_report(all_sentiments_predictions,df_test['sentiment']))


# ACCURACY = 81 %


# # OPTION 2

# ### Adding a new selected_text column in the Test Dataset on the basis of Test Data text column.

# In[ ]:


df_test  = pd.read_csv('../input/tweet-sentiment-extraction/test.csv')


# In[ ]:


df_test.head()


# In[ ]:


df_test['text_length'] = df_test['text'].apply(lambda x : len(x))


# In[ ]:


def test_select(i):
    list_text = [text for text in i if text not in string.punctuation]
    join_test_text = ''.join(list_text)
    clean_test_text = [ text for text in join_test_text.split() if text.lower() not in stopwords.words('english')]
    return clean_test_text


# In[ ]:


df_test['selected_text'] = df_test['text'].apply(test_select)


# In[ ]:


df_test.head()


# In[ ]:


bag_of_words = CountVectorizer(analyzer=test_select).fit(df_test['text'])


df_test_bow_trans = bag_of_words.transform(df_test['text'])


tfidf = TfidfTransformer().fit(df_test_bow_trans)


df_test_tfidf = tfidf.transform(df_test_bow_trans)


sentiment_detect_model = MultinomialNB().fit(df_test_tfidf,df_test['sentiment'])


all_sentiments_predictions = sentiment_detect_model.predict(df_test_tfidf)


# In[ ]:


print(confusion_matrix(all_sentiments_predictions,df_test['sentiment']))


# In[ ]:


print(classification_report(all_sentiments_predictions,df_test['sentiment']))


# ACCURACY = 91 %


# In[ ]:


# Therefore , option 2 has increased accuracy by 10%.


# ## Option 1 = 81 %
# 
# ## Option 2 = 91 %

# # Submission

# In[ ]:


df_test.head(2)


# In[ ]:


def joined(i):
    joined = " , ".join(i)
    return joined


# In[ ]:


df_test['selected_text2'] = df_test['selected_text'].apply(joined)


# In[ ]:


df_test.head()


# In[ ]:


df_test2 = df_test[['textID','selected_text2']]


# In[ ]:


df_test2.rename(columns={'selected_text2':'selected_text'},inplace=True)


# In[ ]:


df_test2.head(1)


# In[ ]:


df_test2.to_csv('submission.csv',index=False)


# In[ ]:




