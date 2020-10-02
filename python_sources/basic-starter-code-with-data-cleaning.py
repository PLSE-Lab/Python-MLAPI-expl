#!/usr/bin/env python
# coding: utf-8

# # Import Libraries

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from nltk.corpus import stopwords

import re
import string

from sklearn import feature_extraction, linear_model, model_selection, preprocessing
from sklearn.metrics import accuracy_score


# In[ ]:


stop=set(stopwords.words('english'))


# In[ ]:


train = pd.read_csv("/kaggle/input/nlp-getting-started/train.csv")
test = pd.read_csv("/kaggle/input/nlp-getting-started/test.csv")
#train = pd.read_csv("train.csv")
#test = pd.read_csv("test.csv")


# In[ ]:


df = pd.concat([train, test], sort = False)
df.shape


# # Clean Data

# In[ ]:


#Function for removing URL
def remove_URL(text):
    url = re.compile(r'https?://\S+|www\.\S+')
    return url.sub(r'',text)


# In[ ]:


#Function for removing HTML codes
def remove_html(text):
    html=re.compile(r'<.*?>')
    return html.sub(r'',text)


# In[ ]:


#Function for removing Emojis
def remove_emoji(text):
    emoji_pattern = re.compile("["
                           u"\U0001F600-\U0001F64F"  # emoticons
                           u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                           u"\U0001F680-\U0001F6FF"  # transport & map symbols
                           u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                           u"\U00002702-\U000027B0"
                           u"\U000024C2-\U0001F251"
                           "]+", flags=re.UNICODE)
    return emoji_pattern.sub(r'', text)


# In[ ]:


#Function for removing punctuations
def remove_punct(text):
    table=str.maketrans('','',string.punctuation)
    return text.translate(table)


# In[ ]:


df['text']=df['text'].apply(lambda x : remove_URL(x))
df['text']=df['text'].apply(lambda x : remove_html(x))
df['text']=df['text'].apply(lambda x : remove_emoji(x))
df['text']=df['text'].apply(lambda x : remove_punct(x))


# In[ ]:


df.head()


# In[ ]:


df_train = df[df['target'].notnull()]
df_train.head()


# In[ ]:


df_test = df[df['target'].isnull()]
df_test.head()


# # Make Test and Train

# In[ ]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(df_train['text'], df_train['target'], test_size=0.30, random_state=101)


# # Building Vectors

# In[ ]:


#Create instance
count_vectorizer = feature_extraction.text.CountVectorizer(analyzer = 'word', max_features = 5000)


# In[ ]:


#Fit Transform train data
X_train_fit = count_vectorizer.fit(X_train)
X_test_fit = count_vectorizer.fit(X_test)


# In[ ]:


X_train_vectors = count_vectorizer.transform(X_train)
X_test_vectors = count_vectorizer.transform(X_test)


# In[ ]:


X = pd.DataFrame(X_train_vectors.toarray())
X.columns = count_vectorizer.get_feature_names()


# In[ ]:


#X.to_csv('vector.csv', index = False)
X.head()


# # Model

# In[ ]:


clf = linear_model.RidgeClassifier()


# In[ ]:


clf.fit(X, y_train)


# In[ ]:


scores = model_selection.cross_val_score(clf, X, y_train, cv=7, scoring="f1")
y_test_pred = clf.predict(X_test_vectors)
test_accuracy = accuracy_score(y_test_pred, y_test)
print('Train Accuracy : %0.3f' % scores.mean())
print('Test Accuracy : %0.3f' % test_accuracy.mean())


# # Predict

# In[ ]:


#get sample file for creating submission file
sample_submission = pd.read_csv("/kaggle/input/nlp-getting-started/sample_submission.csv")
#sample_submission = pd.read_csv("sample_submission.csv")


# In[ ]:


X_sub_vecttor = count_vectorizer.fit_transform(df_test['text'])


# In[ ]:


sample_submission["target"] = clf.predict(X_sub_vecttor).astype(int)


# In[ ]:


sample_submission.head()


# # Submit

# In[ ]:


#Got to the Output section of this Kernel -> click on Submit to Competition
sample_submission.to_csv("submission.csv", index=False)


# In[ ]:




