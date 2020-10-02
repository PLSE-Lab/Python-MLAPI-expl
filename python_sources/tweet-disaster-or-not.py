#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
import re
from nltk.stem import WordNetLemmatizer
from nltk.corpus import names, stopwords
import unicodedata
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report
from sklearn.linear_model import LogisticRegression


# In[ ]:





# # Importing Files

# In[ ]:


tweet_train_df = pd.read_csv('../input/nlp-getting-started/train.csv')
tweet_test_df = pd.read_csv('../input/nlp-getting-started/test.csv')
Id = tweet_test_df['id']


# # Data Familiarization

# In[ ]:


tweet_train_df.head()


# Keyword and location  contains lots of Missing values.

# In[ ]:


tweet_train_df.isna().sum()


# In[ ]:


tweet_train_df.info()


# In[ ]:


tweet_train_df.shape


# 7613 rows of tweets with 4 columns id, keyword, location and text all object

# # Data Imputation

# In[ ]:


def data_imputation(data):
    data['keyword'].fillna(' ', inplace=True)
    data['location'].fillna(' ', inplace=True)
    return data


# In[ ]:


tweet_train_df = data_imputation(tweet_train_df)


# In[ ]:


tweet_test_df = data_imputation(tweet_test_df)


# In[ ]:


tweet_train_df.isna().sum()


# In[ ]:


tweet_train_df['text'] = tweet_train_df['text'] +' '+ tweet_train_df['location'] +' '+ tweet_train_df['keyword']
tweet_test_df['text'] = tweet_test_df['text'] +' '+ tweet_test_df['location'] +' '+ tweet_test_df['keyword']


# All the missing columns are filled with appropriate values, later maybe we will change value of keyword dynamically for each row accordingly for now it's Na.
# Missing locations are set to Unknown

# # Text Filtering
# Filtering Keyword, location and text columns by removing numbers, hashtags, names, Url and mentions

# In[ ]:


target = tweet_train_df['target']


# In[ ]:


tweet_train_df.drop(['target', 'location', 'keyword', 'id'], axis=1, inplace=True)
tweet_test_df.drop(['location', 'keyword', 'id'], axis=1, inplace=True)


# In[ ]:


lemmetizer = WordNetLemmatizer()


# In[ ]:


all_names = set(names.words())


# In[ ]:


stop_words = set(stopwords.words('english'))


# In[ ]:


tf_idf = TfidfVectorizer(min_df=0.1, max_df=0.7)


# In[ ]:


def cleaned_string(string):
    # Removing all the digits
    string = re.sub(r'\d', '', string)
    
    # Removing accented data
    string = unicodedata.normalize('NFKD', string).encode('ascii', 'ignore').decode('utf-8', 'ignore')
    
    # Removing Mentions
    string = re.sub(r'@\w+', ' ', string)
    
    # Removing links 
    string = re.sub(r'(https?:\/\/)?([\da-zA-Z\.-\/\#\:]+)\.([\da-zA-Z\.\/\:\#]{0,9})([\/\w \.-\/\:\#]*)', ' ', string)
    
    # Removing all the digits special caharacters
    string = re.sub(r'\W', ' ', string)
        
    
    # Removing double whitespaces
    string = re.sub(r'\s+', ' ', string, flags=re.I)
    

    
    string = string.strip()
    
    #Removing all Single characters
    string = re.sub(r'\^[a-zA-Z]\s+','' , string)
    
    
    # Lemmetizing the string and removing stop words
    string = string.split()
    string = [lemmetizer.lemmatize(word) for word in string if word not in stop_words and word not in all_names]
    string = ' '.join(string)
    
    # Lowercasing all data
    string = string.lower()
        
    return string


# In[ ]:


def clean_text(data):
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            data.iloc[i, j] = cleaned_string(data.iloc[i, j])
    return data
            
            
    


# In[ ]:


tweet_cleaned_test_df = clean_text(tweet_test_df)


# In[ ]:


tweet_cleaned_test_df.shape


# In[ ]:


tweet_cleaned_test_df.head()


# In[ ]:


tweet_cleaned_train_df = clean_text(tweet_train_df)


# In[ ]:


tweet_train_df.shape


# In[ ]:


tweet_cleaned_train_df.head()


# In[ ]:


X_train, X_valid, y_train, y_valid = train_test_split(tweet_cleaned_train_df['text'], target,random_state = 0)


# In[ ]:


catboost = LogisticRegression()


# In[ ]:


pipeline_sgd = Pipeline([
    ('tfidf',  TfidfVectorizer()),
    ('nb', catboost,)
])


# In[ ]:


model = pipeline_sgd.fit(X_train, y_train)


# In[ ]:


y_predict = model.predict(X_valid)


# In[ ]:


print(classification_report(y_valid, y_predict))


# In[ ]:


y_pred_test = model.predict(tweet_cleaned_test_df['text'])


# In[ ]:


# Saving result on test set
output = pd.DataFrame({'Id': Id,
                       'target': y_pred_test})

output.to_csv(r'submission.csv', index=False)


# In[ ]:





# In[ ]:




