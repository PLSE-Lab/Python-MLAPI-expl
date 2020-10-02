#!/usr/bin/env python
# coding: utf-8

# # Real or Not? NLP with Disaster Tweets
# 
# Machine learning can be very usefull, but take a long time to train and test. Others approches exist and can also have pretty good result. In this kernel, we are going to see how to implement the TF-IDF method and use a SVM network to do the classification.
# 
# ## Competition 
# 
# In this competition, we want to predict which Tweets are about real disasters and which ones are not. Then for each tweet (input), we want to know if it refers to a disaster event (output). The output of our system will be a boolean (true or false).
# 
# - Explore our data.
# - Preprocess our data
# - TF-IDF implementation.
# - SVM.
# - Result and tuning.

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


import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold, KFold, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score


# # Explore our data

# In[ ]:


# Read the data 
import pandas as pd

df = pd.read_csv("/kaggle/input/nlp-getting-started/train.csv")
df_test  = pd.read_csv("/kaggle/input/nlp-getting-started/test.csv")

# Show some training data
df.head()


# In[ ]:


df_test.head()


# As you can see, in our training data, we have 3 features :
# - Keyword : a particular keyword from the tweet (may be blank)
# - Location : the location the tweet was sent from (may be blank)
# - Text : the text of the tweet
# 
# In our approach, we will use only the **keyword** and the **text** features. 

# In[ ]:


print("Training :")
print("Length of the data :", len(df))
# Missing value in the training set
print(df.isnull().sum())


# In[ ]:


print("Test :")
print("Length of the data :", len(df_test))
# Missing value in the test set
print(df_test.isnull().sum())


# In[ ]:


# Distributy if the target 
target_values = df['target'].value_counts()
sns.barplot(target_values.index, target_values)
plt.gca().set_ylabel('samples')


# In[ ]:


# df['target_mean'] = df.groupby('keyword')['target'].transform('mean')

# fig = plt.figure(figsize=(8, 72), dpi=100)

# sns.countplot(y=df.sort_values(by='target_mean', ascending=False)['keyword'],
#               hue=df.sort_values(by='target_mean', ascending=False)['target'])

# plt.tick_params(axis='x', labelsize=15)
# plt.tick_params(axis='y', labelsize=12)
# plt.legend(loc=1)
# plt.title('Target Distribution in Keywords')

# plt.show()

# df.drop(columns=['target_mean'], inplace=True)


# # Text preprocessing - Normalization
# 
# In this part, we are going to normalize our data. Firt of all, we are going to **tokenize** our data. We are going to split our phrase into array of word. We need to separate the word and also the ponctuation.

# In[ ]:


from nltk.tokenize import word_tokenize

# Extract all the words
tokens = word_tokenize(df["text"][0])

# Lowercase the words
tokens = [word.lower() for word in tokens]

print(df["text"][0])
print(tokens)


# Then, we are going to remove all the ponctuation.

# In[ ]:


# Remove all tokens that are not alphabetic
words = [word for word in tokens if word.isalpha()]
print(words)


# In order to describe our tweet with word, we need to remove the "stopwords". That means all the basic word like 'the', 'you'... These words are present in all documents, so they can't be use to differentiate tweet. 

# In[ ]:


# Filters - Remove stop words
from nltk.corpus import stopwords

# Get all stop words
stop_words = set(stopwords.words("english"))

words = [word for word in words if not word in stop_words]
print(words)


# In[ ]:


# Stem Words (Racinisation)
# Process of reducins inflected words to their word stem, base or root form.
from nltk.stem.porter import PorterStemmer

porter = PorterStemmer()
stemmed = [porter.stem(word) for word in words]

print(stemmed)


# In[ ]:


import re
import string

#Function for removing URL
def remove_URL(text):
    url = re.compile(r'https?://\S+|www\.\S+')
    return url.sub(r'',text)

#Function for removing HTML codes
def remove_html(text):
    html=re.compile(r'<.*?>')
    return html.sub(r'',text)

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


#Function for removing punctuations
def remove_punct(text):
    table=str.maketrans('','',string.punctuation)
    return text.translate(table)

df['text']=df['text'].apply(remove_URL)
df['text']=df['text'].apply(remove_html)
df['text']=df['text'].apply(remove_emoji)
df['text']=df['text'].apply(remove_punct)


df_test['text'] = df_test['text'].apply(remove_URL)
df_test['text'] = df_test['text'].apply(remove_html)
df_test['text'] = df_test['text'].apply(remove_emoji)
df_test['text'] = df_test['text'].apply(remove_punct)


# In[ ]:


from nltk.stem import WordNetLemmatizer

# Get all stop words
stop_words = set(stopwords.words("english"))
porter = PorterStemmer()
lemmatizer = WordNetLemmatizer()

def preprocess_text(text):
    
    # Extract all the words
    tokens = word_tokenize(text)
    
    # Lowercase the words
    tokens = [word.lower() for word in tokens]
    
    # Remove all tokens that are not alphabetic
    words = [word for word in tokens if word.isalpha()]
    
    # Remove word in the stop word
    words = [word for word in words if not word in stop_words]

    # Get the root of the word 
    stemmed = [porter.stem(word) for word in words]
    
    # Lematize the word
    lematized = [lemmatizer.lemmatize(word) for word in stemmed]

    return lematized

df["preprocess_text"] = df.text.apply(preprocess_text)
df_test["preprocess_text"] = df_test.text.apply(preprocess_text)
df.head()


# In[ ]:


def join_list(tab):
    return " ".join(tab)
df["text_preprocessed"] = df["preprocess_text"].apply(join_list)
df_test["text_preprocessed"] = df_test["preprocess_text"].apply(join_list)

def transform_keyword(word) :
    # Split when %20
    return word.split('%20')

# Transform NaN value to empty string
df["keyword"] = df.keyword.fillna(" ")
df_test["keyword"] = df_test.keyword.fillna(" ")

df["keyword"] = df["keyword"].apply(transform_keyword).apply(join_list)
df_test["keyword"] = df_test["keyword"].apply(transform_keyword).apply(join_list)

# Concant keyword to the phrases
df["text_preprocessed"] = df["keyword"] + " " + df["text_preprocessed"] 
df_test["text_preprocessed"] = df_test["keyword"] + " " + df_test["text_preprocessed"] 


# # TF-IDF 

# In[ ]:


from sklearn.model_selection import train_test_split

X_all = pd.concat([df["text_preprocessed"], df_test["text_preprocessed"]])

tfidf = TfidfVectorizer(stop_words = 'english')
tfidf.fit(X_all)

X = tfidf.transform(df["text_preprocessed"])
X_test = tfidf.transform(df_test["text_preprocessed"])
del X_all

train, test = train_test_split(df, test_size=0.2)

train_x = train["text_preprocessed"]
train_y = train["target"]

test_x = test["text_preprocessed"]
test_y = test["target"]


X_train, X_val, y_train, y_val = train_test_split(X, df["target"], test_size=0.1, random_state=42)


# # SVM

# In[ ]:


parameters = { 
    'gamma': [0.001, 0.01, 0.1, 0.4, 0.5, 0.6, 0.7, 1], 
    'kernel': ['rbf'], 
    'C': [0.001, 0.01, 0.1, 1, 1.5, 2, 3, 10],
}

# {'C': 2, 'gamma': 0.9, 'kernel': 'rbf'}


# In[ ]:


# parameters = { 
#     'gamma':  [0.5],
#     'kernel': ['rbf'], 
#     'C':[2]
# }
model = GridSearchCV(SVC(), parameters, cv=10, n_jobs=-1).fit(X_train, y_train)


# In[ ]:


model.cv_results_['params'][model.best_index_]


# In[ ]:


y_val_pred = model.predict(X_val)
accuracy_score(y_val, y_val_pred), f1_score(y_val, y_val_pred)


# In[ ]:


confusion_matrix(y_val, y_val_pred)


# In[ ]:


y_test_pred = model.predict(X_test)


# In[ ]:


sub_df = pd.read_csv(os.path.join('../input/nlp-getting-started/', 'sample_submission.csv'))
sub_df["target"] = y_test_pred
sub_df.to_csv("submission.csv",index=False)

