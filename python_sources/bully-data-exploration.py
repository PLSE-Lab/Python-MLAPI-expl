#!/usr/bin/env python
# coding: utf-8

# In[ ]:


get_ipython().system('apt-get -y install libenchant-dev')


# In[ ]:


get_ipython().system('pip install pyenchant')


# In[ ]:


import re
import enchant
import nltk

from nltk.corpus import stopwords, words
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tokenize import word_tokenize

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')


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


url = "/kaggle/input/formspring-data-for-cyberbullying-detection/formspring_data.csv"
df = pd.read_csv(url, sep='\t')
df = df[df['ans'].notna()]


# In[ ]:


df.columns


# ### Three Results
# The data has three results, and I would like to only view data where all three results match.

# In[ ]:


sum(df.ans1 == df.ans2) / len(df)


# In[ ]:


sum(df.ans2 == df.ans3) / len(df)


# In[ ]:


sum(df.ans1 == df.ans3) / len(df)


# In[ ]:


index = (df.ans1 == df.ans2) & (df.ans2 == df.ans3) & (df.ans1 == df.ans3)


# In[ ]:


df = df[index]


# In[ ]:


df.head()


# In[ ]:


# Putting back apostrophe's may help
df.ques = df.ques.str.replace("&#039;", "'") 

# Removing may not be effective since there is a word check implemented
df.ques = df.ques.str.replace("<br>", "") 
df.ques = df.ques.str.replace("&quot;", "") 
df.ques = df.ques.str.replace("<3", "") 


# In[ ]:


bully_df = df[df.ans1 == "Yes"].reset_index(drop=True)

df[df.ans1 == "Yes"].head()


# ### Tokenizer
# Build a custom function that can gather actual words from a give string.

# In[ ]:


stop_words = stopwords.words("english")
lemmatizer = WordNetLemmatizer()
enchant_dict = enchant.Dict("en_US")

def tokenize(text):
    # normalize case and remove punctuation
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())
    
    # tokenize text
    tokens = word_tokenize(text)
    
    # lemmatize andremove stop words
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    
    #tokens = [word for word in tokens if enchant_dict.check(word)]

    return tokens


# In[ ]:


for i in range(0,6):
    print(bully_df.ques[i])
    print(tokenize(bully_df.ques[i]),"\n")


# ### Whole Process
# The process includes tokenizing the input strings and creating pipelines for the ML model.

# In[ ]:


import nltk
nltk.download(['punkt', 'wordnet'])

import re
import numpy as np
import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer


# In[ ]:


def load_data(path):
    df = pd.read_csv(path, sep='\t')
    df = df[df['ans'].notna()]
    
    index = (df.ans1 == df.ans2) & (df.ans2 == df.ans3) & (df.ans1 == df.ans3)
    df = df[index].reset_index(drop=True)
    df = df[["ques","ans","ans1"]]
    
    X = df.ans.values
    y = df.ans1.values
    return X, y

def tokenize(text):

    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens

def display_results(y_test, y_pred):
    labels = np.unique(y_pred)
    confusion_mat = confusion_matrix(y_test, y_pred, labels=labels)
    accuracy = (y_pred == y_test).mean()

    print("Labels:", labels)
    print("Confusion Matrix:\n", confusion_mat)
    print("Accuracy:", accuracy)
    
def main():
    url = "/kaggle/input/formspring-data-for-cyberbullying-detection/formspring_data.csv"
    X, y = load_data(url)
    X_train, X_test, y_train, y_test = train_test_split(X, y)

    vect = CountVectorizer(tokenizer=tokenize)
    tfidf = TfidfTransformer()
    clf = RandomForestClassifier()

    # train classifier
    X_train_counts = vect.fit_transform(X_train)
    X_train_tfidf = tfidf.fit_transform(X_train_counts)
    clf.fit(X_train_tfidf, y_train)

    # predict on test data
    X_test_counts = vect.transform(X_test)
    X_test_tfidf = tfidf.transform(X_test_counts)
    y_pred = clf.predict(X_test_tfidf)
    
    # predict on test data
    X_test_counts = vect.transform(["whoa stop you stupid bitch"])
    X_test_tfidf = tfidf.transform(X_test_counts)
    print("Given text: 'whoa stop you stupid bitch' ")
    print("Prediction: {}\n".format(clf.predict(X_test_tfidf)))
    

    # display results
    display_results(y_test, y_pred)


main()


# In[ ]:




