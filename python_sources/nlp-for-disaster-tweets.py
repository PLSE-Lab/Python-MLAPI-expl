# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import re
from nltk.tokenize import word_tokenize
from string import punctuation 
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer 
from sklearn.linear_model import LogisticRegression

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


df = pd.read_csv('/kaggle/input/nlp-getting-started/train.csv')
df.head()
t = pd.read_csv('/kaggle/input/nlp-getting-started/test.csv')

X = df.iloc[:, 3].values 
y = df.target.values
A = t.iloc[:, 3].values 

# function for cleaning the data
def process(z):
    processed_tweets = []
 
    for tweet in range(0, len(z)):  
        # Remove all the special characters
        processed_tweet = re.sub('((www\.[^\s]+)|(https?://[^\s]+))', ' ', str(z[tweet]))
        processed_tweet = re.sub('@[^\s]+', '', processed_tweet)
        processed_tweet = re.sub('&amp', '', processed_tweet)
        processed_tweet = re.sub(r'[^a-zA-Z]', ' ', processed_tweet)

        # remove all single characters
        processed_tweet = re.sub(r'\s+[a-zA-Z]\s+', ' ', processed_tweet)
        
        # Remove single characters from the start
        processed_tweet = re.sub(r'\^[a-zA-Z]\s+', ' ', processed_tweet) 

        # Removing two letter words  
        processed_tweet = re.sub(r'\b\w{1,2}\b', ' ', processed_tweet)
           
        # Converting to Lowercase
        processed_tweet = processed_tweet.lower()
       
        # Remove the special characters
        processed_tweet = re.sub(r'[וךû×בן]', ' ', processed_tweet)

        # Substituting multiple spaces with single space
        processed_tweet= re.sub(r'\s+', ' ', processed_tweet, flags=re.I)

        # Remove spaces from start and end
        processed_tweet = re.sub(r'^\s|\s$', '', processed_tweet)

        processed_tweets.append(processed_tweet)
    return processed_tweets;    

train_tweets = process(X)
test_tweets = process(A)

tfidfconverter = TfidfVectorizer(max_features=3000, min_df=4, max_df=0.9, stop_words=stopwords.words('english'))  
a = tfidfconverter.fit_transform(train_tweets).toarray()
X_test = tfidfconverter.transform(test_tweets).toarray()

#LogisticRegression model
logmodel=LogisticRegression(solver='lbfgs')
logmodel.fit(a, y)
predictions = logmodel.predict(X_test)

submission = pd.DataFrame() 
submission['id'] = t.iloc[:, 0].values 
submission['target'] = predictions.astype(int)
submission.to_csv('submission.csv', index=False)


