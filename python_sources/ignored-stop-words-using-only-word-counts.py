#!/usr/bin/env python
# coding: utf-8

# ### This is a simple solution using only word counts with CountVectorizer to make predictions.
# 
# #### Here's the idea:
# - Find and weight words that are used most often in only certain kinds of tweets.
# - Search all subsets of the tweet and calculate a score based on these weights.
# - For positive or negative tweets, the selected text is the most highly weighted subset, within some threshold.
# - Always return the entire text for neutral tweets.

# In[ ]:


import pandas as pd 
import numpy as np
import nltk

# CountVectorizer will help calculate word counts
from sklearn.feature_extraction.text import CountVectorizer

# Import the string dictionary that we'll use to remove punctuation
import string


# In[ ]:


# Import datasets

train = pd.read_csv('/kaggle/input/tweet-sentiment-extraction/train.csv')
test = pd.read_csv('/kaggle/input/tweet-sentiment-extraction/test.csv')
sample = pd.read_csv('/kaggle/input/tweet-sentiment-extraction/sample_submission.csv')


# In[ ]:


# The row with index 13133 has NaN text, so remove it from the dataset

train[train['text'].isna()]


# In[ ]:


train = train.dropna()
train[train['text'].isna()]


# In[ ]:


a=['negative','positive']


# In[ ]:


X_train = train.loc[train['sentiment'].isin(a)]


# In[ ]:


import nltk 
from nltk.corpus import stopwords
import re
from bs4 import BeautifulSoup
from nltk.tokenize import word_tokenize, sent_tokenize 

stop_words = set(stopwords.words('english'))
REPLACE_BY_SPACE_RE = r'[)\!#?.:";-^/(]'
tokenized = sent_tokenize(train['text'][2]) 
stop_words = nltk.corpus.stopwords.words('english')
stop_words.remove('no')
stop_words.remove('not')              
print(stop_words)

def clean_text(text):
    wordsList = ''.join([c for c in text.lower() if text.lower().split() not in stop_words])
    soup = BeautifulSoup(wordsList, "html.parser")
    wordsList = soup.get_text()
    wordsList = re.sub(REPLACE_BY_SPACE_RE,' ',wordsList)
    wordsList = ' '.join(word for word in wordsList.split())
    return wordsList
    

X_train['selected_text'] = X_train['selected_text'].apply(lambda x: clean_text(x))


# Create a training set and a validation set.

# In[ ]:


# Make training/test split
from sklearn.model_selection import train_test_split

X_train, X_val = train_test_split(
    train, train_size = 0.9, random_state = 21)


# In[ ]:


X_train[:20]


# In[ ]:


pos_train = X_train[X_train['sentiment'] == 'positive']
neutral_train = X_train[X_train['sentiment'] == 'neutral']
neg_train = X_train[X_train['sentiment'] == 'negative']


# ### Algorithm for weight calculation:
# 
# 1. For each class $j \in \{positive, neutral, negative\}$
# 
#     a. Find all the words $i$ in the tweets belonging to class $j$.
# 
#     b. Calculate $n_{i, j} =$ the number of tweets in class $j$ containing word $i$. 
# 
#     c. Let $d_j$ be the number of tweets in class $j$.  Calculate $p_{i, j} = \frac{n_{i, j}}{d_j}$, the proportion of tweets in class $j$ that conain word $i$.
# 
#     d. Let $w_{i, j} = p_{i, j} - \sum\limits_{k \neq j}p_{i, k}$ be the weights assigned to each word within each class. 
#     

# In[ ]:


# Use CountVectorizer to get the word counts within each dataset

cv = CountVectorizer(analyzer='word',max_df=0.95)

X_train_cv = cv.fit_transform(X_train['text'])

X_pos = cv.transform(pos_train['text'])
X_neutral = cv.transform(neutral_train['text'])
X_neg = cv.transform(neg_train['text'])


pos_count_df = pd.DataFrame(X_pos.toarray(), columns=cv.get_feature_names())
neutral_count_df = pd.DataFrame(X_neutral.toarray(), columns=cv.get_feature_names())
neg_count_df = pd.DataFrame(X_neg.toarray(), columns=cv.get_feature_names())


# In[ ]:


X_train_cv


# In[ ]:



# Create dictionaries of the words within each sentiment group, where the values are the proportions of tweets that 
# contain those words

pos_words = {}
neutral_words = {}
neg_words = {}

for k in cv.get_feature_names():
    pos = pos_count_df[k].sum()
    neutral = neutral_count_df[k].sum()
    neg = neg_count_df[k].sum()
    pos_words[k] = pos/pos_train.shape[0]
    neutral_words[k] = neutral/neutral_train.shape[0]
    neg_words[k] = neg/neg_train.shape[0]
    
# We need to account for the fact that there will be a lot of words used in tweets of every sentiment.  
# Therefore, we reassign the values in the dictionary by subtracting the proportion of tweets in the other 
# sentiments that use that word.
neg_words_adj = {}
pos_words_adj = {}
neutral_words_adj = {}

for key, value in neg_words.items():
    neg_words_adj[key] = neg_words[key] - (neutral_words[key] + pos_words[key])
        
for key, value in pos_words.items():
    pos_words_adj[key] = pos_words[key] - (neutral_words[key] + neg_words[key])
    
for key, value in neutral_words.items():
    neutral_words_adj[key] = neutral_words[key] - (neg_words[key] + pos_words[key])


# ### Algorithm for finding selected text: 
#   
# 1. For every tweet:
# 
#     a. Let $j$ be the sentiment of the tweet. 
# 
#     b. If $j ==$ neutral return entire text.
# 
#     c. Otherwise, for each subset of words in the tweet, calculate $\sum\limits_{i}w_{i, j}$, where $i$ is the set of words in the tweet
# 
#    d. Return the subset of words with the largest sum, given that it exceeds some tolerance.

# In[ ]:


def calculate_selected_text(df_row, tol = 0):
    
    tweet = df_row['text']
    sentiment = df_row['sentiment']
    
    if(sentiment == 'neutral'):
        return tweet
    
    elif(sentiment == 'positive'):
        dict_to_use = pos_words_adj # Calculate word weights using the pos_words dictionary
    elif(sentiment == 'negative'):
        dict_to_use = neg_words_adj # Calculate word weights using the neg_words dictionary
        
    words = tweet.split()
    words_len = len(words)
    subsets = [words[i:j+1] for i in range(words_len) for j in range(i,words_len)]
        
    score = 0
    selection_str = '' # This will be our choice
    lst = sorted(subsets, key = len) # Sort candidates by length
    
    
    for i in range(len(subsets)):
        
        new_sum = 0 # Sum for the current substring
        
        # Calculate the sum of weights for each word in the substring
        for p in range(len(lst[i])):
            if(lst[i][p].translate(str.maketrans('','',string.punctuation)) in dict_to_use.keys()):
                new_sum += dict_to_use[lst[i][p].translate(str.maketrans('','',string.punctuation))]
                
            
        # If the sum is greater than the score, update our current selection
        if(new_sum > score + tol):
            score = new_sum
            selection_str = lst[i]
            #tol = tol*5 # Increase the tolerance a bit each time we choose a selection

    # If we didn't find good substrings, return the whole text
    if(len(selection_str) == 0):
        selection_str = words                   
        
    return ' '.join(selection_str)


# Calculate the selected text and score for the validation set.

# In[ ]:


pd.options.mode.chained_assignment = None


# In[ ]:


def jaccard(str1, str2): 
    a = set(str1.lower().split()) 
    b = set(str2.lower().split())
    c = a.intersection(b)
    return float(len(c)) / (len(a) + len(b) - len(c))


# ### Generate Submission

# Recalculate word weights using the entire training set.

# In[ ]:


pos_tr = train[train['sentiment'] == 'positive']
neutral_tr = train[train['sentiment'] == 'neutral']
neg_tr = train[train['sentiment'] == 'negative']


# In[ ]:


cv = CountVectorizer(analyzer='word',max_df=0.95, min_df=2,stop_words='english')

final_cv = cv.fit_transform(train['text'])

X_pos = cv.transform(pos_tr['text'])
X_neutral = cv.transform(neutral_tr['text'])
X_neg = cv.transform(neg_tr['text'])

pos_final_count_df = pd.DataFrame(X_pos.toarray(), columns=cv.get_feature_names())
neutral_final_count_df = pd.DataFrame(X_neutral.toarray(), columns=cv.get_feature_names())
neg_final_count_df = pd.DataFrame(X_neg.toarray(), columns=cv.get_feature_names())


# In[ ]:


pos_words = {}
neutral_words = {}
neg_words = {}

for k in cv.get_feature_names():
    pos = pos_final_count_df[k].sum()
    neutral = neutral_final_count_df[k].sum()
    neg = neg_final_count_df[k].sum()
    
    pos_words[k] = pos/(pos_tr.shape[0])
    neutral_words[k] = neutral/(neutral_tr.shape[0])
    neg_words[k] = neg/(neg_tr.shape[0])


# In[ ]:


neg_words_adj = {}
pos_words_adj = {}
neutral_words_adj = {}

for key, value in neg_words.items():
    neg_words_adj[key] = neg_words[key] - (neutral_words[key] + pos_words[key])
    
for key, value in pos_words.items():
    pos_words_adj[key] = pos_words[key] - (neutral_words[key] + neg_words[key])
    
for key, value in neutral_words.items():
    neutral_words_adj[key] = neutral_words[key] - (neg_words[key] + pos_words[key])


# Create and submit the submission file.

# In[ ]:


tol = 0.001

for index, row in test.iterrows():
    
    selected_text = calculate_selected_text(row, tol)
    
    sample.loc[sample['textID'] == row['textID'], ['selected_text']] = selected_text
    


# In[ ]:


sample.to_csv('submission.csv', index = False)


# In[ ]:


sample.head()


# In[ ]:




