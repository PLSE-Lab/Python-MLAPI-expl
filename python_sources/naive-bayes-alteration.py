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


# Make all the text lowercase - casing doesn't matter when 
# we choose our selected text.
train['text'] = train['text'].apply(lambda x: x.lower())
test['text'] = test['text'].apply(lambda x: x.lower())

import re
import time
start_time=time.time()
# remove '\\n'
train['text'] = train['text'].map(lambda x: re.sub('\\n',' ',str(x)))
    
# remove any text starting with User... 
train['text'] = train['text'].map(lambda x: re.sub("\[\[User.*",'',str(x)))
    
# remove IP addresses or user IDs
train['text'] = train['text'].map(lambda x: re.sub("\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}",'',str(x)))
    
#remove http links in the text
train['text'] = train['text'].map(lambda x: re.sub("(http://.*?\s)|(http://.*)",'',str(x)))

end_time=time.time()
print("total time",end_time-start_time)


# Create a training set and a validation set.

# In[ ]:


# Make training/test split
from sklearn.model_selection import train_test_split

X_train, X_val = train_test_split(
    train, train_size = 0.9, random_state = 0)


# In[ ]:


import re
from nltk.stem import WordNetLemmatizer 

words = nltk.corpus.stopwords.words('english')
words.remove('no')
words.remove('not')
wn = WordNetLemmatizer() 

a = ['i','to','the','a','my','you','and','it','is','in','for','im','of','me','on','so','have','that','be','its','with','day','at','was']

i=1


def clean_txt(txt):
    new = ' '.join([w for w in txt.split() if len(w)>2])
    dup = re.sub(r'\b(\w+)( \1\b)+', r'\1', new)
    unw = " ".join([u for u in dup.split() if u.lower() not in a])
    txt = " ".join([wn.lemmatize(word) for word in unw.split()])
    alp = " ".join([d for d in txt.split() if d.lower() in words or not d.isalpha()])
    return alp

X_train['selected_text'] = X_train['selected_text'].apply(lambda x: clean_txt(x))


# In[ ]:


X_train[:10]

Break up the training data into datasets where the sentiment is positive, neutral, or negative
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

cv = CountVectorizer(analyzer='word',max_df=0.90, min_df=0.00,stop_words='english')

X_train_cv = cv.fit_transform(X_train['text'])

X_pos = cv.transform(pos_train['text'])
X_neutral = cv.transform(neutral_train['text'])
X_neg = cv.transform(neg_train['text'])


pos_count_df = pd.DataFrame(X_pos.toarray(), columns=cv.get_feature_names())
neutral_count_df = pd.DataFrame(X_neutral.toarray(), columns=cv.get_feature_names())
neg_count_df = pd.DataFrame(X_neg.toarray(), columns=cv.get_feature_names())


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


# In[ ]:


def calculate_selected_text(df_row, alpha = 0):
    #LaPlace Vars
    a = 1
    v_a = a * 100
    
    tweet = df_row['text']
    words = tweet.split()
    sentiment = df_row['sentiment']
    max_len = -1
    if(sentiment == 'neutral'):
        if(len(words) < 6):
            return tweet
        dict_to_use = neutral_words_adj
    elif(sentiment == 'positive'):
        dict_to_use = pos_words_adj # Calculate word weights using the pos_words dictionary
        max_len = 1
    elif(sentiment == 'negative'):
        dict_to_use = neg_words_adj # Calculate word weights using the neg_words dictionary
        
    words_len = len(words)
    #if(max_len == 1 and (words_len < 12 or words_len > 20)):
    #    subsets = [words[i-1:i] for i in range(words_len+1)]
    #else:
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
        if(new_sum > score + alpha):
            score = new_sum
            selection_str = lst[i]
            alpha += 0.001

    # If we didn't find good substrings, return the whole text
    if(len(selection_str) == 0):
        selection_str = words   
        #if(max_len == 1):
        #    selection_str = words[0:1]
        
    return ' '.join(selection_str)


# Calculate the selected text and score for the validation set.

# In[ ]:


pd.options.mode.chained_assignment = None


# In[ ]:


alpha = 0.0015

X_val['predicted_selection'] = ''

for index, row in X_val.iterrows():
    
    selected_text = calculate_selected_text(row, alpha)
    
    X_val.loc[X_val['textID'] == row['textID'], ['predicted_selection']] = selected_text


# In[ ]:


def jaccard(str1, str2): 
    a = set(str1.lower().split()) 
    b = set(str2.lower().split())
    c = a.intersection(b)
    return float(len(c)) / (len(a) + len(b) - len(c))


# In[ ]:


X_val['jaccard'] = X_val.apply(lambda x: jaccard(x['selected_text'], x['predicted_selection']), axis = 1)

print('The jaccard score for the validation set is:', np.mean(X_val['jaccard']))


# In[ ]:


X_val


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




