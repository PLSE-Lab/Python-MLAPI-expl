#!/usr/bin/env python
# coding: utf-8

# > ### Objective

# In this competition you will be predicting whether a question asked on Quora is sincere or not.
# 
# An insincere question is defined as a question intended to make a statement rather than look for helpful answers. Some characteristics that can signify that a question is insincere:
# 
# **Has a non-neutral tone**
# * Has an exaggerated tone to underscore a point about a group of people
# * Is rhetorical and meant to imply a statement about a group of people
# 
# **Is disparaging or inflammatory**
# * Suggests a discriminatory idea against a protected class of people, or seeks confirmation of a stereotype
# * Makes disparaging attacks/insults against a specific person or group of people 
# * Based on an outlandish premise about a group of people 
# * Disparages against a characteristic that is not fixable and not measurable 
# 
# **Isn't grounded in reality**
# * Based on false information, or contains absurd assumptions
# * Uses sexual content (incest, bestiality, pedophilia) for shock value, and not to seek genuine answers

# In[ ]:



import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
from nltk.corpus import stopwords
import string
from sklearn.model_selection import train_test_split                # to split the data
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score   
from sklearn.metrics import classification_report, confusion_matrix

eng_stopwords = set(stopwords.words("english"))
pd.options.mode.chained_assignment = None

import os
print(os.listdir("../input"))


# #### The files that are provided by the kaggle team:

# In[ ]:


get_ipython().system('ls ../input/')


# **File descriptions**
# 
# * train.csv - the training set
# * test.csv - the test set
# * sample_submission.csv - A sample submission in the correct format
# * enbeddings/ - (see below)

# In[ ]:


# List the embeddings provided by kaggle team
get_ipython().system('ls ../input/embeddings/')


# **the below embedding are also downloadable from here**:
# 
# * GoogleNews-vectors-negative300 - https://code.google.com/archive/p/word2vec/
# * glove.840B.300d - https://nlp.stanford.edu/projects/glove/
# * paragram_300_sl999 - https://cogcomp.org/page/resource_view/106
# * wiki-news-300d-1M - https://fasttext.cc/docs/en/english-vectors.html

# In[ ]:


## Read the train and test dataset and check the top few lines ##
train_df = pd.read_csv("../input/train.csv")
test_df = pd.read_csv("../input/test.csv")
print("Number of rows in train dataset : ",train_df.shape[0])
print("Number of rows in test dataset : ",test_df.shape[0]) 


# In[ ]:


train_df.head()


# In[ ]:


#Check for the class-categorization count and also the class imbalance
cnt_srs = train_df['target'].value_counts()

plt.figure(figsize=(8,4))
sns.barplot(cnt_srs.index, cnt_srs.values, alpha=0.8)
plt.ylabel('Number of Occurrences', fontsize=12)
plt.xlabel('target', fontsize=12)
plt.show()


# * As the distribution of target variable is varying a lot, hence accuracy is not the metric we need to look, May be F1-score.

# In[ ]:


# Let us print some lines of each of the questions cagtegory in quora to try and understand their writing style if possible.
grouped_df = train_df.groupby('target')
for name, group in grouped_df:
    print("Target Name :", name)
    cnt =0
    for ind, row in group.iterrows():
        print(row['question_text'])
        cnt += 1
        if cnt == 2:
            break
    print("\n")


# **Feature Engineering:**
#     
# * Now let us come try to do some feature engineering. This consists of two main parts.
# 
# **Meta features -** features that are extracted from the text like number of words, number of stop words, number of punctuations etc
# 
# **Text based features -** features directly based on the text / words like frequency, svd, word2vec etc.

# **Meta Features:**
#     
#  We will start with creating meta featues and see how good are they at predicting the  authors. 
# 
#  The feature list is as follows:
#         
# * Number of words in the text
# * Number of unique words in the text
# * Number of characters in the text
# * Number of stopwords 
# * Number of punctuations
# * Number of upper case words
# * Number of title case words
# * Average length of the words

# In[ ]:


# Number of words in the text 
train_df["num_words"] = train_df["question_text"].apply(lambda x: len(str(x).split()))
test_df["num_words"] = test_df["question_text"].apply(lambda x: len(str(x).split()))

## Number of unique words in the text ##
train_df["num_unique_words"] = train_df["question_text"].apply(lambda x: len(set(str(x).split())))
test_df["num_unique_words"] = test_df["question_text"].apply(lambda x: len(set(str(x).split())))

## Number of characters in the text ##
train_df["num_chars"] = train_df["question_text"].apply(lambda x: len(str(x)))
test_df["num_chars"] = test_df["question_text"].apply(lambda x: len(str(x)))

## Number of stopwords in the text ##
train_df["num_stopwords"] = train_df["question_text"].apply(lambda x: len([w for w in str(x).lower().split() if w in eng_stopwords]))
test_df["num_stopwords"] = test_df["question_text"].apply(lambda x: len([w for w in str(x).lower().split() if w in eng_stopwords]))

## Number of punctuations in the text ##
train_df["num_punctuations"] =train_df['question_text'].apply(lambda x: len([c for c in str(x) if c in string.punctuation]) )
test_df["num_punctuations"] =test_df['question_text'].apply(lambda x: len([c for c in str(x) if c in string.punctuation]) )

## Number of title case words in the text ##
train_df["num_words_upper"] = train_df["question_text"].apply(lambda x: len([w for w in str(x).split() if w.isupper()]))
test_df["num_words_upper"] = test_df["question_text"].apply(lambda x: len([w for w in str(x).split() if w.isupper()]))

## Number of title case words in the text ##
train_df["num_words_title"] = train_df["question_text"].apply(lambda x: len([w for w in str(x).split() if w.istitle()]))
test_df["num_words_title"] = test_df["question_text"].apply(lambda x: len([w for w in str(x).split() if w.istitle()]))

## Average length of the words in the text ##
train_df["mean_word_len"] = train_df["question_text"].apply(lambda x: np.mean([len(w) for w in str(x).split()]))
test_df["mean_word_len"] = test_df["question_text"].apply(lambda x: np.mean([len(w) for w in str(x).split()]))


# * Let us now plot some of our new variables to see of they will be helpful in predictions.

# In[ ]:


train_df.head(3)


# In[ ]:


train_df.shape


# In[ ]:


train_df['num_words'].loc[train_df['num_words']>50] = 50 #truncation for better visuals
plt.figure(figsize=(10,6))
sns.boxplot(x='target', y='num_words', data=train_df)
plt.xlabel('target category', fontsize=12)
plt.ylabel('Number of words in text', fontsize=12)
plt.title("Number of words by target category", fontsize=15)
plt.show()


# In[ ]:


train_df['num_punctuations'].loc[train_df['num_punctuations']>10] = 10 #truncation for better visuals
plt.figure(figsize=(10,6))
sns.boxplot(x='target', y='num_punctuations', data=train_df)
plt.xlabel('target Name', fontsize=12)
plt.ylabel('Number of puntuations in text', fontsize=12)
plt.title("Number of punctuations by target category", fontsize=15)
plt.show()


# In[ ]:


train_df['num_chars'].loc[train_df['num_chars']>300] = 300 #truncation for better visuals
plt.figure(figsize=(10,6))
sns.boxplot(x='target', y='num_chars', data=train_df)
plt.xlabel('target Name', fontsize=12)
plt.ylabel('Number of characters in text', fontsize=12)
plt.title("Number of characters by target category", fontsize=15)
plt.show()


# ![](http://)* In all the cases shown above , we observe that Number of words, Number of punctuations, Number of characters are more for insincere Text. 

# **Modelling will be done in Next stage...stay tuned!!**

# 

# **References**
# 

# https://www.kaggle.com/sudalairajkumar/simple-feature-engg-notebook-spooky-author#
