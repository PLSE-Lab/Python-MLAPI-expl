#!/usr/bin/env python
# coding: utf-8

# # Getting Started
# First off, we import the necessary packages and set the random seed in order to get consistent results

# In[17]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

get_ipython().run_line_magic('matplotlib', 'notebook')

np.random.seed(19)


# In[18]:


train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')
print("Training samples = ", train.shape[0])
print("Test samples = ", test.shape[0])


# In[19]:


train_rows =train.shape[0]
labels = train[['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']] 
train.drop(['id', 'toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate'], axis=1, inplace=True)
test_id = test.pop('id')
train.head()


# # Check for the null values
# 
# Now we check for the presences of null values in the training and testing data. Then we'll combine both using pd.concat

# In[20]:


print("Null values in training data", train.isnull().sum(), sep="\n")
print("Null values in testing data", test.isnull().sum(), sep="\n")


# In[21]:


data = pd.concat([train, test])
del train
del test
data.shape


# # Data Preprocessing
# 
# We can see that the comments contain various special characters as well as escape sequences (e.g '\n'). <br>
# Here we are removing the escape sequences, digits and the commonly used words (.i.e. stop words) using regular expressions.  We are interested in keeping the special characters as they are used in emoticons or some other expressions.

# In[22]:


import re
import nltk

stop_words = set(nltk.corpus.stopwords.words('english'))

def preprocess_input(comment):
# remove the extra spaces at the end.
    comment = comment.strip()
# lowercase to avoid difference between 'hate', 'HaTe'
    comment = comment.lower()
# remove the escape sequences. 
    comment = re.sub('[\s0-9]',' ', comment)
# Use nltk's word tokenizer to split the sentence into words. It is better than the 'split' method.
    words = nltk.word_tokenize(comment)
# removing the commonly used words.
    #words = [word for word in words if not word in stop_words and len(word) > 2]
    words = [word for word in words if len(word) > 2]
    comment = ' '.join(words)
    return comment


# In[23]:


print("SAMPLE PREPROCESSING")
print("\nOriginal comment: ", data.comment_text.iloc[0], sep='\n')
print("\nProcessed comment: ", preprocess_input(data.comment_text.iloc[0]), sep='\n')


# Applying the preprocessing on whole input data. This will take sometime as the dataset is large.

# In[24]:


data.comment_text = data.comment_text.apply(lambda row: preprocess_input(row))


# In[25]:


data.head()


# # Feature Extraction
# 
# We use Tfidf Vectorizer to convert the collection of raw documents to a matrix of TF-IDF features. Here the feature will be made of character n-grams rather than word. Because the comments contain various characters which may not be defined in ASCII so we will use Unicode.

# In[26]:


from sklearn.feature_extraction.text import TfidfVectorizer

vect = TfidfVectorizer(min_df=0.1, max_df=0.7, 
                       analyzer='char',
                       ngram_range=(1, 3),
                       strip_accents='unicode',
                       sublinear_tf=True,
                       max_features=30000
                      )


# In[27]:


test = data[train_rows:]
train = data[:train_rows]
del data


# In[28]:


vect = vect.fit(train.comment_text)
train = vect.transform(train.comment_text)
test = vect.transform(test.comment_text)


# In[29]:


print('Training feature set = ', train.shape)
print('Testing feature set = ', test.shape)


# # Applying machine learning
# 
# We have to predict the probability associated with each label.  I am using Logistic Regression for simplicity.

# In[30]:


from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
cols = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
y_pred = pd.read_csv('../input/sample_submission.csv')

for c in cols:
    clf = LogisticRegression(C=4, solver='sag')
    clf.fit(train, labels[c])
    y_pred[c] = clf.predict_proba(test)[:,1]
    score = np.mean(cross_val_score(clf, train, labels[c], scoring='roc_auc', cv=5))
    print("ROC_AUC score for", c, "=",  score)


# In[31]:


y_pred.head()


# In[32]:


y_pred.to_csv('my_submission.csv', index=False)

