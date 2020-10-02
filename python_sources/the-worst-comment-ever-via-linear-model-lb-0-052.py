#!/usr/bin/env python
# coding: utf-8

# In this notebook we will do two simple things:
# 1. Train a linear model (logistic regression) for each class a make a submission
# 2. Generate the most neutral and the most toxic comment with the help of trained model.
# 
# Let's go!

# In[ ]:


import re
import numpy as np
import pandas as pd
from scipy import sparse

from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer

CLASSES = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']


# In[ ]:


train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')

# Tets data contains one NaN, so we have to reaplce it with something
test.fillna(' ', inplace=True)


# To work with the text data we first need to normalize it. Here is very simple normalizer. That's how it works:
# 1. Convert everything to the lowercase
# 2. Replace all new lines and tabs with whitespaces
# 3. Replace all non-alphanumerical characters with space (leave only letters, numbers and underscores)
# 4. Replace subsequent whitespaces with single space
# 5. Strip the line (remove whitespace from the begginign and end)
# 
# The normalized text will be put to he column named `normalized`

# In[ ]:


def normalize(text):
    text = text.lower()
    text = text.replace('\n', ' ').replace('\t', ' ')
    text = re.sub('\W', ' ', text)
    text = re.sub('\s+', ' ', text)
    text = text.strip()
    return text


# In[ ]:


train['normalized'] = train.comment_text.map(normalize)
test['normalized'] = test.comment_text.map(normalize)


# We will use two different text vectorizer. One will works on the word level and one on the char level. This will allow us to find interesting interactions and generalize on the grammar level.

# In[ ]:


vect_words = TfidfVectorizer(max_features=5000)
vect_chars = TfidfVectorizer(max_features=1000, analyzer='char', ngram_range=(1, 3))


# In[ ]:


# Creating features
Xtrain_words = vect_words.fit_transform(train.normalized)
Xtrain_chars = vect_chars.fit_transform(train.normalized)

# Combine two different types of features into single sparse matrix
Xtrain = sparse.hstack([Xtrain_words, Xtrain_chars])


# Train the models and save them to the dict to use later

# In[ ]:


models = {}
for toxicity in CLASSES:
    # I encourage you to change this C=5.0 and try different regularization
    lm = LogisticRegression(C=5.0, random_state=42)  
    lm.fit(Xtrain, train[toxicity])
    models[toxicity] = lm
    print("Model for %s trained" % toxicity, flush=True)


# Create predictions for the test set

# In[ ]:


Xtest_words = vect_words.transform(test.normalized)
Xtest_chars = vect_chars.transform(test.normalized)

Xtest = sparse.hstack([Xtest_words, Xtest_chars])

predictions = pd.DataFrame(test.id)
for toxicity in CLASSES:
    predictions[toxicity] = models[toxicity].predict_proba(Xtest)[:, 1]
    
predictions.to_csv('./submission.csv', index=False)


# Now, when the submission is created and the job is done, let's have some fun. We can extract coefficients for every word and character n-gram and see how the contribute to the toxicity of the comment. Thus, we can just sum coefficients for all classes of toxicity to get the most neutral and the most toxic words. And we will use them to generate **the worst comment ever**. What can go wrong?

# In[ ]:


coefficients = pd.DataFrame(index=vect_words.get_feature_names() + vect_chars.get_feature_names())
for toxicity in CLASSES:
    coefficients[toxicity] = models[toxicity].coef_[0]
    
coefficients['total'] = coefficients.sum(1)
coefficients.sort_values('total', inplace=True)


# Ready for the most neutral and non-toxic comment? Here it is:

# In[ ]:


# Let's randomly permutate top-10 words to amke it more interesting
np.random.seed(1)
' '.join(np.random.permutation(coefficients.head(10).index))


# The worst one? Let's go!

# In[ ]:


np.random.seed(1)
' '.join(np.random.permutation(coefficients.tail(10).index))


# It's like watching the *Blood & Concrete: A Love Story* opening scene. I hope I'll not be banned on Kaggle for this notebook :) In my defence I can only say: *"Hey! It's in the data!"*
