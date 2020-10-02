#!/usr/bin/env python
# coding: utf-8

# ## Summary
# 
# The title of this kernel is quite bold and, as all bold affirmations it is not completely true but, it is a good clickbait, isn't it? :-)
# 
# I think we only need to identify the white space character. There are some interesting kernels trying to decrypt all the texts but, at least for the case of difficulty 1 (and maybe 2), it wouldn't be strictly neccessary. If we knew how the white space is encrypted we could tokenize properly (to models like Bag of Words the real words doesn't matter: if chars substitution has been applied it should work the same)
# 
# I have tried to know what are the most probable encripted white space character using brute force: my assumption is that, the same model (Logistic Regression) should perform better if we tokenize using the right character. 

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from collections import Counter
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import os
print(os.listdir("../input"))


# In[ ]:


data = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')
subm = pd.read_csv('../input/sample_submission.csv')


# In[ ]:


data.head()


# In the case of difficulty = 1, from this [kernel](https://www.kaggle.com/mithrillion/enigma-was-gimped-by-weather-reports), I saw that the '1' could be the encrypted version of the white space character. Let's check:

# In[ ]:


data_1 = data.query('difficulty==1')


# In[ ]:


data_1.head()


# In[ ]:


alp = pd.Series(Counter(''.join(data_1['ciphertext'])))
alp.head(10)


# In[ ]:


len(alp)


# In[ ]:


from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, classification_report, confusion_matrix
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import FunctionTransformer
from sklearn.model_selection import GridSearchCV


# In[ ]:


X = data_1.drop('target', axis=1)
y = data_1['target']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, stratify=y, random_state=0)


# In[ ]:


def tokenize(text): 
    return text.split("1")

vectorizer = CountVectorizer(
    analyzer = 'word',
    tokenizer = tokenize,
    lowercase = False,
    ngram_range=(1, 1))

estimator = LogisticRegression(random_state=0)


# In[ ]:


model = Pipeline([('selector', FunctionTransformer(lambda x: x['ciphertext'], validate=False)),
                  ('vectorizer', vectorizer), 
                  ('tfidf', TfidfTransformer()),
                  ('estimator', estimator)])


# In[ ]:


def generate_tokenizer(separator):
    def tokenizer(text):
        return text.split(separator)
    return tokenizer


# I am going to use the tokenizer for the delimiter "1":

# In[ ]:


tokenize_1 = generate_tokenizer("1")

model.steps[1][1].set_params(tokenizer=tokenize_1)


# In[ ]:


model.fit(X_train, y_train)


# In[ ]:


y_pred = model.predict(X_test)


# In[ ]:


f1_score(y_test, y_pred, average='macro')


# It would be necessary to put this score in context: how the other characters could perform? Let's see:

# In[ ]:


def evaluate_delimiters(data):    
    X = data.drop('target', axis=1)
    y = data['target']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=.2, stratify=y, random_state=0)
    
    scores = {}
    
    # let's get all the chars that are used:
    alp = pd.Series(Counter(''.join(data['ciphertext'])))

    for c in alp.keys():
        tokenize = generate_tokenizer(c)
        model.steps[1][1].set_params(tokenizer=tokenize)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        score = f1_score(y_test, y_pred, average='macro')
        scores[c] = score
    return pd.Series(scores).sort_values(ascending=False)


# ### Difficulty 1

# In[ ]:


scores_difficulty_1 = evaluate_delimiters(data.query('difficulty==1'))


# Let's see the delimiters with highest performance. It is pretty clear that '1' could be the delimiter we are looking for:

# In[ ]:


scores_difficulty_1[:10]


# Let's see now for the other difficulties:

# ### Difficulty 2

# In[ ]:


scores_difficulty_2 = evaluate_delimiters(data.query('difficulty==2'))


# The character '8' could be the encrypted version of the white space delimiter:

# In[ ]:


scores_difficulty_2[:10]


# ### Difficulty 3

# In[ ]:


scores_difficulty_3 = evaluate_delimiters(data.query('difficulty==3'))


# For difficulty 3 is not clear. This approach could not be suitable for this hader encryptation:

# In[ ]:


scores_difficulty_3[:10]


# ### Difficulty 4

# In[ ]:


scores_difficulty_4 = evaluate_delimiters(data.query('difficulty==4'))


# As it happend with difficulty 3, the results for difficulty 4 aren't not very clear:

# In[ ]:


scores_difficulty_4[:10]


# ## Conclusion
# Without decrypting the full texts we have found that:
# - **Difficulty 1**: the character '1' is the best delimiter to tokenize the texts. It could be the encrypted version of the white space character.
# - **Difficulty 2**: the character '8' is the best delimiter to tokenize the texts. It could be the encrypted version of the white space character.
# 
# For the rest of the cases (difficulty 3 and 4) we can't be completely sure.
# The model used doesn't provide a great performance but, that wasn't the objective here. Just to identify the possible delimiters to improve the tokenization. There is room to improve the model from that base.

# ## Appendix
# Now that the first cipher has been broken, we are going to see the performance of our model using plain text.  We are going to use the translation table obtained in this nice [kernel](https://www.kaggle.com/rturley/a-first-crack-tools-and-first-cipher-solution) to decrypt completely the texts with difficulty 1:

# In[ ]:


book = {'1': ' ',
 '\x1b': 'e',
 't': 't',
 'O': 'a',
 '^': 'o',
 'a': 'i',
 '\x02': 'n',
 'v': 's',
 '#': 'r',
 '0': 'h',
 '8': 'l',
 's': '\n',
 'A': 'd',
 '_': 'c',
 'c': 'u',
 '-': 'm',
 '\x08': '.',
 'q': '-',
 "'": 'p',
 'd': 'g',
 'o': 'y',
 ']': 'f',
 'W': 'w',
 '\x03': 'b',
 'T': ',',
 'z': 'v',
 ':': 'I',
 '[': '>',
 'f': 'k',
 'G': ':',
 'L': '1',
 '>': 'S',
 '{': 'T',
 '/': 'A',
 '\\': '0',
 '2': 'C',
 'y': ')',
 'e': 'M',
 ';': "'",
 '|': '(',
 'Z': '=',
 'H': '2',
 '\x1c': '*',
 '\x1e': 'R',
 'x': 'D',
 '\x7f': 'N',
 '%': 'O',
 'Q': '\t',
 '9': 'P',
 'E': 'E',
 'F': 'L',
 ')': 'E',
 'u': '3',
 'b': '@',
 'J': 'B',
 '6': '"',
 'g': 'H',
 '*': 'F',
 '<': '9',
 '\t': '5',
 ',': '4',
 '+': 'x',
 'l': 'W',
 'X': 'j',
 '5': '6',
 '"': 'G',
 'n': '8',
 '@': 'U',
 '&': '?',
 'h': 'z',
 '?': '/',
 '\x06': '7',
 '}': 'J',
 '4': 'J',
 'P': '!',
 'w': 'K',
 '\x18': 'V',
 '\x10': 'Y',
 '!': 'X',
 '(': 'Y',
 ' ': '<',
 '\x1a': 'q',
 '`': '>',
 '.': '#',
 'B': '$',
 '~': '+',
 '3': ';',
 'V': 'Q',
 'm': 'q',
 '\x0c': '%',
 'U': '[',
 'i': ']',
 'r': '&',
 'K': 'Z',
 'Y': '~',
 'I': '}',
 'k': '{',
 'S': '\r',
 '$': '\x08',
 'p': '\x02'}


# In[ ]:


dec_table = str.maketrans(book)


# In[ ]:


data_1_clean = data.query('difficulty==1').copy()


# In[ ]:


data_1_clean['ciphertext'] = data_1_clean['ciphertext'].map(lambda x: x.translate(dec_table))


# In[ ]:


data_1_clean['ciphertext'][1]


# In[ ]:


X = data_1_clean.drop('target', axis=1)
y = data_1_clean['target']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, stratify=y, random_state=0)


# In[ ]:


tokenize_ws = generate_tokenizer(" ")
model.steps[1][1].set_params(tokenizer=tokenize_ws)
model.fit(X_train, y_train)


# In[ ]:


y_pred = model.predict(X_test)


# We obtain almost the same performance than working with the encrypted text and guessing the right delimiter to tokenize: 

# In[ ]:


f1_score(y_test, y_pred, average='macro')


# Although we get a similar performance, if we have the plain text it is possible to get better models thanks to preprocessing (stemming,  lemmatization, ...)
