#!/usr/bin/env python
# coding: utf-8

# Objective of this analysis was to find out if lemmatizing 'verb', 'adjective', and 'adverb' to their root words will improve the classification accuracy of predicting recommended or not based on review text.

# In[ ]:


import pandas as pd
import numpy as np
import nltk
from nltk import word_tokenize, FreqDist


# ## 1) Data Understanding & Data Preparation

# In[ ]:


df = pd.read_csv('../input/Womens Clothing E-Commerce Reviews.csv')
df.head()


# In[ ]:


df = df.replace(np.nan, "")
df.head()


# In[ ]:


df.shape


# In[ ]:


df.dtypes


# In[ ]:


df.describe()


# In[ ]:


tokens = word_tokenize(df['Review Text'].to_string())


# In[ ]:


len(tokens)


# In[ ]:


len(set(tokens))


# There were total of 281,146 tokens, of which 31,291 were unique tokens.

# In[ ]:


fd = nltk.FreqDist(tokens)
fd.plot(30)


# ## 2a) Pre-process 1 function

# To predict recommended or not based on review text, we first need to pre-process the text before inputting it into the various classification algorithms. There were 2 pre-processing steps attempted.
# 
# Pre-process 1
# 1. Lowercase the text
# 2. Tokenize the text
# 3. Remove the tokens if it is less than 3 characters
# 4. Lemmatize the tokens (default option used - only tokens with part-of-speech = 'noun' were lemmatized)
# 
# Pre-process 2
# 1. Lowercase the text
# 2. Tokenize the text
# 3. Remove the tokens if it is less than 3 characters
# 4. Lemmatize the tokens (tokens with part-of-speech = 'noun', 'verb', 'adjective', 'adverb' were lemmatized)

# In[ ]:


# WNlemma = nltk.WordNetLemmatizer()

# def pre_process(text):
#     text = text.lower()
#     tokens = nltk.word_tokenize(text)
#     tokens = [t for t in tokens if len(t) > 2]
#     tokens = [WNlemma.lemmatize(t) for t in tokens]
#     text_after_process = " ".join(tokens)
#     return text_after_process


# ## 2b) Pre-process 2 function

# In[ ]:


WNlemma = nltk.WordNetLemmatizer()
from nltk.corpus import wordnet
from nltk import pos_tag

def get_wordnet_pos(treebank_tag):
    if treebank_tag.startswith('V'):
        return wordnet.VERB
    elif treebank_tag.startswith('J'):
        return wordnet.ADJ
    elif treebank_tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN

def pre_process_with_pos_tag(text):
    text = text.lower()
    tokens = nltk.word_tokenize(text)
    tokens = [t for t in tokens if len(t) > 2]
    tokens = [WNlemma.lemmatize(t, get_wordnet_pos(pos_tag(word_tokenize(t))[0][1])) for t in tokens]
    text_after_process = " ".join(tokens)
    return text_after_process


# In[ ]:


# review_text_processed = df['Review Text'].apply(pre_process)
review_text_processed = df['Review Text'].apply(pre_process_with_pos_tag)
review_text_processed


# In[ ]:


review_text_processed_df = pd.DataFrame(
    {
        'review_text_processed':review_text_processed,
        'recommended':df['Recommended IND']
    },
    columns = ['review_text_processed','recommended']
)
review_text_processed_df.head()


# ## 3) Splitting to train & test sets

# After the pre-processing steps, split the dataset into 80% train and 20% test.

# In[ ]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(review_text_processed_df.review_text_processed,
                                                    review_text_processed_df.recommended,
                                                    test_size = 0.2,
                                                    random_state = 5205
                                                   )


# ## 4a) Term Frequency document-term matrix

# There were 2 ways we can construct the document-term matrix - (i) term frequency, (ii) term frequency-inverse document frequency (tf-idf). Term frequency is just the raw count of a term in a document. Tf-idf increases proportionally to the number of times a word appears in the document and is offset by number of documents in the corpus that contain the word.

# In[ ]:


from sklearn.feature_extraction.text import CountVectorizer
count_vect = CountVectorizer()

X_train_counts = count_vect.fit_transform(X_train)
X_train_counts.shape


# In[ ]:


dtm = pd.DataFrame(X_train_counts.toarray().transpose(), index=count_vect.get_feature_names())
dtm = dtm.transpose()
dtm.head()


# ## 4b) Term frequency-inverse document frequency document-term matrix

# In[ ]:


from sklearn.feature_extraction.text import TfidfTransformer
tf_transformer = TfidfTransformer().fit(X_train_counts)
X_train_tf = tf_transformer.transform(X_train_counts)
X_train_tf.shape


# ## 5a) Naive bayes model

# Experimented with 4 classification algorithms - (i) naive bayes, (ii) decision tree, (iii) support vector machine, (iv) logistic regression. Metrics used to evaluate the models are - (i) classification accuracy, (ii) F1 score.

# In[ ]:


from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn import metrics

nb_clf = Pipeline([('vect', CountVectorizer()),
                   ('tfidf', TfidfTransformer()),
                   ('clf', MultinomialNB())
                  ])
nb_clf.fit(X_train, y_train)
nb_predicted = nb_clf.predict(X_test)

print(metrics.confusion_matrix(y_test, nb_predicted))
print(np.mean(nb_predicted==y_test))
print(metrics.classification_report(y_test, nb_predicted))


# ## 5b) Decision tree model

# In[ ]:


from sklearn import tree
dt_clf = Pipeline([('vect', CountVectorizer()),
                   ('tfidf', TfidfTransformer()),
                   ('clf', tree.DecisionTreeClassifier())
                  ])
dt_clf.fit(X_train, y_train)
dt_predicted = dt_clf.predict(X_test)

print(metrics.confusion_matrix(y_test, dt_predicted))
print(np.mean(dt_predicted==y_test))
print(metrics.classification_report(y_test, dt_predicted))


# ## 5c) Support vector machine model

# In[ ]:


from sklearn import svm
from sklearn.svm import SVC
from sklearn.linear_model import SGDClassifier

svm_clf = Pipeline([('vect', CountVectorizer()),
                    ('tfidf', TfidfTransformer()),
                    ('clf', svm.LinearSVC(C=1.0))
                   ])
svm_clf.fit(X_train, y_train)
svm_predicted = svm_clf.predict(X_test)

print(metrics.confusion_matrix(y_test, svm_predicted))
print(np.mean(svm_predicted==y_test))
print(metrics.classification_report(y_test, svm_predicted))


# ## 5d) Logistic regression model

# In[ ]:


from sklearn.linear_model import LogisticRegression
lr_clf = Pipeline([('vect', CountVectorizer()),
                   ('tfidf', TfidfTransformer()),
                   ('clf', LogisticRegression())
                  ])
lr_clf.fit(X_train, y_train)
lr_predicted = lr_clf.predict(X_test)

print(metrics.confusion_matrix(y_test, lr_predicted))
print(np.mean(lr_predicted==y_test))
print(metrics.classification_report(y_test, lr_predicted))


# Comparing the 4 classification models, SVM and logistic regression models had better F1 scores as compared with naive bayes and decision tree models.

# In[ ]:


lr_clf = LogisticRegression()
# lr_clf.fit(dtm, y_train)
lr_clf.fit(X_train_tf, y_train)

lr_clf_coef = (
    pd.DataFrame(lr_clf.coef_[0], index=dtm.columns)
    .rename(columns={0:'Coefficient'})
    .sort_values(by='Coefficient', ascending=False)
)

lr_clf_coef.head()

