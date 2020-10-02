#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import matplotlib.pyplot as plt
from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"


# In[ ]:


train_df = pd.read_csv('../input/train.csv')
test_df = pd.read_csv('../input/test.csv')


# In[ ]:


train_df.shape


# In[ ]:


train_df.info()


# In[ ]:


train_df.target.value_counts()/train_df.target.value_counts().sum()


# In[ ]:


train_df['Number_of_words'] = train_df.question_text.apply(lambda x: len(str(x).split()))


# In[ ]:


train_df['Char_count'] = train_df['question_text'].str.len()


# In[ ]:


train_df.head()


# In[ ]:


def avg_word(sentence):
    words = sentence.split()
    return (sum(len(word) for word in words)/len(words))


# In[ ]:


train_df['avg_word'] = train_df['question_text'].apply(lambda x:avg_word(x))


# In[ ]:


from nltk.corpus import stopwords
stop = stopwords.words('english')


# In[ ]:


train_df['stopwords'] = train_df['question_text'].apply(lambda x: len([x for x in x.split() if x in stop]))


# In[ ]:


train_df.head()


# <b> Basic Preprocessing </b>

# In[ ]:


train_df['question_text'] = train_df['question_text'].apply(lambda x: " ".join(x.lower() for x in x.split()))
test_df['question_text'] = test_df['question_text'].apply(lambda x: " ".join(x.lower() for x in x.split()))


# In[ ]:


train_df.head()


# In[ ]:


train_df['question_text'] = train_df['question_text'].str.replace('[^\w\s]','')
test_df['question_text'] = test_df['question_text'].str.replace('[^\w\s]','')


# In[ ]:


train_df['question_text'] = train_df['question_text'].apply(lambda x: " ".join(x for x in x.split() if x not in stop))
test_df['question_text'] = test_df['question_text'].apply(lambda x: " ".join(x for x in x.split() if x not in stop))

train_df.head()


# In[ ]:


freq_train = pd.Series(' '.join(train_df['question_text']).split()).value_counts()[:30]
train_df['question_text'] = train_df['question_text'].apply(lambda x: ' '.join([i for i in x.split() if i not in freq_train]))
freq_train = pd.Series(' '.join(train_df['question_text']).split()).value_counts()[-30:]
train_df['question_text'] = train_df['question_text'].apply(lambda x: ' '.join([i for i in x.split() if i not in freq_train]))
freq_test = pd.Series(' '.join(test_df['question_text']).split()).value_counts()[:30]
test_df['question_text'] = test_df['question_text'].apply(lambda x: ' '.join([i for i in x.split() if i not in freq_test]))
freq_test = pd.Series(' '.join(test_df['question_text']).split()).value_counts()[-30:]
test_df['question_text'] = test_df['question_text'].apply(lambda x: ' '.join([i for i in x.split() if i not in freq_test]))


# In[ ]:


train_df['Number_of_words'] = train_df.question_text.apply(lambda x: len(str(x).split()))
train_df['Char_count'] = train_df.question_text.str.len()
test_df['Number_of_words'] = test_df.question_text.apply(lambda x: len(str(x).split()))
test_df['Char_count'] = test_df.question_text.str.len()
train_df.drop(['avg_word','stopwords'],axis=1,inplace=True)
train_df[train_df['target']==1].head(20)


# In[ ]:


from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.model_selection import train_test_split    


# In[ ]:


y = train_df['target']
X = train_df['question_text']
X_train,X_test,y_train,y_test = train_test_split(X,y,random_state=1,test_size=0.3)


# In[ ]:


train_df['question_text'].apply(lambda x: len(x.split(' '))).sum()
test_df['question_text'].apply(lambda x: len(x.split(' '))).sum()


# In[ ]:


from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfTransformer,CountVectorizer

nb = Pipeline([('vect', CountVectorizer()),
               ('tfidf', TfidfTransformer()),
               ('clf', MultinomialNB()),
              ])
nb.fit(X_train, y_train)
from sklearn.metrics import classification_report,accuracy_score
y_pred = nb.predict(X_test)

print('accuracy %s' % accuracy_score(y_pred, y_test))
print(classification_report(y_test, y_pred))


# **Linear SVM**

# In[ ]:


from sklearn.linear_model import SGDClassifier

sgd = Pipeline([('vect', CountVectorizer()),
                ('tfidf', TfidfTransformer()),
                ('clf', SGDClassifier(loss='hinge', penalty='l2',alpha=1e-3, random_state=42, max_iter=5, tol=None)),
               ])
sgd.fit(X_train, y_train)
y_pred = sgd.predict(X_test)

print('accuracy %s' % accuracy_score(y_pred, y_test))
print(classification_report(y_test, y_pred))


# In[ ]:


from sklearn.linear_model import LogisticRegression

logreg = Pipeline([('vect', CountVectorizer()),
                ('tfidf', TfidfTransformer()),
                ('clf', LogisticRegression(C=1e3)),
               ])
logreg.fit(X_train, y_train)

y_pred = logreg.predict(X_test)

print('accuracy %s' % accuracy_score(y_pred, y_test))
print(classification_report(y_test, y_pred))


# In[ ]:


# from tqdm import tqdm
# tqdm.pandas(desc="progress-bar")
# from gensim.models import Doc2Vec
# from sklearn import utils
# import gensim
# from gensim.models.doc2vec import TaggedDocument
# import re

# def label_sentences(corpus, label_type):
#     """
#     Gensim's Doc2Vec implementation requires each document/paragraph to have a label associated with it.
#     We do this by using the TaggedDocument method. The format will be "TRAIN_i" or "TEST_i" where "i" is
#     a dummy index of the post.
#     """
#     labeled = []
#     for i, v in enumerate(corpus):
#         label = label_type + '_' + str(i)
#         labeled.append(TaggedDocument(v.split(), [label]))
#     return labeled
# X_train, X_test, y_train, y_test = train_test_split(train_df['question_text'], train_df['target'], random_state=0, test_size=0.3)
# X_train = label_sentences(X_train, 'Train')
# X_test = label_sentences(X_test, 'Test')
# all_data = X_train + X_test


# In[ ]:


# model_dbow = Doc2Vec(dm=0, vector_size=300, negative=5, min_count=1, alpha=0.065, min_alpha=0.065)
# model_dbow.build_vocab([x for x in tqdm(all_data)])

# for epoch in range(30):
#     model_dbow.train(utils.shuffle([x for x in tqdm(all_data)]), total_examples=len(all_data), epochs=1)
#     model_dbow.alpha -= 0.002
#     model_dbow.min_alpha = model_dbow.alpha


# In[ ]:


# import  numpy as np
# def get_vectors(model, corpus_size, vectors_size, vectors_type):
#     """
#     Get vectors from trained doc2vec model
#     :param doc2vec_model: Trained Doc2Vec model
#     :param corpus_size: Size of the data
#     :param vectors_size: Size of the embedding vectors
#     :param vectors_type: Training or Testing vectors
#     :return: list of vectors
#     """
#     vectors = np.zeros((corpus_size, vectors_size))
#     for i in range(0, corpus_size):
#         prefix = vectors_type + '_' + str(i)
#         vectors[i] = model.docvecs[prefix]
#     return vectors
    
# train_vectors_dbow = get_vectors(model_dbow, len(X_train), 300, 'Train')
# test_vectors_dbow = get_vectors(model_dbow, len(X_test), 300, 'Test')


# In[ ]:


# logreg = LogisticRegression(n_jobs=1, C=1e5)
# logreg.fit(train_vectors_dbow, y_train)
# logreg = logreg.fit(train_vectors_dbow, y_train)
# y_pred = logreg.predict(test_vectors_dbow)


# In[ ]:


# print('accuracy %s' % accuracy_score(y_pred, y_test))
# print(classification_report(y_test, y_pred))


# In[ ]:


y_pred1 = logreg.predict(test_df['question_text'])
y_pred1.shape


# In[ ]:


sub = pd.read_csv('../input/sample_submission.csv')
sub['prediction']=y_pred1
sub.to_csv('submission.csv',index=False)

