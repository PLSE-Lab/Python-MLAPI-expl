#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import scipy as sp


# In[2]:


train = pd.read_csv('../input/train.csv.gz')
train_checks = pd.read_csv('../input/train_checks.csv.gz')

train.fillna('', inplace=True)


# In[3]:


from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.wrappers.scikit_learn import KerasClassifier
from keras.utils import np_utils
from keras import regularizers
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, RobustScaler, Normalizer, LabelBinarizer
from sklearn.pipeline import make_pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics import log_loss


# In[5]:


tfidf_chars = TfidfVectorizer(analyzer='char', ngram_range=(2,7), max_features=10000)
tfidf_words = TfidfVectorizer(ngram_range=(1,2), max_features=100000)


# In[6]:


X = sp.sparse.hstack((tfidf_chars.fit_transform(train.name), tfidf_words.fit_transform(train.name)))

labeler = LabelEncoder()
y = labeler.fit_transform(train.category)


# In[7]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, shuffle=True, random_state=17)


# In[9]:


from keras.optimizers import Adam

def baseline_model():
    # create model
    model = Sequential()
    model.add(Dense(1024, input_dim=X.shape[1], activation='sigmoid', ))
    model.add(Dropout(0.2))
    model.add(Dense(256, activation='sigmoid'))
    model.add(Dropout(0.1))
    model.add(Dense(25, activation='softmax'))
    # Compile model
    model.compile(loss='categorical_crossentropy', optimizer=Adam(lr = 0.0001), metrics=['accuracy'])
    return model

estimator = KerasClassifier(build_fn=baseline_model, epochs=30, batch_size=30, verbose=2)


# In[ ]:





# In[ ]:





# estimator.fit(x=X_train.tocsr(), y=y_train, shuffle=True, validation_data=(X_test.tocsr(), pd.get_dummies(y_test)))

# score = log_loss(y_test, estimator.predict_proba(X_test))
# score

# In[16]:





# In[21]:


best_epochs = 30
# Retrain with full data


# In[22]:


estimator = KerasClassifier(build_fn=baseline_model, epochs=best_epochs, batch_size=30, verbose=2)

estimator.fit(x=X.tocsr(), y=pd.get_dummies(y), shuffle=True)


# In[23]:


test = pd.read_csv('../input/test.csv.gz')
test_checks = pd.read_csv('../input/test_checks.csv.gz')


X_test_nn = sp.sparse.hstack((tfidf_chars.transform(test.name), tfidf_words.transform(test.name)))
p_test = estimator.predict_proba(X_test_nn.tocsr())


# In[24]:


clipping = 0.0001


# In[25]:


for i, c in enumerate(labeler.classes_):
    p = p_test[:, i]
    p[p < clipping] = clipping
    p[p > (1.0 - clipping)] = (1.0 - clipping)
    p_test[:, i] = p


# In[26]:


test = test[['id']]
for i, c in enumerate(labeler.classes_):
    test[c] = p_test[:, i]
test.to_csv('submission.csv', encoding = 'utf-8', index = False)


# In[ ]:



