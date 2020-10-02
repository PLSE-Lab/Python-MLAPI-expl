#!/usr/bin/env python
# coding: utf-8

# ## QUORA INSINCERE PROJECT 

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns 
import random

import os
print(os.listdir("../input"))


# ## Data extraction and load 

# In[ ]:


df = pd.read_csv(os.path.join('../input', 'train.csv'))
df_test = pd.read_csv(os.path.join('../input', 'test.csv'))
X_test = df_test['question_text']
df.head()


# In[ ]:


df.tail()


# In[ ]:


df.isna().sum()


# In[ ]:


df['target'].unique()
df[df['question_text'] == ''].sum()


# ** Datas don't have NaN values and '', datas don't need to be cleaned **

# In[ ]:


sincere_q = (df['target'] == 0).sum()
insincere_q = (df['target'] == 1).sum()

sincere_q, insincere_q


# In[ ]:


rate_sincere_q = (sincere_q/len(df['target']))*100
rate_insincere_q = (insincere_q/len(df['target']))*100
rate_sincere_q, rate_insincere_q
print( '{}% of questions are sincere and {}% are insincere'.format(rate_sincere_q, rate_insincere_q))


# ** Proportion very unbalanced, we have to try to scale datas ** 

# In[ ]:


index_insincere_q = np.array(df[df['target'] == 1].index) # len = 80810 
index_sincere_q = np.array(df[df['target'] == 0].index)
index_sincere_q_reduc = random.sample(list(index_sincere_q), int(1.8*len(index_insincere_q)))


# In[ ]:


X = pd.concat([df['question_text'][index_insincere_q], df['question_text'][index_sincere_q_reduc]])
y = pd.concat([df['target'][index_insincere_q], df['target'][index_sincere_q_reduc]])
#X = df['question_text']
#y = df['target']
X.shape, y.shape


# ** The proportion of insincere questions and sincere questions is the same ** [](http://)

# ** Split datas in train and test set **

# In[ ]:


from sklearn.model_selection import train_test_split
X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=.2, random_state=42, stratify=y)
X_train.shape, y_train.shape, X_valid.shape, y_valid.shape


# ** Preprocessing with nltk ** 

# In[ ]:


import nltk
from nltk.tokenize import word_tokenize
from nltk.tokenize import sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.stem import PorterStemmer


# ** Tokenizing **

# ** Creation of a tokenize's function permitting to automatically tokenize our train and test set ** 

# In[ ]:


def tokenize(data):
    tokenized_docs = [word_tokenize(doc.lower()) for doc in data]
    alpha_tokens = [[t for t in doc if t.isalpha() == True] for doc in tokenized_docs]
    stemmer = PorterStemmer ()
    stemmed_tokens = [[stemmer.stem(alpha) for alpha in doc] for doc in alpha_tokens]
    X_stem_as_string = [" ".join(x_t) for x_t in stemmed_tokens]
    return X_stem_as_string


# In[ ]:


X_train_pr = tokenize(X_train)
X_valid_pr = tokenize(X_valid)
X_test_pr = tokenize(X_test)


# ** Preprocessing pipeline **

# In[ ]:


from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.pipeline import Pipeline


# In[ ]:


vct = CountVectorizer(stop_words='english', ngram_range=(2, 3), lowercase=False)
svd = TruncatedSVD(n_components=100, random_state=42)
tfvec = TfidfVectorizer(stop_words='english', lowercase=False)


# In[ ]:


preprocessing_pipe = Pipeline([
    ('vectorizer', tfvec),
    ('svd', svd),
])


# In[ ]:


lsa_train = preprocessing_pipe.fit_transform(X_train_pr)
lsa_train.shape


# **Visualizing the most informative features ** 

# In[ ]:


components = pd.DataFrame(data=svd.components_, columns=preprocessing_pipe.named_steps['vectorizer'].get_feature_names())
components


# In[ ]:


fig, axes = plt.subplots(10, 2, figsize=(18, 30))
for i, ax in enumerate(axes.flat):
    components.iloc[i].sort_values(ascending=False)[:10].sort_values().plot.barh(ax=ax)


# ## MACHINE LEARNING MODELS

# ** Creation of a pipeline with prepocessing pipeline ** 

# In[ ]:


import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.naive_bayes import MultinomialNB


# In[ ]:


rf = RandomForestClassifier (class_weight='balanced_subsample')
xgbc = xgb.XGBClassifier() 
mb = MultinomialNB()
pipe = Pipeline([
    ('vectorizer', tfvec),
    ('mb', mb)
])


# In[ ]:


X_train_pr = tokenize(X_train)


# In[ ]:


pipe.fit(X_train_pr, y_train)
y_pred = pipe.predict(X_valid_pr)


# ** Confusion Matrix ** 

# In[ ]:


from sklearn.metrics import confusion_matrix, classification_report


# In[ ]:


cm = confusion_matrix(y_valid, y_pred)
cm


# In[ ]:


labels = ['sincere', 'unsincere']
df_cm = pd.DataFrame(cm, columns=labels, index=labels)
df_cm


# ** VALIDATION **

# In[ ]:


from sklearn.model_selection import cross_val_score


# In[ ]:


score = cross_val_score(pipe, X_valid_pr, y=y_valid, cv=5, scoring='f1_macro')
score


# In[ ]:


print(classification_report(y_valid, y_pred))


# In[ ]:


y_test_true = pipe.predict(X_test_pr)


# In[ ]:


#df_sample_submission = pd.DataFrame({'qid' : df_test['qid'], 'y_pred' : y_test_true})
#index_insin = np.array(df_sample_submission[df_sample_submission['y_pred'] == 1].index) 
#df_sample_submission['qid'][index_insin]
sub = pd.read_csv('../input/sample_submission.csv')
sub.prediction = y_test_true
sub.to_csv("submission.csv", index=False)

