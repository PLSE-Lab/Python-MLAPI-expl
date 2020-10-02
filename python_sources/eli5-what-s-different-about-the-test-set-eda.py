#!/usr/bin/env python
# coding: utf-8

# We know that the test set is different from the train set for this competition, and that has been causing problems on matching the leaderboard with cross-validation. The test set was stated to be drawn from a different distribution, but what is different about it? Olivier found that [there's a language imbalance](https://www.kaggle.com/ogrellier/check-languages-distribution), but what else could be driving this difference?
# 
# First, let's show we can distinguish training from test set.

# In[ ]:


import pandas as pd
from sklearn.model_selection import train_test_split

train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')


# In[ ]:


train['is_train'] = 1
test['is_train'] = 0

merge = pd.concat([train, test])
merge.drop(['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate', 'id'], axis=1, inplace=True)

X_train, X_test, y_train, y_valid = train_test_split(merge, merge['is_train'], test_size=0.2, random_state=144)
print(X_train.shape)
print(X_test.shape)


# In[ ]:


from sklearn.feature_extraction.text import TfidfVectorizer

tfidf_vec = TfidfVectorizer(ngram_range=(1, 2),
                            analyzer='word',
                            stop_words=None,
                            max_features=200000,
                            binary=True)
train_tfidf = tfidf_vec.fit_transform(X_train['comment_text'])
test_tfidf = tfidf_vec.transform(X_test['comment_text'])
print(train_tfidf.shape)
print(test_tfidf.shape)


# In[ ]:


from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

model = LogisticRegression(solver='sag')
model.fit(train_tfidf, y_train)
pred_test_y = model.predict_proba(test_tfidf)[:, 1]
print('AUC of guessing test: {}'.format(roc_auc_score(y_valid, pred_test_y)))


# In[ ]:


X_test['pred'] = pred_test_y
X_test.head(20)


# So what happened here? Well rather than predicting whether an individual comment was "toxic" or not, we instead tried to predict whether an individual comment was from the train set or not. This is called **adversarial validation**. It looks like we succeeded with a AUC of 0.7. (...Not a high AUC by the standards of this competition, but a high AUC by the standards of models typically.)
# 
# But now what is driving these predictions? [ElL5](http://eli5.readthedocs.io/en/latest/) is a library that can help us with that - it lets you look at the model weights and TFIDF vectorizer under the hood and see what's going on. (Maybe more like ELI25 than ELI5, but still much easier to understand.) I first [saw this in action](https://www.kaggle.com/lopuhin/eli5-for-mercari) for the Mercari competition, which also heavily featured text mining.

# In[ ]:


import eli5
eli5.show_weights(model, vec=tfidf_vec)


# Ok, that's cool, but let's go a bit deeper! (Scroll down to see.)

# In[ ]:


eli5.show_weights(model, vec=tfidf_vec, top=200)


# It looks like discussions about talk pages, contributions, and numbers are what distinguishes the two sets so much. It also looks like some toxic words, like "stupid", "bullshit", "crap", "sucks", etc. are more unique to the train set, but "stupid bitch" is more unique to the test set. Maybe insults change over time?
# 
# Let's repeat the same on a per-character basis and see what we can learn there. (NB: This part takes awhile to run.)

# In[ ]:


tfidf_vec_char = TfidfVectorizer(ngram_range=(2, 6),
                                 analyzer='char',
                                 stop_words='english',
                                 max_features=200000,
                                 binary=True)
train_tfidf_char = tfidf_vec_char.fit_transform(X_train['comment_text'])
test_tfidf_char = tfidf_vec_char.transform(X_test['comment_text'])
print(train_tfidf_char.shape)
print(test_tfidf_char.shape)


# In[ ]:


char_model = LogisticRegression(solver='sag')
char_model.fit(train_tfidf_char, y_train)
pred_test_y2 = model.predict_proba(test_tfidf_char)[:, 1]
print('AUC of guessing test: {}'.format(roc_auc_score(y_valid, pred_test_y2)))


# In[ ]:


eli5.show_weights(char_model, vec=tfidf_vec_char, top=200)


# Based on the AUC of the LR and the nonsensical output of ELI5, it doesn't look like the character-based model picks up anything useful for distinguishing the train and test set. There's just idiosyncratic punctuation differences and some focus on the talk page, as would be expected.
# 
# Overall, I think this helps explain a bit of the problem, but I'm unsure about whether there's anything here that can be done to help. Maybe cleaning up numbers and wiki-specific terminology would help?

# 
