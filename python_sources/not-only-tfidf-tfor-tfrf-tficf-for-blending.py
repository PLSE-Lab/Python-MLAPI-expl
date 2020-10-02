#!/usr/bin/env python
# coding: utf-8

# **The main purpose of this kernel to show that there are a lot of other ways to vectorize texts, not only tfidf.**
# 
# It is common to use TFIDF in text-type competions on Kaggle, but as far as we are solving classification task we have much more types of text vectorization: 
# 1. TFICF: tf & inverse category frequency
# 2. TFOR: tf & odds ratio
# 3. TFRF: tf & relevance frequency
# 
# More detailed examples and implementation you could find at [Textvec](https://github.com/zveryansky/textvec) (commits and stars are welcomed!)
# 
# TLDR: you can use Textvec like sklearn TfidfVectorizer and add to blending.

# **TFOR (Odds ratio) explanation:**
# 
# What is odds ratio? Wiki: [The odds ratio (OR) is a statistic defined as the ratio of the odds of A in the presence of B and the odds of A without the presence of B. This statistic attempts to quantify the strength of the association between A and B.](https://en.wikipedia.org/wiki/Odds_ratio)
# 
# In general this mean that for every word we could count the odds of label 1 if this word is in text.
# 
# Here is an example of using OR for dimension reduction with CI:

# In[ ]:


import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from textvec import vectorizers


train = pd.read_csv('../input/train.csv').fillna(' ')#.sample(10000, random_state=13)
train_target = train['target'].values

train_text = train['question_text']

X_train, X_test, y_train, y_test = train_test_split(train_text, train_target, test_size=0.1, random_state=13)

count_vec = CountVectorizer(strip_accents='unicode',
    token_pattern=r'\w{1,}',
    ngram_range=(1, 1)).fit(X_train)

tfor_vec = vectorizers.TforVectorizer(sublinear_tf=True)
tfor_vec.fit(count_vec.transform(X_train), y_train)
train_or, ci_95 = tfor_vec.transform(count_vec.transform(X_train), confidence=True)
test_or = tfor_vec.transform(count_vec.transform(X_test))

classifier = LogisticRegression(C=10, solver='sag', random_state=13)
classifier.fit(train_or, y_train)
val_preds = classifier.predict_proba(test_or)[:,1]
print('ROC_AUC -> ', roc_auc_score(y_test, val_preds))
print('shape -> ', train_or.shape)


# In[ ]:


classifier = LogisticRegression(C=10, solver='sag', random_state=13)
classifier.fit(train_or[:,ci_95], y_train)
val_preds = classifier.predict_proba(test_or[:,ci_95])[:,1]
print('ROC_AUC -> ', roc_auc_score(y_test, val_preds))
print('shape -> ', train_or[:,ci_95].shape)


# As you could see we achieved nearly the same score but with 8 times smaller dimension.
# 
# Now lets test the correlation with TFIDF:

# In[ ]:


import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from textvec import vectorizers


train = pd.read_csv('../input/train.csv').fillna(' ')#.sample(100000, random_state=13)
test = pd.read_csv('../input/test.csv').fillna(' ')#.sample(10000, random_state=13)
test_qid = test['qid']
train_target = train['target'].values

train_text = train['question_text']
test_text = test['question_text']

tfidf_vec = TfidfVectorizer(
    sublinear_tf=True,
    strip_accents='unicode',
    token_pattern=r'\w{1,}',
    ngram_range=(1, 1))
tfidf_vec.fit(pd.concat([train_text, test_text]))
train_idf = tfidf_vec.transform(train_text)


count_vec = CountVectorizer(strip_accents='unicode',
    token_pattern=r'\w{1,}',
    ngram_range=(1, 1)).fit(train_text)

tfrf_vec = vectorizers.TfrfVectorizer(sublinear_tf=True)
tfrf_vec.fit(count_vec.transform(train_text), train_target)
train_rf = tfrf_vec.transform(count_vec.transform(train_text))

tfor_vec = vectorizers.TforVectorizer(sublinear_tf=True)
tfor_vec.fit(count_vec.transform(train_text), train_target)
train_or = tfor_vec.transform(count_vec.transform(train_text))

tficf_vec = vectorizers.TfIcfVectorizer(sublinear_tf=True)
tficf_vec.fit(count_vec.transform(train_text), train_target)
train_icf = tficf_vec.transform(count_vec.transform(train_text))

tfbinicf_vec = vectorizers.TfBinIcfVectorizer(sublinear_tf=True)
tfbinicf_vec.fit(count_vec.transform(train_text), train_target)
train_binicf = tfbinicf_vec.transform(count_vec.transform(train_text))

results = {}

def validate_results(train_data_vecs, name):
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=13)
    for i, (train_index, val_index) in enumerate(skf.split(train_text, train_target)):
        x_train, x_val = train_data_vecs[list(train_index)], train_data_vecs[list(val_index)]
        y_train, y_val = train_target[train_index], train_target[val_index]
        classifier = LogisticRegression(C=10, solver='sag', random_state=13)
        classifier.fit(x_train, y_train)
        val_preds = classifier.predict_proba(x_val)[:,1]
        current_results = results.get(name,{'preds': [], 'target': []})
        current_results['preds'].extend(val_preds)
        current_results['target'].extend(y_val)
        results[name] = current_results

validate_results(train_rf, 'rf')
validate_results(train_idf, 'idf')
validate_results(train_or, 'or')
validate_results(train_binicf, 'binicf')
validate_results(train_icf, 'icf')


# In[ ]:


import seaborn as sns
import matplotlib.pylab as plt
get_ipython().run_line_magic('matplotlib', 'inline')
res = []
for k, v in results.items():
    res.append((k, roc_auc_score(v['target'],np.array(v['preds'])) ,v['preds']))
res = sorted(res, key= lambda x:-x[1])
corrs = np.corrcoef(list(zip(*res))[2])
accs = list(zip(*res))[1]
labels = [f'{x}:{accs[i]:.4f}' for i, x in enumerate(list(zip(*res))[0])]
fig, ax = plt.subplots(figsize=(10,10)) 
ax = sns.heatmap(corrs, 
                 linewidth=0.5, 
                 annot=corrs, 
                 square=True, 
                 ax=ax, 
                 xticklabels=labels,
                 yticklabels=labels)

plt.show()


# As you see -- it could be blended, and I hope it will imporove your LB score.

# In[ ]:




