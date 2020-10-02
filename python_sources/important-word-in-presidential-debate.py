#!/usr/bin/env python
# coding: utf-8

# Debate comparison between Clinton and Trump. It is trying to simply detect the characteristics of their words and find the keyword of debates.

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import nltk
import matplotlib.pyplot as plt

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.


# # Check dataset
# 
# It includes the text, its speaker and date when it was spoken.

# In[ ]:


df = pd.read_csv('../input/debate.csv', encoding = 'iso-8859-1')

df.head()


# In[ ]:


clinton = df[df['Speaker'] == 'Clinton']

clinton_text = clinton['Text']
clinton.head()


# In[ ]:


trump = df[df['Speaker'] == 'Trump']

trump_text = trump['Text']
trump.head()


# # Stemming
# 
# Checking their term usage distribution. We could remove stopwords which is defined in [NLTK](http://www.nltk.org/) library in English.

# In[ ]:


print("Clinton Texts: {}".format(len(clinton_text)))
print("Trump Texts: {}".format(len(trump_text)))


# In[ ]:


from nltk.tokenize import word_tokenize
from nltk.probability import FreqDist
from nltk.corpus import stopwords

stopwords = stopwords.words('english')
symbols = ["'", '"', '`', '.', ',', '-', '!', '?', ':', ';', '(', ')', '...']

clinton_words = []
for s in clinton_text:
    clinton_words += [w for w in word_tokenize(s) if w not in stopwords + symbols] 
    
fdist = FreqDist(clinton_words)

fdist.plot(30)
    


# In[ ]:


trump_words = []
for s in trump_text:
    trump_words += [w for w in word_tokenize(s) if w not in stopwords + symbols] 
    
fdist = FreqDist(trump_words)

fdist.plot(30)


# In comparison with Trump, Clinton calls to Trump using his name "Donald". Both candidate uses "people" frequently.
# 
# Next let me check the time they are speaking.
# 
# # The timerange of speaking

# In[ ]:


counts = df.groupby(['Speaker', 'Date'], as_index=False).count()

counts = counts[(counts['Speaker'] == 'Trump') | (counts['Speaker'] == 'Clinton')]
g = sns.factorplot(x="Date", y="Line", hue="Speaker", data=counts, size=6, kind="bar", palette="muted")
g.despine(left=True)
g.set_ylabels("The number of speak")


# In[ ]:


def text_len(sentenses):
    len_sum = 0
    for s in sentenses:
        len_sum += len([w for w in word_tokenize(s) if w not in stopwords + symbols])
    return len_sum

speaker_texts = df[['Speaker', 'Date', 'Text']]
speaker_texts = speaker_texts.groupby(['Speaker', 'Date'], as_index=False).aggregate(text_len)

speaker_texts = speaker_texts[(speaker_texts['Speaker'] == 'Trump') | (speaker_texts['Speaker'] == 'Clinton')]
speaker_texts.head()
g = sns.factorplot(x="Date", y="Text", hue="Speaker", data=speaker_texts, size=6, kind="bar", palette="muted")
g.despine(left=True)
g.set_ylabels("The total length of spoke sentenses")


# Overall Trump spoke more than Clinton. And sentences of Trump was growing one by one, though I'm not sure the total time of each debate took.
# 
# 
# Finally let's finding important words with TF-IDF
# 
# # Calculate words importances

# In[ ]:


from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer

clinton_count = CountVectorizer()

clinton_words = clinton_count.fit_transform(clinton['Text'].values)

tfidf = TfidfTransformer(use_idf=True, norm='l2', smooth_idf=True)
clinton_tfidf = tfidf.fit_transform(clinton_words).toarray()
print(clinton_tfidf.shape)
print("Average TF-IDF of Clinton: {}".format(np.average(np.max(clinton_tfidf, axis=1))))


# In[ ]:


from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer

trump_count = CountVectorizer()

trump_words = trump_count.fit_transform(trump['Text'].values)

tfidf = TfidfTransformer(use_idf=True, norm='l2', smooth_idf=True)
trump_tfidf = tfidf.fit_transform(trump_words).toarray()
print(trump_tfidf.shape)
print("Average TF-IDF of Trump: {}".format(np.average(np.max(trump_tfidf, axis=1))))
    


# If we can regard words which has high TF-IDF as keyword in their talks, Trump said more sentences which includes keywords. 
# 
# At the last I tried to detect **"APPLAUSED"** sentences by audience.

# In[ ]:


df.iloc[df[df['Text'] == '(APPLAUSE)']['Line'] - 2].head()


# In[ ]:


applaused_train = df
applaused_train['Applaused'] = 0
applaused_train.loc[df[df['Text'] == '(APPLAUSE)']['Line'] - 2, 'Applaused'] = 1

clinton_applaused = applaused_train[df['Speaker'] == 'Clinton']['Applaused'].values
trump_applaused = applaused_train[df['Speaker'] == 'Trump']['Applaused'].values


# We'll create a classifier to decide which remarks are "Applaused"

# In[ ]:


X = np.r_[clinton['Text'].values, trump['Text'].values]
y = np.r_[clinton_applaused, trump_applaused]

from xgboost.sklearn import XGBClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import KernelPCA
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.cross_validation import StratifiedKFold
from sklearn.metrics import matthews_corrcoef, roc_auc_score

c = CountVectorizer()
X = c.fit_transform(X).toarray()
tfidf = TfidfTransformer(use_idf=True, norm='l2', smooth_idf=True)
X = tfidf.fit_transform(X).toarray()

pipe = Pipeline([('scaler', StandardScaler()), 
                 ('pca', KernelPCA(n_components=10)), 
                 ('clf', XGBClassifier())])

kfold = StratifiedKFold(y=y, n_folds=10)
scores = []
preds = np.ones(y.shape[0])
for i, (train, test) in enumerate(kfold):
    preds[test] = pipe.fit(X[train], y[train]).predict_proba(X[test])[:,1]
    print("fold {}, ROC AUC: {:.3f}".format(i, roc_auc_score(y[test], preds[test])))

thresholds = np.linspace(0.01, 0.99, 50)
mcc = np.array([matthews_corrcoef(y, preds>thr) for thr in thresholds])
best_threshold = thresholds[mcc.argmax()]

plt.plot(mcc)
plt.xlabel('Threshold')
plt.ylabel('Matthews Correlation Coefficient')
print("Best threshold: {}".format(best_threshold))


# In[ ]:


from sklearn.ensemble import RandomForestClassifier

pipe = Pipeline([('scaler', StandardScaler()), 
                 ('clf', RandomForestClassifier(random_state=13))])

kfold = StratifiedKFold(y=y, n_folds=10)
scores = []
preds = np.ones(y.shape[0])
for i, (train, test) in enumerate(kfold):
    preds[test] = pipe.fit(X[train], y[train]).predict_proba(X[test])[:,1]
    print("fold {}, ROC AUC: {:.3f}".format(i, roc_auc_score(y[test], preds[test])))
    
feature_importances = pipe.named_steps['clf'].feature_importances_
importants = np.argsort(feature_importances)
for k, v in c.vocabulary_.items():
    if v == importants[-1]:
        print("The most important word can be \"{}\"".format(k))


# We can calculate the most important word in this Random Forest model. It's "yes", interesting. But the word is changed almost everytime I run with changing random seed.  Unfortunately that's not reliable. 
