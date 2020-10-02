# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import Pipeline
from sklearn.ensemble import GradientBoostingClassifier

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

# import os
# for dirname, _, filenames in os.walk('/kaggle/input'):
#     for filename in filenames:
#         print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session

test = pd.read_csv('/kaggle/input/nlp-getting-started/test.csv')
train = pd.read_csv('/kaggle/input/nlp-getting-started/train.csv')

X_train, X_test, y_train, y_test = train_test_split(train['text'], train['target'], test_size=0.25, random_state=42)

X_train.head()
X_train_token = X_train.apply(nltk.word_tokenize)

vectorizer = TfidfVectorizer(stop_words='english')
from sklearn.naive_bayes import MultinomialNB

# Train on sample
# Xtfidf = vectorizer.fit_transform(X_train)
# print(vectorizer.get_feature_names())
# nbayes = MultinomialNB().fit(Xtfidf, y_train)

text_nbayes = Pipeline([
    ('tfidf', TfidfVectorizer()),
    ('nbayes', MultinomialNB()),
])

text_svm = Pipeline([
    ('tfidf', TfidfVectorizer()),
    ('svm', SGDClassifier()),
])

text_dectree = Pipeline([
    ('tfidf', TfidfVectorizer()),
    ('dtree', DecisionTreeClassifier()),
])

text_gb = Pipeline([
    ('tfidf', TfidfVectorizer()),
    ('gboost', GradientBoostingClassifier(n_estimators=100, learning_rate=1.0, max_depth=3, random_state=0)),
])

text_nbayes.fit(X_train, y_train)
text_svm.fit(X_train, y_train)
text_dectree.fit(X_train, y_train)
text_gb.fit(X_train, y_train)

text_svm.score(X_test, y_test)
text_gb.score(X_test, y_test)

predicted = text_nbayes.predict(X_test)
np.mean(predicted == y_test)

predicted = text_svm.predict(X_test)
np.mean(predicted == y_test)

predicted = text_dectree.predict(X_test)
np.mean(predicted == y_test)

predicted = text_gb.predict(X_test)
np.mean(predicted == y_test)

# Metrics
from sklearn.metrics import classification_report, confusion_matrix
print(confusion_matrix(y_test, predicted))
print(classification_report(y_test, predicted))

## Predict on the entry
#Ytfidf = vectorizer.transform(test['text'])
predicted = text_nbayes.predict(test['text'])

for doc, category in zip(test['text'], predicted):
    print('%r => %s' % (doc, category))
    
test['target'] = predicted

sub = test[['id', 'target']]

sub.to_csv('submission.csv',index=False)

