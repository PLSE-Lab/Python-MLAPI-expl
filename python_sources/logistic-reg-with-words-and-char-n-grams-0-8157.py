import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from scipy.sparse import hstack
from sklearn.metrics import roc_curve, roc_auc_score

train = pd.read_csv('../input/reddit_train.csv', encoding = 'latin-1').fillna(' ')
test = pd.read_csv('../input/reddit_test.csv', encoding = 'latin-1').fillna(' ')

train_text = train['BODY']
test_text = test['BODY']
all_text = pd.concat([train_text, test_text])

word_vectorizer = TfidfVectorizer(
    sublinear_tf=True,
    strip_accents='unicode',
    analyzer='word',
    token_pattern=r'\w{1,}',
    stop_words='english',
    ngram_range=(1, 1),
    max_features=10000)
word_vectorizer.fit(train_text)
train_word_features = word_vectorizer.transform(train_text)
test_word_features = word_vectorizer.transform(test_text)

char_vectorizer = TfidfVectorizer(
    sublinear_tf=True,
    strip_accents='unicode',
    analyzer='char',
    stop_words='english',
    ngram_range=(2, 6),
    max_features=50000)
char_vectorizer.fit(train_text)
train_char_features = char_vectorizer.transform(train_text)
test_char_features = char_vectorizer.transform(test_text)

train_features = hstack([train_char_features, train_word_features])
test_features = hstack([test_char_features, test_word_features])

train_target = train['REMOVED']
classifier = LogisticRegression(C=0.1, solver='sag')

cv_score = np.mean(cross_val_score(classifier, train_features, train_target, cv=3, scoring='roc_auc'))
cv_score

classifier.fit(train_features, train_target)
preds = classifier.predict_proba(test_features)[:, 1]

submission = pd.concat([pd.DataFrame(preds), test['REMOVED']], axis=1)
submission.to_csv('submission.csv', index=False)

roc_auc_score(test['REMOVED'], preds)