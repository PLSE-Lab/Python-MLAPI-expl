import numpy as np
import pandas as pd
import re
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import cross_val_score


RE_SPLIT = r'\W+'
reSplit = re.compile(RE_SPLIT, re.UNICODE)
RE_SPLIT2 = r'\w+'
reSplit2 = re.compile(RE_SPLIT2, re.UNICODE)

def tok2(text):
    """Tokenizer: 
    """
    t1 = [str.lower(x) for x in re.split(reSplit, text)]
    t2 = [str.lower(x) for x in map(lambda x: x.strip(), re.split(reSplit2, text))]
    if t2[0] == u'':
        t2.remove(u'')
    z = list(zip(t1, t2))
    return list(filter(None, [z[i][j] for i in range(len(z)) for j in range(2)]))


train_fname = '../input/train.csv'
test_fname = '../input/test.csv'

train_data = pd.read_csv(train_fname, encoding='utf-8', header=0)
train_data = train_data.fillna(' ')
# train_data['total'] = train_data['title']
# train_data['total'] = train_data['author']
# train_data['total'] = train_data['text']
# train_data['total'] = train_data['title'] + ' ' + train_data['text']
train_data['total'] = train_data['title'] + ' ' + train_data['author']
# train_data['total'] = train_data['title'] + ' ' + train_data['author'] + ' ' + train_data['text']

# count_vectorizer = CountVectorizer(tokenizer=tok2, binary=False, ngram_range=(1, 2))
count_vectorizer = CountVectorizer(tokenizer=tok2, binary=True, ngram_range=(1, 2))

train_X = count_vectorizer.fit_transform(train_data['total'].values)
train_y = train_data['label'].values

clf = LogisticRegression(verbose=True)

scores = cross_val_score(clf, train_X, train_y, cv=5)
print("Accuracy: %0.6f (+/- %0.6f)" % (scores.mean(), scores.std() * 2))
