# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.
data = pd.read_csv('../input/Tweets.csv')

airwords = ['airline','air','plane','flight']
airlines = data['airline']

from nltk.corpus import stopwords
stop = stopwords.words('english')

import re

def preprocessor(text):
    words = text.split()
    del words[0]
    words = [re.sub('[^0-9a-zA-Z]+', '', w.replace('#','').lower()) for w in words if not w.startswith('http') and not w.startswith('@')]
    finalwords = [w for w in words if (not w.isdigit() and w not in airwords and w not in airlines)]
    return ' '.join(finalwords)

#print(data.head(3))
data = data.loc[:14000,['text','airline_sentiment']]
data['text'] = data['text'].apply(preprocessor)

print(preprocessor("This is a tweet flight 979 and i'm happy #flight #travel @maniche04"))

test_data = data[4200:]
train_data = data[:9800]

print(train_data['text'][91:120])
#print(test_data['text'])

# extracting the feature list
from sklearn.feature_extraction.text import CountVectorizer
#count = CountVectorizer(ngram_range=(1,2))
#bag = count.fit_transform(s for s in data.loc[:,'text'])
#print(count.vocabulary_)

# tranforming the feature list
from sklearn.feature_extraction.text import TfidfTransformer
#tf_transformer = tf_transformer.fit_transform(bag)

# training
from sklearn.naive_bayes import MultinomialNB
#clf = MultinomialNB().fit(bag,data['airline_sentiment'])

# implement pipeline
from sklearn.pipeline import Pipeline
text_clf = Pipeline([('vect',CountVectorizer(ngram_range=(4,5))),('tfidf',TfidfTransformer()),('clf',MultinomialNB())])
text_clf = text_clf.fit(train_data['text'],train_data['airline_sentiment'])

print('Test Accuracy: %.3f' % text_clf.score(test_data['text'], test_data['airline_sentiment']))




