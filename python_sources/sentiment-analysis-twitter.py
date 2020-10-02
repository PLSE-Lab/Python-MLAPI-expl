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
airwords = ['airline','air','plane']

# Importing NLTK
import nltk

# preprocessor
from nltk.corpus import stopwords
stop = stopwords.words('english')

# sanitize words
def sanitize(word):
    word = word.replace('#','').lower()
    return word
    
# Tokenizer Function - Slices Paragraphs and Sentences to Words
def splitter(text):
    wordlist = []
    for sentence in nltk.sent_tokenize(text):
         [wordlist.append(sanitize(word)) for word in text.split() if (word not in stop and word not in airwords)]
    
    del wordlist[0]
    return wordlist

# Use only two columns
df = data.loc[:14000,['text','airline_sentiment']]

# Randomize the Data
np.random.seed(0)
df = df.reindex(np.random.permutation(df.index))

df['words'] = df['text'].apply(splitter)

words = []
[[words.append(w) for w in wlist] for wlist in df['words']]

allfeatures = nltk.FreqDist(w for w in words)
allfeatures = list(allfeatures)[:2000]

def word_features(docwords):
    features = {}
    for w in allfeatures:
        features['contains({})'.format(w)] = (w in docwords)
    return features
    
featuresets = []
for i in range(0,len(df['words'])):
    featuresets.append([word_features(df['words'][i]),df['airline_sentiment'][i]])
    
train_set = featuresets[10000:]
test_set = featuresets[:4000]
classifier = nltk.NaiveBayesClassifier.train(train_set)

print(nltk.classify.accuracy(classifier,test_set))

classifier.show_most_informative_features(10)
#print(test_set[0])
    








