# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import pandas as pd
from nltk.corpus import stopwords
import re
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import numpy as np
import itertools

stopwords_set = set(stopwords.words("english"))
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    plt.show()

def process_tweets(text):
    # array of words longer or equal len 3
    words_filtered = [e.lower() for e in text.split() if len(e) >= 3]

    # remove if link, hash, tag, or RT
    words_cleaned = [word for word in words_filtered
        if 'http' not in word
        and not word.startswith('@')
        and not word.startswith('#')
        and word != 'rt']

    # remove stopwords
    words_without_stopwords = [word for word in words_cleaned if not word in stopwords_set]

    text = " ".join(words_without_stopwords)

    tweet_text = re.sub(r'[^\w]', ' ', text)
    tweet_text = tweet_text.lower().strip()

    return tweet_text

data = pd.read_csv('../input/Sentiment.csv')
data = data[['text','sentiment']]
data = data[data.sentiment != "Neutral"] # removing all neutrals from data

#data['text'] = data['text'].apply(process_tweets)

train, test = train_test_split(data, test_size=0.20)
# tweets = get_tweet_set(data)
# print(tweets)

vectorizer = CountVectorizer()
x_train = vectorizer.fit_transform(train['text'])
y_train = train['sentiment']
# print(vectorizer.get_feature_names())

x_test = vectorizer.transform(test['text'])
y_test = test['sentiment']

# classifier = MultinomialNB()
classifier = GaussianNB()
classifier.fit(x_train.toarray(), y_train)

pred = classifier.predict(x_test.toarray())

accuracy = accuracy_score(y_test, pred)
print(accuracy)

conf_matrix = confusion_matrix(y_test, pred, labels=["Negative", "Positive"])
print(conf_matrix)

classes = ["Negative", "Positive"]
plot_confusion_matrix(conf_matrix, classes,normalize=True)
