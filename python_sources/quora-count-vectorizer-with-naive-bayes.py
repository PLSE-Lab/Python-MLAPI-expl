from sklearn import naive_bayes
from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd
import numpy as numpy

train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')

sentences1 = numpy.array(train['question_text'])
sentences2 = numpy.array(test['question_text'])
sentences = numpy.concatenate((sentences1, sentences2), axis=0)

vectorizer = CountVectorizer()
vectorizer.fit(sentences)

X_train = vectorizer.transform(train['question_text'])
X_test = vectorizer.transform(test['question_text'])

classifier = naive_bayes.MultinomialNB()
classifier.fit(X_train, train['target'])

predictions = classifier.predict(X_test)

my_submission = pd.DataFrame({'qid': test['qid'], 'prediction': predictions})
my_submission.to_csv('submission.csv', index=False)
