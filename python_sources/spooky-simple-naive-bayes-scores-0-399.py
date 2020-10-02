# Importing the libraries
import numpy as np
import pandas as pd

# Input data files
train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')


# Exploratory data analysis

# Percent missing values for each column
# there are no missing values
percent_missing = 100 * train.isnull().sum()/len(train)

# look at class imbalance
# classes are fairly balanced
eap = (train.author == 'EAP').sum()
hpl = (train.author == 'HPL').sum()
mws = (train.author == 'MWS').sum()


# Cleaning the texts
import re
import nltk
#nltk.download('stopwords')
from nltk.corpus import stopwords

train_corpus = []
for i in range(len(train)):
    review = re.sub('[^a-zA-Z0-9]', ' ', train['text'][i])
    review = review.lower()
    review = review.split()
    review = [word for word in review if not word in set(stopwords.words('english'))]
    review = ' '.join(review)
    train_corpus.append(review)

test_corpus = []
for i in range(len(test)):
    review = re.sub('[^a-zA-Z0-9]', ' ', test['text'][i])
    review = review.lower()
    review = review.split()
    review = [word for word in review if not word in set(stopwords.words('english'))]
    review = ' '.join(review)
    test_corpus.append(review)

del i
del review


# format data for input
X_Train = np.array(train_corpus)
X_Test = np.array(test_corpus)
y = train.iloc[:, 2].values

# remove unneeded old variables
del test, test_corpus, train, train_corpus


## Multinomial Naive Bayes Classifier ##
# Build pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
classifier = Pipeline([('vect', CountVectorizer()),
                      ('tfidf', TfidfTransformer()),
                      ('clf', MultinomialNB()),
])
classifier.fit(X_Train, y)

# parameter tuning with grid search
from sklearn.model_selection import GridSearchCV
parameters = {'vect__ngram_range': [(1, 1), (1, 2), (1, 3)],
              'tfidf__use_idf': (True, False),
              'clf__alpha': (0, 0.01, 0.05, 0.1, 0.3, 0.5),
}
gs_clf = GridSearchCV(classifier, parameters)
gs_clf.fit(X_Train, y)


# Predicting the Test set results
y_pred_proba = gs_clf.predict_proba(X_Test)

# y_pred_proba contain the results
