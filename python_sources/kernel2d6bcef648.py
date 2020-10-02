# Natural Language Processing

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('train.csv')
Tweet=dataset.iloc[:,3:]

# Cleaning the texts
import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
corpus = []
for i in range(0, 7613):
    tweet = re.sub('[^a-zA-Z]', ' ', Tweet['text'][i])
    tweet = tweet.lower()
    tweet= tweet.split()
    ps = PorterStemmer()
    tweet = [ps.stem(word) for word in tweet if not word in set(stopwords.words('english'))]
    tweet = ' '.join(tweet)
    corpus.append(tweet)

# Creating the Bag of Words model
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features = 1500)
X = cv.fit_transform(corpus).toarray()
y = Tweet.iloc[:, 1].values

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)

# Fitting Naive Bayes to the Training set
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

#checking test results
dataset2=pd.read_csv('test.csv')
test_x=dataset2.iloc[:,[0,3]]
id=dataset2.iloc[:,0].values

corpus2 = []
for i in range(0, 3263):
    tweet2 = re.sub('[^a-zA-Z]', ' ', test_x['text'][i])
    tweet2 = tweet2.lower()
    tweet2= tweet2.split()
    ps2 = PorterStemmer()
    tweet2 = [ps2.stem(word) for word in tweet2 if not word in set(stopwords.words('english'))]
    tweet2 = ' '.join(tweet2)
    corpus2.append(tweet2)
    

test_X = cv.transform(corpus2).toarray()

y_prediction=classifier.predict(test_X)

submission_result=pd.DataFrame({'id':id,'target':y_prediction})
submission_result.to_csv("SubmissionResult.csv", index=False)