# This script concatenates all news headlines of a day into one and uses the tf-idf scheme to extract a feature vector.
# An SVM with rbf kernel without optimization of hyperparameters is used as a classifier.

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import pandas
import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import roc_auc_score
from datetime import date

# read data
data = pandas.read_csv("../input/Combined_News_DJIA.csv", )

# concatenate all news into one
data["combined_news"] = data.filter(regex=("Top.*")).apply(lambda x: ''.join(str(x.values)), axis=1)

# convert to feature vector
feature_extraction = TfidfVectorizer()
X = feature_extraction.fit_transform(data["combined_news"].values)

# split into training- and test set
TRAINING_END = date(2014,12,31)
num_training = len(data[pandas.to_datetime(data["Date"]) <= TRAINING_END])
X_train = X[:num_training]
X_test = X[num_training:]
y_train = data["Label"].values[:num_training]
y_test = data["Label"].values[num_training:]

# train classifier
clf = SVC(probability=True, kernel='rbf')
clf.fit(X_train, y_train)

# predict and evaluate predictions
predictions = clf.predict_proba(X_test)
print('ROC-AUC yields ' + str(roc_auc_score(y_test, predictions[:,1])))