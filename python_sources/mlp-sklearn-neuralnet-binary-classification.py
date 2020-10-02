'''
Sept01-2018
Mahesh Babu Mariappan (https://www.linkedin.com/in/mahesh-babu-mariappan)
Source code for MLP neuralnet sklearn
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory
# Any results you write to the current directory are saved as output.

Results:
test accuracy: 0.7841038487815988
test f1score: 0.5553470919324578
train accuracy: 0.7909996095275283
train f1score: 0.5749454040103237

'''
import os
import pandas
from sklearn import preprocessing
import time
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score, precision_recall_fscore_support
import warnings

warnings.filterwarnings('ignore')

dataFrame = pandas.read_csv('../input/Surgical-deepnet.csv')

dataframe = dataFrame.dropna(axis=0).copy()

startTime = time.time()

predictors = dataFrame.iloc[:, dataFrame.columns != 'complication']
target = dataFrame.iloc[:, dataFrame.columns == 'complication']

X = pandas.DataFrame()
for column in predictors.columns:
    if predictors[column].dtype == 'object':
        X[column] = predictors[column].copy()

objects = []
for each in range(len(X.columns)):
    objects.append(preprocessing.LabelEncoder())

for column, obj in zip(X.columns, objects):
    X[column] = obj.fit_transform(X[column])

for column in X.columns:
    predictors[column] = X[column]

X_train, X_test, y_train, y_test = train_test_split(predictors, target.values.ravel(), test_size=0.3)

neuralNet = MLPClassifier()

model = neuralNet.fit(X_train, y_train)
prediction = model.predict(X_test)

precision, recall, fbeta, support = precision_recall_fscore_support(y_test, prediction)

endTime = time.time()

# Accuracy 
print("test accuracy:", accuracy_score(y_test, prediction))

# F1 score
print("test f1score:", f1_score(y_test,prediction))

############################################################
#let's compare the train prediction scores
train_prediction = model.predict(X_train)

precision, recall, fbeta, support = precision_recall_fscore_support(y_train, train_prediction)
# Accuracy 
print("train accuracy:", accuracy_score(y_train, train_prediction))

# F1 score
print("train f1score:", f1_score(y_train, train_prediction))
