# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Load dataset
data = pd.read_csv("../input/creditcard.csv")

# Drop 'Amount'
data = data.drop(['Time', 'Amount'],axis=1)
data.head()

# Create X and y

y = data['Class']
X = data.drop(['Class'], axis=1)

# Import 'train_test_split'
from sklearn.model_selection import train_test_split

# Shuffle and split the data into training and testing subsets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Try RandomForestClassifier
'''Random forests or random decision forests are an ensemble learning method for classification, regression and other tasks, that operate by constructing a multitude of decision trees at training time and outputting the class that is the mode of the classes (classification) or mean prediction (regression) of the individual trees (source: Wikipedia)'''

from sklearn.ensemble import RandomForestClassifier

clf = RandomForestClassifier()
clf = clf.fit(X_train, y_train)

y_pred1 = clf.predict(X_train)

# calculate accuracy score
clf.score(X_test, y_test)

# calculate precision and recall scores

from sklearn.metrics import precision_score
from sklearn.metrics import recall_score

precision = precision_score(y_train, y_pred1, average='binary')
recall = recall_score(y_train, y_pred1, average='binary')

print(precision)
print(recall)

# generate confusion matrix

from sklearn.metrics import confusion_matrix
import seaborn as sns

cm3 = confusion_matrix(y_train,y_pred1)

df_cm3 = pd.DataFrame(cm3, index = ['True (positive)', 'True (negative)'])
df_cm3.columns = ['Predicted (positive)', 'Predicted (negative)']

sns.heatmap(df_cm3, annot=True, fmt="d")


# Try Gaussian Naive Bayes Classifier
'''Naive Bayes is a simple technique for constructing classifiers: models that assign class labels to problem instances, represented as vectors of feature values, where the class labels are drawn from some finite set. It is not a single algorithm for training such classifiers, but a family of algorithms based on a common principle: all naive Bayes classifiers assume that the value of a particular feature is independent of the value of any other feature, given the class variable.(source: Wikipedia)'''
from sklearn.naive_bayes import GaussianNB 

clf = GaussianNB()
clf.fit(X_train, y_train)
y_pred2 = clf.predict(X_train)

# calculate accuracy score
clf.score(X_test, y_test)

# calculate precision and recall score

precision = precision_score(y_train, y_pred2, average='binary')
recall = recall_score(y_train, y_pred2, average='binary')

print(precision)
print(recall)

# generate confusion matrix

cm = confusion_matrix(y_train,y_pred2)

df_cm = pd.DataFrame(cm, index = ['True (positive)', 'True (negative)'])
df_cm.columns = ['Predicted (positive)', 'Predicted (negative)']

sns.heatmap(df_cm, annot=True, fmt="d")

# Try Logistic Regression Classifier
'''Logistic regression is a regression model where the dependent variable is categorical (source: Wikipedia)'''

from sklearn.linear_model import LogisticRegression

clf = LogisticRegression()
clf.fit(X,y)
y_pred3 = clf.predict(X_train)

# calculate accuracy score
clf.score(X_test, y_test)

# calculate precision and recall score

precision = precision_score(y_train, y_pred3, average='binary')
recall = recall_score(y_train, y_pred3, average='binary')

print(precision)
print(recall)

# generate confusion matrix

cm2 = confusion_matrix(y_train,y_pred3)

df_cm2 = pd.DataFrame(cm2, index = ['True(positive)', 'True(negative)'])
df_cm2.columns = ['Predicted (positive)', 'Predicted (negative)']

sns.heatmap(df_cm2, annot=True, fmt="d")



from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.