# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from pandas import read_csv, concat

from sklearn.model_selection import KFold, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC

from sklearn.linear_model import LogisticRegression

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# We use https://stattrek.com/statistics/random-number-generator.aspx#error to calculate random numbers

# kaggle competitions submit -c titanic -f random_submission.csv -m "Message"

# Any results you write to the current directory are saved as output.

# Load data
data_path = os.path.join(os.getcwd(), "../input//titanic/train.csv")
dataset = read_csv(data_path, skipinitialspace=True)

dataset.head(5)

# We need to drop some insignificant features and map the others.
# Ticket number and fare should not contribute much to the performance of our models.
# Name feature has titles (e.g., Mr., Miss, Doctor) included.
# Gender is definitely important.
# Port of embarkation may contribute some value.
# Using port of embarkation may sound counter-intuitive; however, there may 
# be a higher survival rate for passengers who boarded in the same port.

dataset['Title'] = dataset.Name.str.extract(' ([A-Za-z]+)\.', expand=False)
dataset = dataset.drop(['PassengerId', 'Ticket', 'Cabin', 'Name'], axis=1)

pd.crosstab(dataset['Title'], dataset['Sex'])

# We will replace many titles with a more common name, English equivalent,
# or reclassification
dataset['Title'] = dataset['Title'].replace(['Lady', 'Countess','Capt', 'Col',\
 	'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Other')

dataset['Title'] = dataset['Title'].replace('Mlle', 'Miss')
dataset['Title'] = dataset['Title'].replace('Ms', 'Miss')
dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')
    
dataset[['Title', 'Survived']].groupby(['Title'], as_index=False).mean()

# Now we will map alphanumerical categories to numbers
title_mapping = { 'Mr': 1, 'Miss': 2, 'Mrs': 3, 'Master': 4, 'Other': 5 }
gender_mapping = { 'female': 1, 'male': 0 }
port_mapping = { 'S': 0, 'C': 1, 'Q': 2 }

# Map title
dataset['Title'] = dataset['Title'].map(title_mapping).astype(int)

# Map gender
dataset['Sex'] = dataset['Sex'].map(gender_mapping).astype(int)

# Map port
freq_port = dataset.Embarked.dropna().mode()[0]
dataset['Embarked'] = dataset['Embarked'].fillna(freq_port)
dataset['Embarked'] = dataset['Embarked'].map(port_mapping).astype(int)

# Fix missing age values
dataset['Age'] = dataset['Age'].fillna(dataset['Age'].dropna().median())

dataset.head()

# Prepare the data
X = dataset.drop(['Survived'], axis = 1).values
y = dataset[['Survived']].values

X = StandardScaler().fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state = None)

# Prepare cross-validation (cv)
cv = KFold(n_splits = 5, random_state = None)

# Performance
p_score = lambda model, score: print('Performance of the %s model is %0.2f%%' % (model, score * 100))

# Classifiers
names = [
    "Logistic Regression", "Logistic Regression with Polynomial Hypotheses",
    "Linear SVM", "RBF SVM", "Neural Net",
]

classifiers = [
    LogisticRegression(),
    make_pipeline(PolynomialFeatures(3), LogisticRegression()),
    SVC(kernel="linear", C=0.025),
    SVC(gamma=2, C=1),
    MLPClassifier(alpha=1),
]
# iterate over classifiers
models = []
trained_classifiers = []
for name, clf in zip(names, classifiers):
    scores = []
    for train_indices, test_indices in cv.split(X):
        clf.fit(X[train_indices], y[train_indices].ravel())
        scores.append( clf.score(X_test, y_test.ravel()) )
    
    min_score = min(scores)
    max_score = max(scores)
    avg_score = sum(scores) / len(scores)
    
    trained_classifiers.append(clf)
    models.append((name, min_score, max_score, avg_score))
    
fin_models = pd.DataFrame(models, columns = ['Name', 'Min Score', 'Max Score', 'Mean Score'])
fin_models.sort_values(['Mean Score']).head()


# ----------------------------- ----------------------------- ----------------------------- -----------------------------
# ----------------------------- ----------------------------- ----------------------------- -----------------------------
# ----------------------------- ----------------------------- ----------------------------- -----------------------------
# Finaliza Train Inicia Test
# ----------------------------- ----------------------------- ----------------------------- -----------------------------
# ----------------------------- ----------------------------- ----------------------------- -----------------------------
# ----------------------------- ----------------------------- ----------------------------- -----------------------------

data_path = os.path.join(os.getcwd(), "../input//tittest21/test2.csv")
dataset = read_csv(data_path, skipinitialspace=True)

dataset.head(5)

# We need to drop some insignificant features and map the others.
# Ticket number and fare should not contribute much to the performance of our models.
# Name feature has titles (e.g., Mr., Miss, Doctor) included.
# Gender is definitely important.
# Port of embarkation may contribute some value.
# Using port of embarkation may sound counter-intuitive; however, there may 
# be a higher survival rate for passengers who boarded in the same port.

dataset['Title'] = dataset.Name.str.extract(' ([A-Za-z]+)\.', expand=False)
dataset = dataset.drop(['PassengerId', 'Ticket', 'Cabin', 'Name'], axis=1)

pd.crosstab(dataset['Title'], dataset['Sex'])

# We will replace many titles with a more common name, English equivalent,
# or reclassification
dataset['Title'] = dataset['Title'].replace(['Lady', 'Countess','Capt', 'Col',\
 	'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Other')

dataset['Title'] = dataset['Title'].replace('Mlle', 'Miss')
dataset['Title'] = dataset['Title'].replace('Ms', 'Miss')
dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')
    
#dataset[['Title', 'Survived']].groupby(['Title'], as_index=False).mean()

# Now we will map alphanumerical categories to numbers
title_mapping = { 'Mr': 1, 'Miss': 2, 'Mrs': 3, 'Master': 4, 'Other': 5 }
gender_mapping = { 'female': 1, 'male': 0 }
port_mapping = { 'S': 0, 'C': 1, 'Q': 2 }

# Map title
dataset['Title'] = dataset['Title'].map(title_mapping).astype(int)

# Map gender
dataset['Sex'] = dataset['Sex'].map(gender_mapping).astype(int)

# Map port
freq_port = dataset.Embarked.dropna().mode()[0]
dataset['Embarked'] = dataset['Embarked'].fillna(freq_port)
dataset['Embarked'] = dataset['Embarked'].map(port_mapping).astype(int)

# Fix missing age values
dataset['Age'] = dataset['Age'].fillna(dataset['Age'].dropna().median())

dataset.head()

X = dataset.drop(['Survived'], axis = 1).values
y = dataset[['Survived']].values

# Prepare the data
X = StandardScaler().fit_transform(X)

trained_classifiers[4].predict(X)
