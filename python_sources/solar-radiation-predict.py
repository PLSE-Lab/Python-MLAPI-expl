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
from sklearn import model_selection
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

# Load dataset
names = ['utime','date','time','radiation','temp','pressure','humidity','wind','speed','timeRise','timeSet']
dataset = pd.read_csv('../input/SolarPrediction.csv',header='infer')

# shape
print(dataset.shape)

print(dataset.head(20))

# stats
print(dataset.describe())

# Split-out validation dataset
array = dataset.values
X = array[:,4:9]
Y = array[:,3]
Y = Y.astype(str)
validation_size = 0.20
seed = 7
X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(X, Y, test_size=validation_size, random_state=seed)

# Test options and evaluation metric
scoring = 'accuracy'


# Spot Check Algorithms
models = []
#models.append(('LR', LogisticRegression()))
#models.append(('LDA', LinearDiscriminantAnalysis()))
#models.append(('KNN', KNeighborsClassifier()))          # KNN: 0.056678 (0.002976)
#models.append(('CART', DecisionTreeClassifier()))        # CART: 0.052241 (0.004466)
#models.append(('NB', GaussianNB()))
models.append(('SVM', SVC()))
# evaluate each model in turn
results = []
names = []
for name, model in models:
	kfold = model_selection.KFold(n_splits=10, random_state=seed)
	cv_results = model_selection.cross_val_score(model, X_train, Y_train, cv=kfold, scoring=scoring)
	results.append(cv_results)
	names.append(name)
	msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
	print(msg)

'''
# Make predictions on validation dataset
predict = False
model = LogisticRegression()
if predict:
    print('-'*50)
    print('start predict')
    
    #knn.fit(X_train, Y_train)
    model.fit(X, Y)
    #predictions = model.predict(X_validation)
    predictions = model.predict(X_test)
    #print predictions
    with open('result.csv', 'w') as fout:
        fout.write('PassengerId,Survived\n')
        for i in range(0, len(predictions)):
            fout.write('{id},{v}\n'.format(id=dataset_test.values[i][0], v=predictions[i]))
else:
    print ('-'*50)
    model.fit(X_train, Y_train)
    predictions = model.predict(X_validation)
    print(predictions)
    print(accuracy_score(Y_validation, predictions))
    print('-'*50)
    print(confusion_matrix(Y_validation, predictions))
    print('-'*50)
    print(classification_report(Y_validation, predictions))
    print('-'*50)
    '''