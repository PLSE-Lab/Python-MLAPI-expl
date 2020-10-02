import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
from pandas.tools.plotting import scatter_matrix
from sklearn import model_selection
# from sklearn.metrics import classification_report
# from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
# from sklearn.linear_model import LinearRegression
# from sklearn.tree import DecisionTreeClassifier
# from sklearn.neighbors import KNeighborsClassifier
# from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
# from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
import operator
# from subprocess import check_output
# print(check_output(["ls", "../input"]).decode("utf8"))

path = '../input/iris/Iris.csv'
dataset = pd.read_csv(path, header=0)

array = dataset.values
X = array[:,1:5]
Y = array[:,5]
validation_size = 0.46
# seed_array = np.arange(100)
seed = 69
scoring = 'accuracy'
results = {}
# for seed in seed_array:
X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(X,Y, test_size=validation_size, random_state=seed)
# models.append(('SVM', SVC()))
# results = []
# names = []
# for name, model in models:
#     kfold = model_selection.KFold(n_splits=10,random_state=seed)
#     cv_results = model_selection.cross_val_score(model, X_train, Y_train, cv=kfold, scoring=scoring)
#     results.append(cv_results)
#     names.append(name)
#     msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
#     print(msg)
svm = SVC()
svm.fit(X_train, Y_train)
predictions = svm.predict(X_validation)
score = accuracy_score(Y_validation, predictions)
print(score)
print(seed)
#     results[seed] = score
# print(results)
# max(results.iteritems(), key=operator.itemgetter(1))[0]

# knn = KNeighborsClassifier()
# knn.fit(X_train, Y_train)
# predictions = knn.predict(X_validation)
# print(confusion_matrix(Y_validation, predictions))
# print(classification_report(Y_validation, predictions))

# print(dataset.shape)
# (150, 6) 150 instances and 6 attributes

# print(dataset.head(20))
# take top 20 records

# print(dataset.describe())
# descriptiopns count,mean,std,min,max

# print(dataset.groupby('Species').size())
# species distribution

# dataset.plot(kind='box', subplots=True, layout=None, sharex=False, sharey=False)
# plt.show()
