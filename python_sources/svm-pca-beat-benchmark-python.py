import pandas as pd

# The competition datafiles are in the directory ../input
# Read competition data files:
train = pd.read_csv("../input/train.csv")
test  = pd.read_csv("../input/test.csv")

import random

from sklearn import decomposition, cross_validation, metrics, neighbors, preprocessing
from sklearn import svm
from sklearn.grid_search import GridSearchCV
 
#read data
data = train

#remove zero sum columns
#data = data.loc[data.sum(axis=1)!=0,data.sum(axis=0)!=0]
feature  = data.iloc[:,1:]
label    = data['label']

#pca


#divide the data into train and tesb set
x_train,x_test,y_train,y_test = cross_validation.train_test_split(feature, label, test_size = 0.1 , random_state = 123)

# std_scale = preprocessing.StandardScaler().fit(x_train)
# x_train = std_scale.transform(x_train)
# x_test  = std_scale.transform(x_test)

pca = decomposition.PCA(n_components=0.8,whiten=True)
x_train = pca.fit_transform(x_train)
x_test  = pca.transform(x_test)

#fit model

#param_grid = [
#  {'C': [1, 10, 100, 1000], 'kernel': ['linear']},
#  {'C': [1, 10, 100, 1000], 'gamma': [0.001, 0.0001], 'kernel': ['rbf']},
# ]

svm = svm.SVC()
#grid_search = GridSearchCV(svm, param_grid=param_grid)
#grid_search.fit(x_train, y_train)
svm.fit(x_train, y_train)
#test model
y_pred = svm.predict(x_test)

#accuracy, confusion matrix and report
print(metrics.accuracy_score(y_test,y_pred))
print(metrics.confusion_matrix(y_test,y_pred))
print(metrics.classification_report(y_test,y_pred))

####  validation ####

#read data
valid = test

#apply pca trasform
x_valid = pca.transform(valid)

#predict labels
y_valid = pd.DataFrame(svm.predict(x_valid))

#write to csv file for submission
y_valid.to_csv("submission_svm_pca_data.csv")
