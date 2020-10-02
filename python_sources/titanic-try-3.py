# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn import model_selection
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import RadiusNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import Normalizer
from sklearn.preprocessing import Binarizer
from sklearn.preprocessing import scale
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

# TRAIN DATA
path ='../input/train.csv'
df = pd.read_csv(path,header = 0)
# map string to ints
df['Gender'] = df['Sex'].map({'female' : 0, 'male' : 1}).astype(int)
# remove unused cols, Sex is already transformed to Gender
df.drop(['Name','Cabin','Sex','Ticket','PassengerId'], axis = 1, inplace = True)
### Data cleaning
# Embarked: convert nulls to common place
if len(df.loc[df.Embarked.isnull(),'Embarked']) > 0:
    df.loc[df.Embarked.isnull(),'Embarked'] = df['Embarked'].dropna().mode().values
Ports = list(enumerate(np.unique(df['Embarked']))) # all values of Embarked
Ports_dict = { name : i for i, name in Ports } # set up a dict in the form Ports :index
df['Embarked'] = df['Embarked'].map(lambda x: Ports_dict[x]).astype(int) # convert all embark string to int
# Age: convert nulls to median
median_age = df['Age'].dropna().median()
if len(df.loc[df.Age.isnull(),'Age']) > 0:
    df.loc[(df.Age.isnull()),'Age'] = median_age
# TEST DATA
# TODO
path_test_data = '../input/test.csv'
df_test = pd.read_csv(path_test_data, header = 0)
df_test['Gender'] = df_test['Sex'].map({'female' : 0, 'male' : 1}).astype(int)
df_test.drop(['Name','Cabin','Sex','Ticket','PassengerId'], axis = 1,inplace =True)
df_test['Embarked'] = df_test['Embarked'].map(lambda x: Ports_dict[x]).astype(int)
array = df.values
Y = array[:,0]
selector = [x for x in range(array.shape[1]) if x != 0]
X = array[:,selector]
seed = 7
validation_size = 0.20
scoring = 'accuracy'
Xs = scale(X)
X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(Xs, Y, test_size=validation_size, random_state=seed)

array_test = df_test.values
X_test = array_test[:]
# print(df.head())
# # scaling
# scaler = MinMaxScaler(feature_range=(0,1))
# rescaledX = scaler.fit_transform(X)
# # standarazing
# # scaler = StandardScaler().fit(X)
# # rescaledX = scaler.transform(X)
# # normalizing
# # scaler = Normalizer().fit(X)
# # normalizedX = scaler.transform(X)
# # Binarizing
# # binarizer = Binarizer(threshold=0.0).fit(X)
# # binaryX = binarizer.transform(X)

# models = []
# models.append(('LR', LogisticRegression()))
# models.append(('LDA', LinearDiscriminantAnalysis()))
# models.append(('LSVM', LinearSVC()))
# models.append(('SVM', SVC()))
# models.append(('KNN', KNeighborsClassifier()))
# models.append(('CART', DecisionTreeClassifier()))
# models.append(('NB', GaussianNB()))
# results = []
# names = []
# for name,model in models:
#     kfold = model_selection.KFold(n_splits=10, random_state=seed)
#     cv_results = model_selection.cross_val_score(model, X_train, Y_train, cv=kfold,n_jobs=-1, scoring=scoring)
#     results.append(cv_results)
#     names.append(name)
#     msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
#     print(msg)

# lr = LogisticRegression()
# lr.fit(X_train,Y_train)
# predictions = lr.predict(X_validation)
# print("LOGISTIC REGRESSION")
# print(accuracy_score(Y_validation,predictions))
# print(classification_report(Y_validation,predictions))
# lda = LinearDiscriminantAnalysis()
# lda.fit(X_train,Y_train)
# predictions_lda = lda.predict(X_validation)
# print("Linear Discriminant Analysis")
# print(accuracy_score(Y_validation,predictions_lda))
# print(classification_report(Y_validation, predictions_lda))
# lsvm = LinearSVC()
# lsvm.fit(X_train,Y_train)
# predictions_lsvm = lsvm.predict(X_validation)
# print("Linear SVC")
# print(accuracy_score(Y_validation,predictions_lsvm))
# print(classification_report(Y_validation, predictions_lsvm))
# svm = SVC()
# svm.fit(X_train,Y_train)
# predictions_svm = svm.predict(X_validation)
# print("SVC")
# print(accuracy_score(Y_validation,predictions_svm))
# print(classification_report(Y_validation, predictions_svm))
print("KNeighbors Classifier")
knn = KNeighborsClassifier()
knn.fit(X_train,Y_train)
predictions_knn = knn.predict(X_validation)
print(accuracy_score(Y_validation, predictions_knn))
print(classification_report(Y_validation, predictions_knn))
print(confusion_matrix(Y_validation,predictions_knn))
rnc = RadiusNeighborsClassifier(radius=1.8)
rnc.fit(X_train,Y_train)
predictions_rnc = rnc.predict(X_validation)
print("Radius Neighbors Classifier")
print(accuracy_score(Y_validation, predictions_rnc))
print(classification_report(Y_validation, predictions_rnc))

print("KNeighbors Classifier test data")
knn_test = KNeighborsClassifier()
knn_test.fit(X_train,Y_train)
predictions_knn_test = knn_test.predict(X_test)
output_path = 'results.csv'
predictions_knn_test.to_csv(outputpath)

