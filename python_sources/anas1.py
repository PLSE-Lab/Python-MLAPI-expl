import numpy as np
import pandas as pd
import sklearn.linear_model as lm
from sklearn import tree,metrics,base
from sklearn.neural_network import MLPClassifier,MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV,train_test_split,cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB

data = pd.read_csv('../input/voice.csv')
print ("Dimension of data {}".format(data.shape))
#print(data.head())

##Encoding male female
#male = 0, female = 1
data.label = np.where(data.label=='male', 0, 1)
#print(data.head())

X = data.drop(['label'],axis=1)
y = data.label
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)


##Neural Network - Classifier
    ## Different solver and number of hidden layers
print("Neural Network - Classifier")
clf = MLPClassifier(activation = 'logistic',solver='lbfgs', alpha=0.0001,
                     random_state=1)
clf.fit(X_train,y_train)

predictedlabel = clf.predict(X_test)
precision = metrics.precision_score(y_test, predictedlabel, average='weighted', sample_weight=None)
accuracy = metrics.accuracy_score(y_test, predictedlabel, normalize=True, sample_weight=None)
print("Accuracy is: " + str(accuracy))
print("Precision is: " + str(precision))


print("Neural Network - Classifier with different hidden layers and solver")
clf = MLPClassifier(activation = 'logistic',solver='adam', hidden_layer_sizes=(200,25), alpha=0.001,
                     random_state=1,max_iter=300)
clf.fit(X_train,y_train)

predictedlabel = clf.predict(X_test)
precision = metrics.precision_score(y_test, predictedlabel, average='weighted', sample_weight=None)
accuracy = metrics.accuracy_score(y_test, predictedlabel, normalize=True, sample_weight=None)
print("Accuracy is: " + str(accuracy))
print("Precision is: " + str(precision))

##Neural Netowkr - Regressor
print("Neural Network - Regression")
clf = MLPRegressor(activation = 'logistic',solver='lbfgs', alpha=0.0001,
                     random_state=1)
clf.fit(X_train,y_train)

predictedlabel = clf.predict(X_test)
##Changing less than 0.5 to 0
    ##and greater than equal to 0.5 to 1

predictedlabel [predictedlabel<0.5] = 0
predictedlabel [predictedlabel>=0.5] = 1

precision = metrics.precision_score(y_test, predictedlabel, average='weighted', sample_weight=None)
accuracy = metrics.accuracy_score(y_test, predictedlabel, normalize=True, sample_weight=None)
print("Accuracy is: " + str(accuracy))
print("Precision is: " + str(precision))


print("Neural Network Regressor with different number of hidden layers")
clf = MLPRegressor(activation = 'logistic',solver='adam', hidden_layer_sizes=(200,25), alpha=0.001,
                     random_state=1,max_iter=300)
clf.fit(X_train,y_train)

predictedlabel = clf.predict(X_test)
predictedlabel [predictedlabel<0.5] = 0
predictedlabel [predictedlabel>=0.5] = 1

precision = metrics.precision_score(y_test, predictedlabel, average='weighted', sample_weight=None)
accuracy = metrics.accuracy_score(y_test, predictedlabel, normalize=True, sample_weight=None)
print("Accuracy is: " + str(accuracy))
print("Precision is: " + str(precision))

##Random Forest
print("Random Forest: ")
random_forest = RandomForestClassifier(n_estimators=100)
random_forest.fit(X_train, y_train)
y_pred = random_forest.predict(X_test)
precision = metrics.precision_score(y_test, y_pred, average='weighted', sample_weight=None)
accuracy = metrics.accuracy_score(y_test, y_pred, normalize=True, sample_weight=None)

print("Accuracy is: " + str(accuracy))
print("Precision is: " + str(precision))

print(random_forest.score(X_train, y_train))

##KNN
print("K nearest Neighbor")

knn = KNeighborsClassifier(n_neighbors = 3)
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)
precision = metrics.precision_score(y_test, y_pred, average='weighted', sample_weight=None)
accuracy = metrics.accuracy_score(y_test, y_pred, normalize=True, sample_weight=None)

print("Accuracy is: " + str(accuracy))
print("Precision is: " + str(precision))
print(knn.score(X_train, y_train))

submission =  X_test.copy()
submission['label'] = y_pred.astype(int)
submission.to_csv('output.csv', index=False)
print ("Dimension of output data {}".format(submission.shape))

##Decision Trees
print("Decision Trees")

dec = tree.DecisionTreeClassifier()
dec.fit(X_train, y_train)

y_pred = dec.predict(X_test)
precision = metrics.precision_score(y_test, y_pred, average='weighted', sample_weight=None)
accuracy = metrics.accuracy_score(y_test, y_pred, normalize=True, sample_weight=None)

print("Accuracy is: " + str(accuracy))
print("Precision is: " + str(precision))

##No need for data cleaning
####Actual Model Building

