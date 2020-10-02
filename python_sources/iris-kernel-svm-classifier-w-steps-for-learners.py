"""
    @author: Rony Lussari
    LinkedIn: https://www.linkedin.com/in/ronylussari/
    
    Solution to the famous Iris Data Set with UNNECESSARY  code repetition just for LEARNING purpose.
    
    Code applying what was learnt in the "Super Data Science" course called "Machine Learning A to Z on Udemy".

"""

#------------------IRIS CLASSIFICATION------------------------
#Importing libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#importing dataset
dataset = pd.read_csv('../input/Iris.csv')

#Feature selection
selected_features = [0, 1, 2, 3]
X = dataset.iloc[:,selected_features].values
y= dataset.iloc[:, -1].values

#Split dataset into training nad test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= 0.25) 

#Fitting Kernel SVM to the training set
from sklearn.svm import SVC
classifier = SVC(kernel='rbf')
classifier.fit(X_train, y_train)

#Predictiong the test set results
y_pred = classifier.predict(X_test)

#Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

# Applying k-Fold Cross Validation
from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator= classifier, X = X_train, y = y_train, cv=5, n_jobs=-1)
acc_mean = accuracies.mean()
acc_std = accuracies.std()

print("Avg. Accuracy: ", round(acc_mean*100),"% ", " Std: ", round(acc_std*100,2),"%")

# Applying Grid Search to find the best model and the best parameters
from sklearn.model_selection import GridSearchCV
parameters = [{'C': [1, 10, 100, 1000], 'kernel': ['linear']},
              {'C': [1, 10, 100, 1000], 'kernel': ['poly'], 
                   'gamma': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9], 'degree':[1, 2, 3, 4, 5, 6, 7, 8, 9] },
              {'C': [1, 10, 100, 1000], 'kernel': ['rbf'], 
                    'gamma': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]},
              {'C': [1, 10, 100, 1000], 'kernel': ['sigmoid'], 
                   'gamma': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]}]
grid_search = GridSearchCV(estimator = classifier,
                           param_grid = parameters,
                           scoring = 'accuracy',
                           cv = 5,
                           n_jobs = -1)
grid_search = grid_search.fit(X_train, y_train)
best_accuracy = grid_search.best_score_
best_parameters = grid_search.best_params_

print(best_accuracy)
print( best_parameters)

#Fine tuning
from sklearn.model_selection import GridSearchCV
parameters = [{'C': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.9, 1], 'kernel': ['linear']}]
grid_search = GridSearchCV(estimator = classifier,
                           param_grid = parameters,
                           scoring = 'accuracy',
                           cv = 5,
                           n_jobs = -1)
grid_search = grid_search.fit(X_train, y_train)
best_accuracy = grid_search.best_score_
best_parameters = grid_search.best_params_

print(best_accuracy)
print( best_parameters)

#Testing Logistic Regression ---------------------------------------------
#Fitting Logistic Regression to the training set

dataset = pd.read_csv('../input/Iris.csv')

#Feature selection
selected_features = [0, 1, 2, 3]
X = dataset.iloc[:,selected_features].values
y= dataset.iloc[:, -1].values


from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression()
classifier.fit(X_train, y_train)

#Predictiong the test set results
y_pred = classifier.predict(X_test)

#Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

# Applying k-Fold Cross Validation
from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator= classifier, X = X_train, y = y_train, cv=5, n_jobs=-1)
acc_mean = accuracies.mean()
acc_std = accuracies.std()

print("Avg. Accuracy: ", round(acc_mean*100),"% ", " Std: ", round(acc_std*100,2),"%")

# Applying Grid Search to find the best model and the best parameters
from sklearn.model_selection import GridSearchCV
parameters = [{'C': [1, 10, 100, 1000], 'penalty': ['l1', 'l2']}]

               
grid_search = GridSearchCV(estimator = classifier,
                           param_grid = parameters,
                           scoring = 'accuracy',
                           cv = 5,
                           n_jobs = -1)
grid_search = grid_search.fit(X_train, y_train)
best_accuracy2 = grid_search.best_score_
best_parameters2 = grid_search.best_params_

print(best_accuracy2)
print( best_parameters2)

#Fine tuning
from sklearn.model_selection import GridSearchCV
parameters = [{'C': [0.5, 0.6, 0.9, 1], 'penalty': ['l1', 'l2']}]
grid_search = GridSearchCV(estimator = classifier,
                           param_grid = parameters,
                           scoring = 'accuracy',
                           cv = 5,
                           n_jobs = -1)
grid_search = grid_search.fit(X_train, y_train)
best_accuracy2 = grid_search.best_score_
best_parameters2 = grid_search.best_params_

print(best_accuracy2)
print( best_parameters2)

# Comparing the models
print('SVM Accuracy:                 ', best_accuracy)
print('Logistic Regression Accuracy: ', best_accuracy2)

#Similar results, but Kernel SVM is slightly better.

#-------------------------------------------------------------------------------------

#FINAL CODE: Fitting to the choose model (Kernel SVM), now with better parameters-------------------------

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#importing dataset
dataset = pd.read_csv('../input/Iris.csv')

#Feature selection
selected_features = [0, 1, 2, 3]
X = dataset.iloc[:,selected_features].values
y= dataset.iloc[:, -1].values

#Splitting dataset into training nad test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= 0.25) 


#Fitting Kernel SVM to the training set
from sklearn.svm import SVC
classifier = SVC(C=0.6, kernel='linear')
classifier.fit(X_train, y_train)

#Predictiong the test set results
y_pred = classifier.predict(X_test)

#Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

# Applying k-Fold Cross Validation
from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator= classifier, X = X_train, y = y_train, cv=5, n_jobs=-1)
acc_mean = accuracies.mean()
acc_std = accuracies.std()

print("Avg. Accuracy: ", round(acc_mean*100),"% ", " Std: ", round(acc_std*100,2),"%")


#Using LDA to plot results in 2 dimensions 

#Feature selection
selected_features = [0, 1, 2, 3]
X = dataset.iloc[:,selected_features].values
y= dataset.iloc[:, -1].values


#Encoding categorical data
from sklearn.preprocessing import LabelEncoder
labelencoder_y = LabelEncoder()
y= labelencoder_y.fit_transform(y)


#Spliting dataset
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

#Feature scalling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)


# Applying LDA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
lda = LDA(n_components = 2)
X_train = lda.fit_transform(X_train, y_train)
X_test = lda.transform(X_test)


# Fitting Logistic Regression to the Training set
from sklearn.svm import SVC
classifier = SVC(C=0.6, kernel='linear')
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)


# Visualising the Test set results (Code from SDS ML A-Z)
from matplotlib.colors import ListedColormap
X_set, y_set = X_test, y_test
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green', 'blue')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ListedColormap(('red', 'green', 'blue'))(i), label = j)
plt.title('Kernel SVM Classifier on Iris Data Set (Test set)')
plt.xlabel('LDA-1')
plt.ylabel('LDA-2')
plt.legend()

plt.show()