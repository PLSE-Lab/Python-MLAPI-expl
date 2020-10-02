# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

# Mushrooms Classification Dataset

# Import Libraries
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report

# Importing Dataset
dataset = pd.read_csv("../input/mushrooms.csv")

# Shape of Dataset (Samples, Features)
print(dataset.shape)

# Select Cloumns List
dataset.columns

# Describe Data
print(dataset.describe())

# Identify which columns have missing values
print(dataset.isnull().sum())

# Class Distribution
print(dataset.groupby('class').size())

# Encoding Categorical Data
dataset.convert_objects(convert_numeric = True)
dataset.fillna(0, inplace = True)

def handle_non_numerical_data(dataset):
    columns = dataset.columns. values
    
    for column in columns:
        text_digit_vals = {}
        def convert_to_int(val):
            return text_digit_vals[val]
        
        if dataset[column].dtype !=  np.int64 and dataset[column].dtype !=  np.float64:
            column_contents = dataset[column].values.tolist()
            unique_elements = set(column_contents)
            x = 0
            for unique in unique_elements:
                if unique not in text_digit_vals:
                    text_digit_vals[unique] = x
                    x+=1
            dataset[column] = list(map(convert_to_int, dataset[column]))
            
    return dataset

dataset = handle_non_numerical_data(dataset)
print(dataset.head())

# Assign Independent & Dependent Variable
X = dataset.iloc[:, 1:23].values
Y = dataset.iloc[:, 0].values

# Splitting The Dataset into Training Set and Testing Set
X_train, X_test, Y_train, Y_test = train_test_split(X, Y,test_size=0.20,random_state=198)

# Check if split is actually correct
print(float(X_train.shape[0]) / float(dataset.shape[0]))
print(float(X_test.shape[0]) / float(dataset.shape[0]))

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.fit_transform(X_test)
print(X_train)
print(X_test)

# Print Mean of X_train and X_test (Execute Each Line Once) => Expected (Near Zero)
print(X_train.mean(axis=0))
print(X_test.mean(axis=0))

# Print Std of X_train and X_test (Execute Each Line Once) => Expected (All Ones)
print(X_train.std(axis=0))
print(X_test.std(axis=0))

# Fitting SVM Classifier to Training Set
from sklearn.svm import SVC
classifier = SVC(kernel = 'rbf', gamma = 'auto', random_state = 0)
classifier.fit(X_train, Y_train)

# Predicting the Test Set Results
Y_Pred = classifier.predict(X_test)

# Making the Confusion Matrix
cm = confusion_matrix(Y_test, Y_Pred)
print("Accuracy: %s" %classifier.score(X_test, Y_test))

# Stratified K Fold Cross Validation
from sklearn.cross_validation import cross_val_score
cvs = cross_val_score(estimator = classifier, X = X_train, y = Y_train, cv = 10, n_jobs = -1)
print('CV Accuracy Scores: %s' % cvs)
print('CV Accuracy: %.3f +/- %.3f' %(np.mean(cvs), np.std(cvs)))
print(classification_report(Y_test, Y_Pred))


# Stratified K Fold Cross Validation
from sklearn.cross_validation import cross_val_score
cvs = cross_val_score(estimator = classifier, X = X_train, y = Y_train, cv = 10, n_jobs = -1)
print('CV Accuracy Scores: %s' % cvs)
print('CV Accuracy: %.3f +/- %.3f' %(np.mean(cvs), np.std(cvs)))
print(classification_report(Y_test, Y_Pred))

# Any results you write to the current directory are saved as output.