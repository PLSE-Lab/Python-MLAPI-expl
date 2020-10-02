'''
In the Following code we will perform machine learning to determine if
a pokemon is classified as a legendary or not, using Random Forest 
Classification
'''

#Importing the Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

#Importing the dataset
dataset = pd.read_csv('/kaggle/input/pokemon/pokemon.csv')
X = dataset.iloc[:,[19,22,25,27,28,33,34,35,38,]]
y = dataset.iloc[:,-1]

'''
We choose to keep the columns of attack, base total, defense, 
height(meters), hp, attack, special defense, speed, and weight(kilograms). 
We will train our dataset to determine if it can learn 
if a pokemon is legendary or not
'''

# Taking care of missing data, we use mean data instead of filling in with
#real values of the pokemon
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values = np.nan, strategy = 'mean',
                        fill_value = 'constant')
imputer = imputer.fit(X)
X = imputer.transform(X)

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Fitting Random Forest Classification to the Training set
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators = 10, 
                                    criterion = 'entropy')
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix and printing the results
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)

#Applying K-Fold Cross Validation to see accuracy and standard deviation
from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator = classifier, X = X_train,
                             y = y_train, cv = 10)
print('accuracy = ' + str(accuracies.mean()))
print('standard deviation = ' + str(accuracies.std()))