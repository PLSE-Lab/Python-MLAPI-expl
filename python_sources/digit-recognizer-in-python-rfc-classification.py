# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import time

# Importing the dataset
dataset = pd.read_csv('train.csv')
X_train = dataset.iloc[:, 1:].values
y_train = dataset.iloc[:, 0].values
X_t=pd.read_csv('test.csv')

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_t)

# Fitting Random Forest Classification to the Training set
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators = 80, criterion = 'entropy', random_state = 0)
classifier.fit(X_train, y_train)

# Predicting the Test set results

y_pred = classifier.predict(X_test)

out_file = open("RFC-Jub80-PredictedImageslable.csv", "w")
out_file.write("ImageId,Label\n")
for i in range(len(y_pred)):
    out_file.write(str(i+1) + "," + str(int(y_pred[i])) + "\n")
out_file.close()
