# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('../input/train.csv')
X = dataset.iloc[:,1:].values
y = dataset.iloc[:,:1].values

# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

# Creating the RF classifier
from sklearn.ensemble import RandomForestClassifier

for i in range(10,50,10):
    print(i)

sc_=np.empty((10,1),float)
i_index = [10,20,30,40,50,60,70,80,90,100]

for i in range(1,11):
    Rf_classifier = RandomForestClassifier(n_estimators= i*10)
    Rf_classifier.fit(X_train, y_train)
    sc_[i-1,0] = Rf_classifier.score(X_test, y_test)
print (sc_)
#Plotting the accuracy vs no of trees
plt.plot(i_index, sc_)
plt.title('Accuracy vs No of Trees')
plt.xlabel('No of trees')
plt.ylabel('Accuracy')    

# Reading actual test set
test_set = pd.read_csv("../input/test.csv")
#Retrain the model on the whole training set
Rf_classifier.fit(X,y)
#Predict on actual test set
y_pred = Rf_classifier.predict(test_set)
#Submit to Kaggle
submission=pd.DataFrame({
        "ImageId": list(range(1,len(y_pred)+1)),
        "Label": y_pred
    })
submission.to_csv("Digit_RF-a_AL.csv", index=False, header=True)