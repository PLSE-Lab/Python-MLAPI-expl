# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
from sklearn.metrics import matthews_corrcoef



# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


import keras
from keras import Sequential
from keras.layers import Dense, Dropout

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split



# Function definition to perform calculate the performance measures.
def modelPerformanceMeasure(y_test, y_pred) :
    cm = confusion_matrix(y_test, y_pred)
    print(cm)
    precission = (1.0 * cm[1][1])/(cm[1][0]+cm[1][1])
    recall = (1.0 * cm[1][1])/(cm[0][1] + cm[1][1])
    accuracy = (1.0 * cm[0][0]+cm[1][1])/(cm[0][0]+cm[1][1]+cm[0][1]+cm[1][0])
    f1_score = 2 * precission * recall /(precission + recall)
    mcc = matthews_corrcoef(y_test, y_pred)
    print("Precission : {}, Recall: {}, Accurcy : {}, f1_score : {}, mcc_score : {}".format(precission, recall, accuracy, f1_score, mcc))
    
    return (precission, recall, f1_score, mcc)



# Based on the threshhold decide 1 or 0
def convertBinary(arr, threshhold= 0.5) :
    result = []
    print("Threshhold : ", threshhold)
    
    for val in arr :
        if val >= threshhold :
            result.append(1)
        else :
            result.append(0)
    return result



# Load the training and test datasets
dataset = pd.read_csv('../input/criminal_train.csv')
testset = pd.read_csv('../input/criminal_test.csv')

dataset.head()


# Since the training set is a highly inbalanced dataset,
# Repeating the count of 1's to make it little balanced
tempSet = dataset[dataset.Criminal==1]
randomIds=np.random.randint(0, len(tempSet.Criminal), int(len(tempSet.Criminal) * .90))
dataset = dataset.append(tempSet.iloc[randomIds,:])
dataset = dataset.sample(frac=1).reset_index(drop=True)

randomIds=np.random.randint(0, len(tempSet.Criminal), int(len(tempSet.Criminal) * .70))
dataset = dataset.append(tempSet.iloc[randomIds,:])
dataset = dataset.sample(frac=1).reset_index(drop=True)

randomIds=np.random.randint(0, len(tempSet.Criminal), int(len(tempSet.Criminal) * .50))
dataset = dataset.append(tempSet.iloc[randomIds,:])
dataset = dataset.sample(frac=1).reset_index(drop=True)

randomIds=np.random.randint(0, len(tempSet.Criminal), int(len(tempSet.Criminal) * .20))
dataset = dataset.append(tempSet.iloc[randomIds,:])
dataset = dataset.sample(frac=1).reset_index(drop=True)



# Removing the not not necessary columns and target variable from the training set.
X = dataset.drop(["PERID","Criminal"], axis =1)
testset_ids = testset.PERID
testset = testset.drop(["PERID"], axis=1)


# Getting the target variable into the variable y
y = dataset.loc[:,"Criminal"]
y = y.astype("category")
y.value_counts()



# Missing value treatment
for i in range(0, 70) :
    if sum(pd.isnull(X.iloc[:,i])) != 0:
        print(i)
# No missing values found.    


# Splitting the dataset into the Training set and Test set
X_train, X_test, y_train, y_test = train_test_split(X, y, 
                             test_size = 0.25, 
                             random_state = 100)

y_train.value_counts()
y_test.value_counts()


for i in range(0, 70) :
    if len(X.iloc[:,i].unique()) < 20 :
        X.iloc[:,i] = X.iloc[:,i].astype("category")
        testset.iloc[:,i] = testset.iloc[:,i].astype("category")


# Feature Scaling
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

testset = sc.transform(testset)



classifierANN = Sequential()
classifierANN.add(Dense(units=64,
                activation="relu", 
                input_dim=70, 
                kernel_initializer='uniform',
            ))

classifierANN.add(Dropout(rate=0.1))
classifierANN.add(Dense(units=32,
                activation="relu", 
                kernel_initializer='uniform',
            ))
classifierANN.add(Dropout(rate=0.1))
classifierANN.add(Dense(units=1, 
                activation="sigmoid", 
                kernel_initializer='uniform'
            ))


optimizer=keras.optimizers.Nadam(lr=0.001, 
                                 beta_1=0.7, 
                                 beta_2=0.999, 
                                 epsilon=None, 
                                 schedule_decay=0.004)

classifierANN.compile(loss='binary_crossentropy', 
                      optimizer=optimizer, 
                      metrics=['accuracy'] )

classifierANN.fit(X_train, y_train, epochs=25, 
                  batch_size = 15, 
                  validation_data=(X_test, y_test),
                 )


for i in range(1, 10, 1) :
    threshhold = i/10.0
    y_pred = convertBinary(classifierANN.predict(X_test), threshhold=threshhold)
    print(modelPerformanceMeasure(y_test, y_pred) )

threshhold = 0.405
y_pred = convertBinary(classifierANN.predict(X_test), threshhold=threshhold)
print(modelPerformanceMeasure(y_test, y_pred) )


testset_result = convertBinary(classifierANN.predict(testset), threshhold=threshhold)

pd.Series(testset_result).value_counts()


result = pd.DataFrame({
            "PERID" : testset_ids,
            "Criminal" : testset_result
})
 
result.to_csv("./output.csv", columns=["PERID", "Criminal"])
