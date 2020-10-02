# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

#reference : Deep Learning A-Z™: Hands-On Artificial Neural Networks
#--------PROBLEM STATEMENT --------------
'''
Given a Customer Bank Churn Dataset 
Predict whether a customer will exit the bank or Not
'''
      
#Any results you write to the current directory are saved as output.

#load data using pandas 
churn_data = pd.read_csv('/kaggle/input/predicting-churn-for-bank-customers/Churn_Modelling.csv')
churn_data.head()

# remove fields that are not required
# row number , customer id , surname fields are not required as they are not giving any relevant information
# keep features as X , put that as y
X = churn_data.iloc[:,3:-1]
y = churn_data.iloc[:,-1]

#there are some categorical fields like geography and gender 
#we have to encode them into dummy fields using one hot or label encoding
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
labelEncoder = LabelEncoder()
X.iloc[:,1] = labelEncoder.fit_transform(X.iloc[:,1])
X.iloc[:,2] = labelEncoder.fit_transform(X.iloc[:,2])

#we observe that geography is label encoded , however there is no order for geography
#for gender there is no need to one hot encode as there are only two categories
#hence we go for one hot encoding
one_hot_encoder = OneHotEncoder(categorical_features=[1])
X = one_hot_encoder.fit_transform(X).toarray()
print(X[0:5])
#remove one column from one hot encoded array
X = X[:,1:]
#since some of the fields are numeric - we have to standardize the fields to feed it to neural network
#feature scaling for columns 2,4,6,10
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X = sc.fit_transform(X)

#now divide the data into train and test sets 
from sklearn.model_selection import train_test_split
X_train , X_test , y_train , y_test = train_test_split(X,y , test_size= 0.2 , random_state=0)

#----Model building ----
#use keras sequential model
from keras.models import Sequential
from keras.layers import Dense

classifier = Sequential()
classifier.add(Dense(units=6, kernel_initializer='glorot_uniform', activation='relu' ,input_dim=11))
classifier.add(Dense(units=6, kernel_initializer='glorot_uniform', activation='relu'))
classifier.add(Dense(units=1, kernel_initializer='glorot_uniform', activation='sigmoid'))

#compile the neural network
classifier.compile(optimizer='adam',loss='binary_crossentropy', metrics=['accuracy'])

#fitting it to ANN
classifier.fit(X_train,y_train,batch_size=100, epochs=100)

#----- Making predictions and evaluating the model------
# Predicting the Test set results
y_pred = classifier.predict(X_test)
y_pred = (y_pred > 0.5)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)

# Predicting a single new observation
"""Predict if the customer with the following informations will leave the bank:
Geography: France
Credit Score: 600
Gender: Male
Age: 40
Tenure: 3
Balance: 60000
Number of Products: 2
Has Credit Card: Yes
Is Active Member: Yes
Estimated Salary: 50000"""

# France is 0 ,0 
# Gender male is 1
#hence data 
new_data = sc.transform(np.array([[0.0,0,600,1,40,3,60000,2,1,1,50000]]))
new_prediction = classifier.predict(new_data)
new_prediction = (new_prediction>0.5)
print('Predcition for the data : ' +str(new_prediction[0]))



# Evaluating the ANN
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
from keras.models import Sequential
from keras.layers import Dense
def build_classifier():
    classifier = Sequential()
    classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu', input_dim = 11))
    classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu'))
    classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))
    classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
    return classifier
classifier = KerasClassifier(build_fn = build_classifier, batch_size = 10, epochs = 100)
accuracies = cross_val_score(estimator = classifier, X = X_train, y = y_train, cv = 10, n_jobs = -1)
mean = accuracies.mean()
variance = accuracies.std()

#------Hyper Parameter Tuning -------------------
#The above neural network is created using some basic assumptions
#But we need to determine the best fit initial parameters like optimizer,batch size ,epochs etc
#So Hyperparameter tuning is required for it -Use a technique called GridSearchCV for it 

# Tuning the ANN
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV

def build_classifier(optimizer):
    classifier = Sequential()
    classifier.add(Dense(units = 6, kernel_initializer = 'glorot_uniform', activation = 'relu', input_dim = 11))
    classifier.add(Dense(units = 6, kernel_initializer = 'glorot_uniform', activation = 'relu'))
    classifier.add(Dense(units = 1, kernel_initializer = 'glorot_uniform', activation = 'sigmoid'))
    classifier.compile(optimizer = optimizer, loss = 'binary_crossentropy', metrics = ['accuracy'])
    return classifier

classifier = KerasClassifier(build_fn=build_classifier)

parameters = {'batch_size' : [25,32],
             'epochs' : [100,500],
             'optimizer' : ['adam','rmsprop']}


#using grid search with above parameters and k-cross validation as 10
grid_search = GridSearchCV(estimator=classifier,
                          param_grid=parameters,
                          scoring='accuracy',
                          cv=10)

grid_search = grid_search.fit(X_train,y_train)
best_parameters = grid_search.best_params_
best_accuracy = grid_search.best_score_

print(best_parameters)
print(best_accuracy)





