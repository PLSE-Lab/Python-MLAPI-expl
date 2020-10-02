"""
@author: vipulsrivastav
"""
""" Problem Description

And here we've got a snapshot of bank which  measured some things about their customers.
Bank has been seeing unusual churn rates, churn is when people leave the company and they've
seen customers leaving at unusually high rates and they want to understand what the problem is and they
want to assess and address that problem.
We've got to take a sample of our customers by the way this is a sample that 10000 is a tiny number
for this bank. This fictional bank operates in Europe in three countries France Spain and Germany 

The_pred can be used on the test set on the customers of the bank.
And by ranking the probabilities from the highest to the lowest it 
gets a ranking of the customers most likely to leave the bank.

So then for example the bank to have a look at the 10 percent highest probabilities of their customers
to leave the bank and so make it a segment and then analyzed in more depth thanks to data mining techniques
to understand why the customers of this segment are the most likely to leave the bank.

"""

#importing libraries

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

file = '../input/Churn_Modelling.csv'

#importing the dataset
dataset = pd.read_csv(file)
X = dataset.iloc[:,3:13].values
y = dataset.iloc[:,13].values

#handle missing values using impute transform, if needed

#encode the categorial independent variables

# since we have 2 categorical variables (country, gender), we
# need to have 2 label encoder objects

# When to use Label Encoder vs One hot Encoder

# https://datascience.stackexchange.com/questions/9443/when-to-use-one-hot-encoding-vs-labelencoder-vs-dictvectorizor


from sklearn.preprocessing import LabelEncoder , OneHotEncoder
label_encoder_x_country = LabelEncoder() 
label_encoder_x_gender = LabelEncoder()

one_hot_encoder_x_country = OneHotEncoder()

X[:,1] = label_encoder_x_country.fit_transform(X[:,1])
X[:,2] = label_encoder_x_gender.fit_transform(X[:,2])

one_hot_encoder_x_country = OneHotEncoder(categorical_features =[1])
X = one_hot_encoder_x_country.fit_transform(X).toarray()
#print X to see the changes!

X = X[:,1:] # handling dummy variables
#https://www.moresteam.com/WhitePapers/download/dummy-variables.pdf

#train test split

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.25, random_state = 0)

#Feature Scaling!
# https://medium.com/greyatom/why-how-and-when-to-scale-your-features-4b30ab09db5e

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.fit_transform(X_test)


#importing keras
import keras

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout

#rectifier fxn for hidden layers and sigmoid for output layer based on forums
#Improving the ANN using dropout to avoid overfitting
#Initializing the model

model = Sequential()
# relies on experiments to choose the best parameters, right now
# adding the input layer and the first hidden layer 
#11 independent variables
model.add(Dense(activation="relu", units=6, kernel_initializer="uniform",input_dim=11)) 
model.add(Dropout(0.1))

#second hidden layer
model.add(Dense(activation="relu", units=6, kernel_initializer="uniform"))
model.add(Dropout(0.1))

#final output layer
# units ~ no of categories
# softmax activation when there are more than 2 categories
model.add(Dense(activation="sigmoid", units=1, kernel_initializer="uniform")) 

#compile
#optimizer to use the algo to find optimal set of weights, categorical/binary crossentropy
model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

#fit model
model.fit(X_train, y_train,batch_size=10, nb_epoch=100)

#Evaluating the ANN using k folds cross validation
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
from keras.models import Sequential
from keras.layers import Dense
def build_classifier():
    model = Sequential()
    model.add(Dense(activation="relu", units=6, kernel_initializer="uniform",input_dim=11)) 
    model.add(Dense(activation="relu", units=6, kernel_initializer="uniform"))
    model.add(Dense(activation="sigmoid", units=1, kernel_initializer="uniform")) 
    model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
    
    return model

model = KerasClassifier(build_fn = build_classifier,batch_size=10, nb_epoch=100)

# accuracies = cross_val_score(estimator = model, X= X_train, y = y_train, cv =10, n_jobs=-1) for GPU
accuracies = cross_val_score(estimator = model, X= X_train, y = y_train, cv =10)

print("Model mean :", accuracies.mean())
print("Model variance :", accuracies.std())

###Trying to Find best Hyperparameters (batch size, optimizer, epoch ) using GridSearch CV

## Tuning the ANN

from sklearn.model_selection import GridSearchCV

def build_classifier(optimizer):
    model = Sequential()
    model.add(Dense(activation="relu", units=6, kernel_initializer="uniform",input_dim=11)) 
    model.add(Dense(activation="relu", units=6, kernel_initializer="uniform"))
    model.add(Dense(activation="sigmoid", units=1, kernel_initializer="uniform")) 
    model.compile(optimizer = optimizer, loss = 'binary_crossentropy', metrics = ['accuracy'])
    
    return model

model = KerasClassifier(build_fn = build_classifier)
hyperparam = {'batch_size' :[25,32],
              'nb_epoch'   :[100,300],
              'optimizer': ['adam','rmsprop']
              }

grid_search = GridSearchCV(estimator = model,
                           param_grid = hyperparam,
                           scoring = 'accuracy',
                           cv =10)

grid_search = grid_search.fit(X_train, y_train)
best_params = grid_search.best_params_
best_accuracy = grid_search.best_score_

print(best_params)




 