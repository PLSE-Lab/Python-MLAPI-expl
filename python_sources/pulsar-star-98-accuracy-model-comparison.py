#!/usr/bin/env python
# coding: utf-8

# <h2>Predicting Pulsar Star with Different Models with Tuners</h2>
# 
# <h4>Essentially, a binary-classification problem</h4>
# 
# <p>
#     This kernel uses the simple Pulsar Star dataset to analyze and predict correct classification using 7 different models. All models yielded over 97% accuracy, with the highest accuracy attained was 98.07% - using Random Forest Search.     
# </p>
# 
# <h3>The models used:</h3>
#     <li>Keras simple dense neural network</li>
#     <li>Random Forest Search</li>
#     <li>Logistic Regression</li>
#     <li>Decision Tree Classifier</li>
#     <li>KNN</li>
#     <li>SVM</li>
#     <li>XGBoost</li>   
# 
# <h3>Tuners Used:</h3>
#     <li>Keras Tuner V1.0.0</li>
#     <li>SciKit Learn GridSearchCV</li>
#     
#     
# Also used AMD PlaidML Back-end for GPU Processing

# In[ ]:


get_ipython().system('pip install keras-tuner')


# In[ ]:


import cv2
import time
import keras
import pickle
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
import tensorflow as tf
from kerastuner.tuners import RandomSearch
from kerastuner.engine.hyperparameters import HyperParameter

df = pd.read_csv("../input/predicting-a-pulsar-star/pulsar_stars.csv")


# In[ ]:


X = df.drop('target_class',axis=1)
y = df['target_class']

#normalize full data
X = preprocessing.scale(X)

#split intro training and testing
x_train,x_test,y_train,y_test = train_test_split(X,y, test_size=0.2, random_state=50)

#Convert all data samples into numpy arrays
x_train = np.array(x_train) 
x_test = np.array(x_test)
y_train = np.array(y_train)
y_test = np.array(y_test)   


# <h4>
#     First let's start with the simple neural network.
# </h4>
# 
# <p>
#     Since this is a binary classification problem, we are going to keep the model simple. Just using Dense layers, let's use Keras Tuner to find the hyper-parameters for our model
# </p>

# In[ ]:


#Create Simple NN
def tune_model(hp):
    model = tf.keras.models.Sequential()
    
    model.add(tf.keras.layers.Dense(hp.Int('dense_units',
                                             min_value=32,
                                             max_value=256,
                                             step=32),
                                             activation='relu'))
    
        
    for i in range(hp.Int('n_layers', 1,4)):
        model.add(tf.keras.layers.Dense(hp.Int(f'dense_{i}_units',
                                             min_value=32,
                                             max_value=256,
                                             step=32),
                                             activation='relu'))
        
    model.add(tf.keras.layers.Dense(1, activation='sigmoid'))
    
    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    
    return model

tuner = RandomSearch(
    tune_model,
    objective='val_accuracy',
    max_trials=10,
    executions_per_trial=2)

tuner.search(
    x=x_train,
    y=y_train,
    epochs=3,
    batch_size=32,
    validation_data=(x_test,y_test)
    )
    
print(tuner.get_best_hyperparameters()[0].values)
print(tuner.get_best_models()[0].summary)


# Now with the output parameters let's create the actual model

# In[ ]:


def create_model():
    model = keras.models.Sequential([
            keras.layers.Dense(160, activation='relu'),
            keras.layers.Dense(224, activation='relu'),
            keras.layers.Dense(256, activation='relu'),
            keras.layers.Dense(96, activation='relu'),
            keras.layers.Dense(1,activation='sigmoid')
            ])
    
    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    
    return model

model = create_model()

history = model.fit(x_train,y_train,batch_size=32,epochs=30,verbose=True)
scores = model.evaluate(x_test,y_test,verbose=1)
print(scores[1])
# 0.9793


# Now, let's explore all the other models using Grid Search to optimize some parameters for highest accuracy
# 

# <h3>Random Forest Classifier</h3> 

# In[ ]:


from sklearn.ensemble import RandomForestClassifier #import the model library
rf = RandomForestClassifier()
parameters = {'n_estimators':[10,50,100,200],'max_depth':[2,4,6,8]}

clf = GridSearchCV(rf, parameters)
clf.fit(x_train,y_train)
sorted(clf.cv_results_.keys())
print(clf.best_params_)

rf = RandomForestClassifier(n_estimators=200, max_depth=8,random_state=0) # sitting model parameters
print("test accuracy: {} ".format(rf.fit(x_train, y_train).score(x_test, y_test))) # printing the results of fitting the model over the testing set
print("train accuracy: {} ".format(rf.fit(x_train, y_train).score(x_train, y_train))) # printing the results of fitting the model over the training set
# 0.9807


# <h3>Logistic Regression</h3>

# In[ ]:


from sklearn import linear_model #import the model library
logreg =linear_model.LogisticRegression()
parameters = {'solver':('lbfgs','sag','saga'),'max_iter':[100,250,500,750]}

clf = GridSearchCV(logreg,parameters)
clf.fit(x_train,y_train)
print(clf.best_params_)

logreg = linear_model.LogisticRegression(random_state = 42,max_iter= 250,solver='saga') # sitting model parameters
print("test accuracy: {} ".format(logreg.fit(x_train, y_train).score(x_test, y_test))) # printing the results of fitting the model over the testing set
print("train accuracy: {} ".format(logreg.fit(x_train, y_train).score(x_train, y_train))) # printing the results of fitting the model over the training set
#0.9801


# <h3>Decision Tree Classifier</h3> 

# In[ ]:


from sklearn import tree #import the model library
dt = tree.DecisionTreeClassifier() # sitting model
parameters = {'max_depth':[2,4,6,8]}

clf = GridSearchCV(dt,parameters)
clf.fit(x_train,y_train)
print(clf.best_params_)

dt = tree.DecisionTreeClassifier(max_depth=4)
print("test accuracy: {} ".format(dt.fit(x_train, y_train).score(x_test, y_test))) # printing the results of fitting the model over the testing set
print("train accuracy: {} ".format(dt.fit(x_train, y_train).score(x_train, y_train))) # printing the results of fitting the model over the training set

#0.9779


# <h3>KNN</h3>

# In[ ]:


from sklearn.neighbors import KNeighborsClassifier #import the model library
knn = KNeighborsClassifier()
parameters = {'n_neighbors':[1,3,5,7]}

clf = GridSearchCV(knn,parameters)
clf.fit(x_train,y_train)
print(clf.best_params_)

neigh = KNeighborsClassifier(n_neighbors=7) # sitting model parameters
print("test accuracy: {} ".format(neigh.fit(x_train, y_train).score(x_test, y_test))) # printing the results of fitting the model over the testing set
print("train accuracy: {} ".format(neigh.fit(x_train, y_train).score(x_train, y_train))) # printing the results of fitting the model over the training set

#0.9793


# <h3></h3>

# <h3>XGBoost</h3>

# In[ ]:


from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
model = XGBClassifier()
parameters = {'max_depth':[2,4,6,8,10],'learning_rate':[0.1,0.001,0.0001],'n_estimators':[50,100,150,200,250,300,350,400]}

clf = GridSearchCV(model,parameters)
clf.fit(x_train,y_train)
print(clf.best_params_)

model = XGBClassifier(learning_rate=0.1,max_depth=4,n_estimators=250)
print("test accuracy: {} ".format(model.fit(x_train, y_train).score(x_test, y_test))) # printing the results of fitting the model over the testing set
print("train accuracy: {} ".format(model.fit(x_train, y_train).score(x_train, y_train))) # printing the results of fitting the model over the training set

#0.9798

