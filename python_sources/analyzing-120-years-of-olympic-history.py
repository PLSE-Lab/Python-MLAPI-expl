#!/usr/bin/env python
# coding: utf-8

# 1. DATA PRE-PROCESSING
# ----------------------------------

# **1.1 Importing required libraries:**

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
from keras.layers import Dense


# **1.2 Importing the dataset as dataFrame:**

# In[ ]:


dataset_athlet = pd.read_csv("../input/athlete_events.csv")
dataset_noc = pd.read_csv("../input/noc_regions.csv")
dataset_noc = dataset_noc.iloc[:,[0,1]]
dataset_final=pd.merge(dataset_athlet, dataset_noc, on = 'NOC',how = 'left')


# **1.3 Storing the independent variables (2=sex; 3=age; 4=height; 5=weight; 10=season; 12=Event; 15=region ) as 'X' and the dependent variable (14=medal won) as 'Y'**

# In[ ]:



X = dataset_final.iloc[:,[2,3,4,5,10,12,15]].values
Y = dataset_final.iloc[:,[14]].values


# **1.4 Taking care of missing (NAN) numerical data**

# In[ ]:


from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0)
imputer = imputer.fit(X[:, 1:4])
X[:,1:4] = imputer.transform(X[:, 1:4])


# **1.5 Taking care of missing (NAN) categorical data (region) in X**

# In[ ]:


for i in range(len(X[:,6])):
    if type(X[:,6][i]) != type('string'):
        X[:,6][i] = 'unknown'


# **1.6 Taking care of missing (NAN) categorical data (Medal won) in Y**

# In[ ]:


for i in range(len(Y[:,0])):
    if type(Y[:,0][i]) != type('asd'):
        Y[:,0][i] = 'No Medals'


# **1.7 Encoding categorical data in X**

# In[ ]:


labelencoder_X = LabelEncoder()
X[:,0] = labelencoder_X.fit_transform(X[:,0])
X[:,4] = labelencoder_X.fit_transform(X[:,4])
X[:,5] = labelencoder_X.fit_transform(X[:,5])
X[:,6] = labelencoder_X.fit_transform(X[:,6])
onehotencoder_X = OneHotEncoder(categorical_features = [4,5,6])
X = onehotencoder_X.fit_transform(X).toarray()


# **1.7 Encoding categorical data in Y**

# In[ ]:


labelencoder_Y = LabelEncoder()
Y[:,0] = labelencoder_Y.fit_transform(Y[:,0])
onehotencoder_Y = OneHotEncoder(categorical_features = [0])
Y = onehotencoder_Y.fit_transform(Y).toarray()


# **1.8 Splitting the dataset into training and  test set**

# In[ ]:


X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.25, random_state=0)


# **1.9 Performing feature scaling in training data and fitting the same in test data**

# In[ ]:


from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)


# 2. NOW LETS CREATE THE ARTIFICIAL NEURAL NETWORK TO PREDICT THE WINNERS IN OLYMPICS
# --------------------------------------------------------------------------------------------------------------------------------

# **2.1 Evaluating the model through K-fold cross validation:**

# In[ ]:


from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
def build_classifier():
    # Initialising the ANN
    classifier = Sequential()
    # Adding the input layer and the first hidden layer
    classifier.add(Dense(units = 180, kernel_initializer = 'uniform', activation = 'relu', input_dim = 278))
    # Adding the second hidden layer
    classifier.add(Dense(units = 180, kernel_initializer = 'uniform', activation = 'relu'))
    # Adding the output layer
    classifier.add(Dense(units = 4, kernel_initializer = 'uniform', activation = 'sigmoid'))
    # Compiling the ANN
    classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
    return classifier

classifier = KerasClassifier(build_fn = build_classifier, batch_size = 10, epochs = 100)
accuracies = cross_val_score(estimator = classifier, X = X_train, y=Y_train, cv = 5)


# **2.2 Calculating the score of our model prediction**

# In[ ]:


score = accuracies.mean()
variance = accuracies.std()

print('the score is:', score)
print('The variance is:', variance)

