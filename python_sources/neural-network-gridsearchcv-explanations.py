#!/usr/bin/env python
# coding: utf-8

# # Importing Dataset

# In[ ]:




import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
dataset = pd.read_csv("../input/insurance/insurance.csv")
print(dataset.head())


# Splitting dataset into independant variables (age, sex, etc) and dependant variables (insurance costs).

# In[ ]:


X = dataset.iloc[:, 0:6].values
y = dataset.iloc[:, 6:7].values


# In[ ]:


print("Age     Gender  BMI     Kids    Smoker  Region    ")
s = [[str(e) for e in row] for row in X[0:5]]
lens = [max(map(len, col)) for col in zip(*s)]
fmt = '\t'.join('{{:{}}}'.format(x) for x in lens)
table = [fmt.format(*row) for row in s]
print ('\n'.join(table))

print("\n")

print("Charge")
s = [[str(e) for e in row] for row in y[0:5]]
lens = [max(map(len, col)) for col in zip(*s)]
fmt = '\t'.join('{{:{}}}'.format(x) for x in lens)
table = [fmt.format(*row) for row in s]
print ('\n'.join(table))


# # Data Preprocessing

# A neural network (and all other algorithms) require numbers/data, and cannot accept strings and words as it will break the network. Therefore, any of the dependant variables that include words will be replaced with numbers.

# In[ ]:


from sklearn.preprocessing import LabelEncoder

labelencoder_X_1 = LabelEncoder()
X[:, 1] = labelencoder_X_1.fit_transform(X[:, 1])
labelencoder_X_4 = LabelEncoder()
X[:, 4] = labelencoder_X_4.fit_transform(X[:, 4])


# In[ ]:


shortenedX = X[0:10]
print("Age     Gender  BMI     Kids    Smoker  Region   ")
s = [[str(e) for e in row] for row in shortenedX]
lens = [max(map(len, col)) for col in zip(*s)]
fmt = '\t'.join('{{:{}}}'.format(x) for x in lens)
table = [fmt.format(*row) for row in s]
print ('\n'.join(table))


# As you can see, we didn't encode region. That is because we cannot tranform it into a range of values. For example, if we encode it and southwest becomes 0 and southeast becomes 1, to the network it seems that southeast (1) is 'more' than southeast (0). We cannot quantify location; we have to represent it in a way that will not give any bias to the neural network for it to seem that one network is more important than another. This can be done by using dummy variables, and having each location be its own variable. Furthermore, we only need 3 locational variables beause if a person is not in 3 locations, we know it belongs in the last location. We must always remove at least one variable when having dummy variables to avoid the dummy variable trap. 

# In[ ]:


from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

columnTransformer = ColumnTransformer([('encoder', OneHotEncoder(), [5])], remainder='passthrough')
X = np.array(columnTransformer.fit_transform(X), dtype = np.str)
X = X[:, 1:]


# In[ ]:


shortenedX = X[0:10]
print("NW      SE      SW      Age     Gender  BMI     Kids    Smoker     ")
s = [[str(e) for e in row] for row in shortenedX]
lens = [max(map(len, col)) for col in zip(*s)]
fmt = '\t'.join('{{:{}}}'.format(x) for x in lens)
table = [fmt.format(*row) for row in s]
print ('\n'.join(table))


# In the neural network we require training data and testing data. A standard split is used: 80/20. 

# In[ ]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)


# In our data we have some datapoints that have a range from 0 to 1 (gender/smoker/region) and other having large ranges of values (age/bmi/chlidren). Because some values in the table are much larger, they will have a much larger impact on the network. To remove this bias, all the values will be scaled to the same range (in this case 0-1). Since this is done to independant variables, it will also be done to the dependant variables. 

# In[ ]:


from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler((0,1))
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

y_train = sc.fit_transform(y_train)
y_test = sc.transform(y_test)


# In[ ]:


shortenedX_train = X_train[0:5]
print("NW      SE      SW      Age                     Gender  BMI                     Kids    Smoker     ")
s = [[str(e) for e in row] for row in shortenedX_train]
lens = [max(map(len, col)) for col in zip(*s)]
fmt = '\t'.join('{{:{}}}'.format(x) for x in lens)
table = [fmt.format(*row) for row in s]
print ('\n'.join(table))


# In[ ]:


shortenedy_train = y_train[0:5]
print("Charge")
s = [[str(e) for e in row] for row in shortenedy_train]
lens = [max(map(len, col)) for col in zip(*s)]
fmt = '\t'.join('{{:{}}}'.format(x) for x in lens)
table = [fmt.format(*row) for row in s]
print ('\n'.join(table))


# # Creating the Neural Network

# We will now be creating our neural network model. The first thing done is import libraries and initializing the classifier/model. Then, each layer of the network is added with the specified number of nuerons (units) and the activation function used. It is standard to use rectified linear unit for all hidden layers. After the architecture is added, the network is compiled. Since this is regression problem, the loss function and metrics (scoring method) will relate to distance from the networks guess to the actual value. 

# In[ ]:


import keras
from keras.models import Sequential
from keras.layers import Dense

classifier = Sequential()
    
# Adding the input layer and the first hidden layer
classifier.add(Dense(units = 128, activation = 'relu'))
    
# Adding the second hidden layer
classifier.add(Dense(units = 64, activation = 'relu'))
    
classifier.add(Dense(units = 32, activation = 'relu'))
    
# Adding the output layer
classifier.add(Dense(units = 1, activation = 'linear'))
    
# Compiling the ANN
classifier.compile(loss='mean_absolute_error', optimizer='adam', metrics=['mean_absolute_error'])
    


# # Evaluating the Neural Network

# Since the classifier is built, we can now train the classifier on the data. I have arbitraliy chosen a batch_size and epoch number without any real rhyme or reason, but I will show how to choose the best batch size, epoch, and optimizer later on. The .fit() function returns an object that has many attributes related to the network (loss, scoring, etc). This allows us gain information on how the model performed. 

# In[ ]:


History = classifier.fit(x = X_train, y = y_train, batch_size = 128, epochs = 150, verbose = 0)


# Using the history object, the mean absolute error was graphed over the epochs, and we can see how the model's performance increased as it passed through each epoch. 

# In[ ]:


plt.plot(History.history['mean_absolute_error'])
plt.title('Loss Function Over Epochs')
plt.ylabel('MAE value')
plt.xlabel('No. epoch')
plt.show()


# We can also use the .predict() method to see how close the predictions of the neural network come to the actual charges. One thing to remember is that y_train and the outputs of the classifier are all scaled using the MinMaxScaler() object. To compare values with this transformation would be hard; therefore, we can inverse the transformation to intuitively see how well our model did.  

# In[ ]:


y_pred = classifier.predict(X_test)

y_predInverse = sc.inverse_transform(y_pred)
y_testInverse = sc.inverse_transform(y_test)


# In[ ]:


combinedArray = np.column_stack((y_testInverse[0:10],y_predInverse[0:10]))
print("Actual Charge   Predicted Charge")
s = [[str(e) for e in row] for row in np.around(combinedArray, 2)]
lens = [max(map(len, col)) for col in zip(*s)]
fmt = '\t'.join('{{:{}}}'.format(x) for x in lens)
table = [fmt.format(*row) for row in s]
print ('\n'.join(table))


# # Improving the Neural Network using GridSearchCV

# In order to use GridSearchCV easily, we need to create a function that would assemble the achitecture of the neural network. Nothing much is new, except including an optimizer parameter as it is a parameter we might want to tune/change, and it is the only hyperparameter that has to be changed in the architecture of the network. 

# In[ ]:


def buildModel(optimizer):
    # Initialising the ANN
    classifier = Sequential()
    
    # Adding the input layer and the first hidden layer
    classifier.add(Dense(units = 128, activation = 'relu'))
    
    # Adding the second hidden layer
    classifier.add(Dense(units = 64, activation = 'relu'))
    
    
    classifier.add(Dense(units = 32, activation = 'relu'))
    
    # Adding the output layer
    classifier.add(Dense(units = 1, activation = 'linear'))
    
    # Compiling the ANN
    classifier.compile(loss='mean_absolute_error', optimizer=optimizer, metrics=['mean_absolute_error'])
    
    return classifier


# Now is where we use GridSearchCV. First what we want to do is create a KerasRegressor classifier which will be used in GridSearch. Next, we will create a dictionary or list of parameters we want to tune and what values we want to tune it to. I put 4 options for batch_size, two options for epochs, and two different optimizer functions. Then, we will create an object of the GridSearchCV that is callibrated to look for the best parameters. After that, we are simply having that object run and find the best parameters, which will return itself with some useful information for us to find the best parameters.   

# In[ ]:


from sklearn.model_selection import GridSearchCV 
from tensorflow.keras.wrappers.scikit_learn import KerasRegressor

classifier = KerasRegressor(build_fn = buildModel)
#What hyperparameter we want to play with
parameters = {'batch_size': [16, 32, 64, 128],
              'epochs': [100, 150],
              'optimizer': ['adam', 'rmsprop']}
grid_search = GridSearchCV(estimator = classifier,
                           param_grid = parameters,
                           scoring = 'neg_mean_absolute_error',
                           cv = 5)
grid_search = grid_search.fit(X_train, y_train, verbose = 0)


# Grid Search has many attributes but the one below shows the best parameters that the object found. This allows us to plug in the best numbers (out of the ones we put in the dictionary) that will result in the best model. 

# In[ ]:


best_parameters = grid_search.best_params_
best_score = grid_search.best_score_

print("Best Parameters: " + str(best_parameters))


# In[ ]:


bestClassifier = buildModel('adam')
HistoryBest = bestClassifier.fit(x = X_train, y = y_train, batch_size = 16, epochs =150 , verbose = 0)
plt.plot(History.history['mean_absolute_error'], label='Initial Parameters')
plt.plot(HistoryBest.history['mean_absolute_error'], label='GridSearchCV Best Parameters')
plt.title('Loss Function Over Epochs')
plt.ylabel('MAE value')
plt.xlabel('No. epoch')
plt.legend(loc="upper right")
plt.show()


# As seen from the graph above, the loss is lower for the classifier with the ideal parameters. Another to check the performance of the model is to check how far off the model is from its predictions to the actual values, and as we can see the model with the best parameters has a lower error. 

# In[ ]:


from sklearn.metrics import mean_absolute_error 

print("Initial Classifier MAE: " + str(mean_absolute_error(y_test, y_pred, sample_weight=None, multioutput='uniform_average')))
print("Best Classifier MAE: " + str(mean_absolute_error(y_test, bestClassifier.predict(X_test), sample_weight=None, multioutput='uniform_average')))


# We previously saw the predicted values of the inital neural network (with arbitrarily chosen hyperparameters) and they were decently close. However, the new classifier is much more precise and accurate as you can see from a small sample of the predicted values and the actual charge. 

# In[ ]:


y_predBestInverse = sc.inverse_transform(bestClassifier.predict(X_test))

combinedArray = np.column_stack((y_testInverse[0:10],y_predInverse[0:10], y_predBestInverse[0:10]))
print("Actual Charge   Initial         Best ")
s = [[str(e) for e in row] for row in np.around(combinedArray, 2)]
lens = [max(map(len, col)) for col in zip(*s)]
fmt = '\t'.join('{{:{}}}'.format(x) for x in lens)
table = [fmt.format(*row) for row in s]
print ('\n'.join(table))

