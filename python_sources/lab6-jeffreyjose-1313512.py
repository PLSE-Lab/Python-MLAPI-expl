#!/usr/bin/env python
# coding: utf-8

# *Name: Jeffrey Jose* <br />
# *ID: 1313512*

# In[30]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from matplotlib import pyplot as plt
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
print(os.listdir("../input"))

#Ignore warnings
import warnings
warnings.filterwarnings("ignore")
# Any results you write to the current directory are saved as output.


# ## Preparing The Data

# In[31]:


from sklearn.preprocessing import OneHotEncoder

#Read the data and replace any missing values
data = pd.read_csv("../input/housing.csv")
data = data.fillna(0)
data.dtypes #ocean_proximity is an object datatype
#Using one-hot encoding, turn categorical values into numerical
#encode labels with value between 0 and n_classes-1
df_cat = data[['ocean_proximity']]
cat_encoder = OneHotEncoder()
df_cat_1hot = cat_encoder.fit_transform(df_cat)
print(df_cat_1hot.toarray())

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer

print("")
print("The pipeline: ")
print("")
data1 = data.drop('median_house_value', axis=1)
num_attrs = list(data1)
num_attrs.remove("ocean_proximity")
cat_attrs = ["ocean_proximity"]
full_pipeline = ColumnTransformer([("num", SimpleImputer(strategy = 'median'), num_attrs),("cat", OneHotEncoder(), cat_attrs)])
x = full_pipeline.fit_transform(data1)
print(x)


# ### Splitting The Dataset Into Train, Validation And Test

# In[32]:


from sklearn.model_selection import train_test_split

#x is all the columns except for ocean proximity
#x = data.iloc[:, 0:9] 
#I have set y to the ocean_proximity
y = data.iloc[:,9]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 1313512)
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size = 0.15/0.8, random_state = 1313512)

#Ensuring that the size of the train, validation and test are what it should roughly be
print("Train Size: " + str(x_train.shape), str(y_train.shape))
print("Validation Size: " + str(x_val.shape), str(y_val.shape))
print("Test Size: " + str(x_test.shape), str(y_test.shape))


# ### Scaling The Data Using StandardScaler

# In[33]:


from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
#Fit the standard scale to the train dataset
scaler.fit(x_train)
#applying scaling to all the train, validation and test data
x_train_scaled = scaler.transform(x_train)
#transform the x validation and the y validation
x_val_scaled = scaler.transform(x_val)
#transform the x_test and the y_test
x_test_scaled = scaler.transform(x_test)


# ### Creating The Neural Network Using The Sequential Model

# In[34]:


from keras.models import Sequential
from keras.layers import Dense
import keras

averageValues = []
#function to create the neural network for a specified number of layers and size of those layers
def sequentialnetwork(n_layers, n1, check = False):
    cvscores = []
    valMAE = []
    model = Sequential()
    if n_layers == 1:
        model.add(Dense(n1, input_dim = 13, activation = 'relu'))
        model.add(Dense(1))
    elif n_layers == 2:
        model.add(Dense(n1, input_dim = 13, activation = 'relu'))
        model.add(Dense(1))
    elif n_layers == 4:
        model.add(Dense(n1, input_dim = 13, activation = 'relu'))
        for i in range(2):
            model.add(Dense(n1, activation = 'relu'))
        model.add(Dense(1))
    else: #if the number of layers is 8
        model.add(Dense(n1, input_dim = 13, activation = 'relu'))
        for i in range(6):
            model.add(Dense(n1, activation = 'relu'))
        model.add(Dense(1))
    #provides a summary of the model   
    model.summary() 
    
    model.compile(loss = 'mse', optimizer = 'adam', metrics = ['mae'])
    modelFit = model.fit(x_train_scaled, y_train, validation_data = (x_val_scaled, y_val), epochs = 30)
    print(modelFit)
    
    valMAE.append(modelFit.history['val_mean_absolute_error'])
    #Accuracy of the model and overall accuracy
    scores = model.evaluate(x_test_scaled, y_test, verbose = 0)
    cvscores.append(scores[1] * 100)
    print("Accuracy: " + "%.2f%% (+/- %.2f%%)" % (np.mean(cvscores), np.std(cvscores)))
    
    #Evaluation of the current model based on the x_test and the y_test
    modelEvaluation = model.evaluate(x_test_scaled, y_test)
    
    #Plotting the graph for mae over epoch
    plt.plot(modelFit.history['val_mean_absolute_error'])
    plt.title('Model MAE Over Epoch')
    plt.ylabel('Mean_Absolute_Error')
    plt.xlabel('Epoch')
    plt.show()
    
    score = model.evaluate(x_val_scaled, y_val)
    print("\n%s: %.2f%%" % (model.metrics_names[1], score[1]*100))

    maeAverage = np.mean(valMAE)
    print("The average of the MAE values are: " + str(maeAverage))
    averageValues.append(maeAverage)
    
    #Conduct extra functions if its the best model
    if(check == True):
        count = 0
        totalCount = 0
        prediction = model.predict(x_test_scaled)
        for i in range(len(x_test_scaled)):
            totalCount += 1
            if prediction[i] < 15000 or prediction[i] > 500000:
                count += 1        
        print("The number of predictions that fall outside the 15000 and 50000 range is: " + str(count) + " out of " + str(totalCount))


# In[35]:


#trying different combinations of layer lengths and sizes and assessing the validation error of each of these
sequentialnetwork(8, 300, check = True)


# In[36]:


import operator
index, value = min(enumerate(averageValues), key = operator.itemgetter(1))
print("The minimum average MAE value is: " + str(value) + " and the index is: " + str(index))


# So the combination that gives out the best results are for a layer size of 300 and the number of layers is 8. <br />
# (8, 300)

# ## Part 1 Questions:

# Use the best network to predict the test set. What is the 'mae' for the test set? <br />
# *The MAE average value that we obtained from our best network is 39853.7005 (4.dp). This is an average of all the mean absoulte errors found on the validation data.* <br />
# How many test examples have predictions outside [15000, 500000]? <br />
# *From our results, 50 out of 4128 predictions fell outside the bounds. (Varies between runs). *<br />
# How much better is the test MAE than your best result from Lab3? <br />
# *The result we got from our lab 3 was 39817 using the polynomial features fit_transform. Under the current test, the result we got was 39853.70. This was worse but I believe that it could be due to the random states involved.* <br />
# <br />
# 
# **NOTE: THE NUMBERS THAT WE GOT FOR RESULTS DID VARY WITH DIFFERENT RUNS AND THEREFORE SOME OF THE NUMBERS USED IN THE QUESTIONS DO NOT COMPLY WITH THE OUTPUT SHOWN.**

# ## Part Two: Image Classification

# In[37]:


from keras.layers import Dense, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.optimizers import SGD
from keras.datasets import cifar10
from keras.layers import Dropout

def plotLosses(history):  
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.show()
    
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255.0
x_test /= 255.0
#x_train = x_train / 255
#x_test = x_test / 255
#Testing the shape of both the x_test and x_train
print('x_test shape: ', x_test.shape)
print('x_train shape: ', x_train.shape)

num_classes = 10
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)    
model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))
model.add(Conv2D(32, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
# Dropout layer added here
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(256, activation='relu'))
# Dropout layer added here
model.add(Dropout(0.25))
model.add(Dense(10, activation='softmax'))

model.summary()

sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer=sgd)

history = model.fit(x_train, y_train, batch_size=32, epochs=30, verbose=2, validation_split=0.15)

plotLosses(history)

score = model.evaluate(x_test, y_test, batch_size=128, verbose=0)
print(model.metrics_names)
print(score)


# The best learning rate I have identified in terms of accuracy is 0.01 with an accuracy of 67.75% compared to a learning rate of 0.1 with an accuracy of 10% and 0.0495 with an accuracy of 33.0%.

# In[38]:


#Predicting the test set
y_prediction = model.predict(x_test)
print(y_prediction)


# In[39]:


print( np.max(y_prediction[0]), np.argmax(y_prediction[0]), y_prediction[0])


# ### Worst Misclassification

# In[41]:


# For each of the ten classes identify the worst misclassification.
# "Worst" is defined as the max value for p_predicted-p_correct.
# Plot these then examples, each with a title including the
# correct label, the wrongly predicted label, and the value of
# p_predicted-p_correct.

def find_misclassified(c, y_test, y_prediction):
    worst_i = -1
    max = 0
    for i in range(len(y_prediction)):
        if y_test[i][c] == 1.0:
            if np.max(y_prediction[i]) > y_prediction[i][c]:#IF missclassified
                d = np.max(y_prediction[i]) - y_prediction[i][c]
                if d > max:
                    max = d
                    worst_i = i
    return worst_i, max

for j in range(10):
    idx, max = find_misclassified(j, y_test, y_prediction)
    display = x_test[idx].reshape(3, 32, 32)
    plt.imshow(x_test[idx:][0])
    plt.title(str(j) + " misclassified as " + str(np.argmax(y_prediction[idx])) + " [" + str(max) + "]")
    plt.show()


# In[ ]:




