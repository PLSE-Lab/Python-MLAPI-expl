#!/usr/bin/env python
# coding: utf-8

# # Introduction to ANN with Digit Recognizer Challenge  
# 
# *5 November 2018*  
# 
# #### ***[Soumya Ranjan Behera](https://www.linkedin.com/in/soumya044)***
# 
# ### In this Kernel I have demonstrated a beginner's approach of implementing an  Artificial Neural Network (ANN) to classify the digits into their respective categories which have scored **96.7 %** Accuracy in the Digit Recognizer Competition.
# 
# ### Goals of this Kenel:  
# * To provide a basic implementation of Artificial Neural Network (ANN)
# * To show a path for approaching Convolutional Neural Network (CNN)
# * A beginner friendly kernel to show a procedure to compete in Kaggle Digit Recognizer Competition

# **Dataset:**  
# * ../input/  
#        |_ train.csv  
#        |_ test.csv  
#        |_ sample_submission.csv

# # 1. Prepare our Data 

# ### Import Numpy, Pandas and matplotlib libraries

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# ### Import Training data as Numpy array

# In[ ]:


dataset = pd.read_csv('../input/train.csv')
X = dataset.iloc[:, 1:].values
y = dataset.iloc[:, 0].values


# ### Let's see the distribution of data

# In[ ]:


import seaborn as sns
sns.countplot(y)


# Since all our target classes are well-balanced we can move to our next step.

# ### Check for Null Values

# In[ ]:


# Check IF some Feature variables are NaN
np.unique(np.isnan(X))[0]


# In[ ]:


# Check IF some Target Variables are NaN
np.unique(np.isnan(y))[0]


# Since no NULL or NaN values are there, we're good to go!

# ### Splitting Dataset into Training set and Test set

# In[ ]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.1, random_state = 42)


# ### Feature Scaling
# This step is to transform the data such that its distribution will have a mean value 0 and standard deviation of 1.  
# Given the distribution of the data, each value in the dataset will have the sample mean value subtracted, and then divided by the standard deviation of the whole dataset.

# In[ ]:


# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


# ### Encoding Categorical Data into Continuous Variable

# In[ ]:


from keras.utils import np_utils
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)


# # 2. Building our ANN

# ### Importing the Keras libraries and packages

# In[ ]:


# Importing the Keras libraries and packages
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout


# ### Initialising the ANN Model

# In[ ]:


# Initialising the ANN
classifier = Sequential()


# ### Adding Input Layer and Hidden Layers

# In[ ]:


# Adding the input layer and the first hidden layer
classifier.add(Dense(units = 784, activation = 'relu', input_dim = 784))

# Adding the second hidden layer
classifier.add(Dense(units = 128, activation = 'relu'))
classifier.add(Dense(units = 128, activation = 'relu'))
classifier.add(Dropout(0.2))

classifier.add(Dense(units = 64, activation = 'relu'))
classifier.add(Dense(units = 64, activation = 'relu'))
classifier.add(Dropout(0.2))

classifier.add(Dense(units = 32, activation = 'relu'))
classifier.add(Dense(units = 32, activation = 'relu'))
classifier.add(Dropout(0.05))


# ### Adding Output Layer

# In[ ]:


# Adding the output layer
classifier.add(Dense(units = 10, activation = 'softmax'))


# ### Compiling our Model

# In[ ]:


# Compiling the ANN
classifier.compile(optimizer = 'sgd', loss = 'categorical_crossentropy', metrics = ['accuracy'])


# # 3. Training our ANN

# ### Fitting the ANN to the Training set

# In[ ]:


# Fitting the ANN to the Training set
history = classifier.fit(X_train, y_train, 
                         validation_data = (X_test, y_test), 
                         batch_size = 28, 
                         epochs = 25)

# Plot training & validation accuracy values
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

# Plot training & validation loss values
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()


# # 4. Perfomance Evaluation

# ### Accuracy of Model

# In[ ]:


model_acc = classifier.evaluate(X_test, y_test)
print(" Model Accuracy is : {0:.1f}%".format(model_acc[1]*100))


# # 5. Prepare for Final Submission

# ### Import the Test Data

# In[ ]:


test_dataset = pd.read_csv('../input/test.csv')
test = test_dataset.iloc[:,:].values


# ### Make Predictions for ' test ' data

# In[ ]:


# Prediction
test_pred = classifier.predict(test)

# Mark probability score > 0.5 as Predicted Label, axis = 1 means insert column-wise 
results = test_pred.argmax(axis=1)


# ### Visualize some test results

# In[ ]:


for i in range(1,10):
    index = np.random.randint(1,28001)
    plt.subplot(3,3,i)
    plt.imshow(test[index].reshape(28,28))
    plt.title("Predicted Label : {}".format(results[index]))
plt.subplots_adjust(hspace = 1.2, wspace = 1.2)
plt.show()
    


# ### Convert Numpy Array format to Pandas Series and then to CSV format

# In[ ]:


results = pd.Series(results,name="Label")
submission = pd.concat([pd.Series(range(1,28001),name = "ImageId"),results],axis = 1)
submission.to_csv("submission.csv",index=False)


# ### Check Submission file for correct format

# In[ ]:


submission.head()


# # Thank You  
# 
# If you liked this kernel please **Upvote**. Don't forget to drop a comment or suggestion.  
# 
# ### *Soumya Ranjan Behera*
# Let's stay Connected! [LinkedIn](https://www.linkedin.com/in/soumya044)  
# 
# **Happy Coding !**
