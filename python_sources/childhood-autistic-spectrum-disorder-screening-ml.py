#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# import warnings filter
from warnings import simplefilter
# ignore all future warnings
simplefilter(action='ignore', category=FutureWarning)

import sys
import pandas as pd
import numpy as np
import sklearn
import matplotlib
import keras

print('Python: {}'.format(sys.version))
print('Numpy: {}'.format(np.__version__))
print('Sklearn: {}'.format(sklearn.__version__))
print('Pandas: {}'.format(pd.__version__))
print('Matplotlib: {}'.format(matplotlib.__version__))
print('Keras: {}'.format(keras.__version__))


# # 1. Importing the Dataset
# 
# We will obtain the data from the UCI Machine Learning Repository; however, since the data isn't contained in a csv or txt file, we will have to download the compressed zip file and then extract the data manually. Once that is accomplished, we will read the information in from a text file using Pandas.

# In[ ]:


# import the dataset

names = ['A1_Score',
        'A2_Score',
        'A3_Score',
        'A4_Score',
        'A5_Score',
        'A6_Score',
        'A7_Score',
        'A8_Score',
        'A9_Score',
        'A10_Score',
        'age',
        'gender',
        'ethnicity',
        'jundice',
        'family_history_of_PDD',
        'contry_of_res',
        'used_app_before',
        'result',
        'age_desc',
        'relation',
        'class']

# read the file
data = pd.read_table('../input/Autism-Child-Data_modified.txt', sep = ',', names = names)


# In[ ]:


# print the shape of the DataFrame, so we can see how many examples we have
print('Shape of DataFrame: {}'.format(data.shape))
print(data.loc[0])


# In[ ]:


# print out multiple patients at the same time
data.loc[:10]


# In[ ]:


# print out a description of the dataframe
data.describe()


# # 2. Data Preprocessing
# 
# This dataset is going to require multiple preprocessing steps. First, we have columns in our DataFrame (attributes) that we don't want to use when training our neural network. We will drop these columns first. Secondly, much of our data is reported using strings; as a result, we will convert our data to categorical labels. During our preprocessing, we will also split the dataset into X and Y datasets, where X has all of the attributes we want to use for prediction and Y has the class labels.

# In[ ]:


# drop unwanted columns
data = data.drop(['result', 'age_desc'], axis=1)


# In[ ]:


data.loc[:10]


# In[ ]:


# create X and Y datasets for training
x = data.drop(['class'], 1)
y = data['class']


# In[ ]:


x.loc[:10]


# In[ ]:


# convert the data to categorical values - one-hot-encoded vectors
X = pd.get_dummies(x)


# In[ ]:


# print the new categorical column labels
X.columns.values


# In[ ]:


# print an example patient from the categorical data
X.loc[1]


# In[ ]:


# convert the class data to categorical values - one-hot-encoded vectors
Y = pd.get_dummies(y)


# In[ ]:


Y.iloc[:10]


# # 3. Split the Dataset into Training and Testing Datasets
# 
# Before we can begin training our neural network, we need to split the dataset into training and testing datasets. This will allow us to test our network after we are done training to determine how well it will generalize to new data. This step is incredibly easy when using the train_test_split() function provided by scikit-learn!

# In[ ]:


from sklearn import model_selection
# split the X and Y data into training and testing datasets
X_train, X_test, Y_train, Y_test = model_selection.train_test_split(X, Y, test_size = 0.2)


# In[ ]:


print(X_train.shape)
print(X_test.shape)
print(Y_train.shape)
print(Y_test.shape)


# # 4. Building the Network - Keras
# 
# In this project, we are going to use Keras to build and train our network. This model will be relatively simple and will only use dense (also known as fully connected) layers. This is the most common neural network layer. The network will have one hidden layer, use an Adam optimizer, and a categorical crossentropy loss. We won't worry about optimizing parameters such as learning rate, number of neurons in each layer, or activation functions in this project; however, if you have the time, manually adjusting these parameters and observing the results is a great way to learn about their function!

# In[ ]:


# build a neural network using Keras
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam

# define a function to build the keras model
def create_model():
    # create model
    model = Sequential()
    model.add(Dense(8, input_dim=96, kernel_initializer='normal', activation='relu'))
    model.add(Dense(4, kernel_initializer='normal', activation='relu'))
    model.add(Dense(2, activation='sigmoid'))
    
    # compile model
    adam = Adam(lr=0.001)
    model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])
    return model

model = create_model()

print(model.summary())


# # 5. Training the Network
# 
# Now it's time for the fun! Training a Keras model is as simple as calling model.fit().

# In[ ]:


# fit the model to the training data
model.fit(X_train, Y_train, epochs=50, batch_size=10, verbose = 1)


# # 6. Testing and Performance Metrics
# 
# Now that our model has been trained, we need to test its performance on the testing dataset. The model has never seen this information before; as a result, the testing dataset allows us to determine whether or not the model will be able to generalize to information that wasn't used during its training phase. We will use some of the metrics provided by scikit-learn for this purpose!

# In[ ]:


# generate classification report using predictions for categorical model
from sklearn.metrics import classification_report, accuracy_score

predictions = model.predict_classes(X_test)
predictions


# In[ ]:


print('Results for Categorical Model')
print(accuracy_score(Y_test[['YES']], predictions))
print(classification_report(Y_test[['YES']], predictions))

