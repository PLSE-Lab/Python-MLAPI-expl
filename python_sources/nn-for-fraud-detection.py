#!/usr/bin/env python
# coding: utf-8

# **Fraud Detection with Neural Network**
# 
# 

# # Neural Network for Fraud detection with SMOTE
# 
# The datasets contains transactions made by credit cards in September 2013 by european cardholders. This dataset presents transactions that occurred in two days, where we have 492 frauds out of 284,807 transactions. The dataset is highly unbalanced, the positive class (frauds) account for 0.172% of all transactions.
# 
# In this notebook for handle the imbalanced data I have used the undersample technique. 
# 
# I have also writed another notebook where handle this situation using [SMOTE](https://www.kaggle.com/davidevegliante/nn-for-fraud-detection-with-smote) technique.
# 
# **These are my first two notebooks, I hope to receive comment and advice for improve the understanding of the tools I used. **
# 
# 
# 
# 

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import os

# read the dataset and print five rows
original_dataset = pd.read_csv('../input/creditcard.csv')

dataset = original_dataset.copy()
print(dataset.head(5))


# Let's see how many example in our dataset we have. 

# In[ ]:


# count how many entry there are for every class
classes_count = pd.value_counts(dataset['Class'])

print("{} Non-fraud example\n{} Fraud examples".format(classes_count[0], classes_count[1]))

# classes_count is a Series. 
classes_count.plot(kind = 'bar')
plt.xlabel('Classes')
plt.ylabel('Frequencies')
plt.title('Fraud Class Hist')


# The classes of the dataset are not represented equally. 

# In[ ]:


# scale the amount feature
from sklearn.preprocessing import StandardScaler
amount_scaler = StandardScaler().fit(dataset[['Amount']])
dataset['AmountScaled'] = amount_scaler.transform(dataset[['Amount']])

# remove the old Amount Feature
dataset.drop(['Time', 'Amount'], axis = 1, inplace = True)

dataset.head(5)


# **  Undersampling with ratio 1**

# In[ ]:


X = dataset.loc[:, dataset.columns != 'Class' ]
y = dataset.loc[:, dataset.columns == 'Class' ]

from imblearn.under_sampling import RandomUnderSampler
rus = RandomUnderSampler(random_state = 0, sampling_strategy = 1.0)
X_resampled, y_resampled = rus.fit_resample(X, y)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size = 0.20, stratify = y_resampled)

assert len(y_train[y_train == 1]) + len(y_test[y_test == 1]) == len(dataset[dataset.Class == 1])
print("train_set size: {} - Class0: {}, Class1: {}".format( len(y_train), len(y_train[y_train == 0]), len(y_train[y_train == 1]) ))
print("test_set size: {} - Class0: {}, Class1: {}".format( len(y_test), len(y_test[y_test == 0]), len(y_test[y_test == 1]) ))


# **NN Structure**
# 

# In[ ]:


import tensorflow as tf
import keras 
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras import regularizers

# init ann
classifier = Sequential()

# in this case we used a rectifier activation function for the hidden layers
# and a sigmoid function for the output layer
classifier.add(Dense(
    input_dim = len(X.columns), # input neurons
    units = 15, # first hidden layer
    kernel_initializer = 'he_normal', 
    bias_initializer = 'zeros',
    activation = 'relu', # activation function (rectifier)
    kernel_regularizer=regularizers.l2(0.006)
))
# add a new hidden layer with the same number of neurons
classifier.add(Dense(
    units = 6, # second hidden layer
    kernel_initializer = 'he_normal',
    bias_initializer = 'zeros',
    activation = 'relu', # activation function (rectifier)
    kernel_regularizer=regularizers.l2(0.006)
))
# add the output layer with one neuron
classifier.add(Dense(
    units = 1, # output layer
    kernel_initializer = 'random_uniform',
    bias_initializer = 'zeros',
    activation = 'sigmoid', # activation function (sigmoid)
    kernel_regularizer=regularizers.l2(0.006)
))

# Compiling the ANN
classifier.compile(
    optimizer = 'adam', 
    loss = 'binary_crossentropy', # cost function
    metrics = ['accuracy']
)

# fit the ann to the Training set
history = classifier.fit(
    X_train, y_train,  # training set
    validation_data = (X_test, y_test),
    batch_size = 40,
    epochs = 70,
    verbose = False
)


# In[ ]:


# summarize history for accuracy
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='lower right')
plt.show()
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper right')
plt.show()


# In[ ]:



from sklearn.metrics import confusion_matrix
y_test_pred = classifier.predict(X_test) > 0.5
cm = confusion_matrix(y_test, y_test_pred)

print('Train Accuracy: {}\nTest Accuracy:{}'.format(history.history['acc'][-1], history.history['val_acc'][-1]))
print(cm)

plt.clf()
plt.imshow(cm, interpolation='nearest', cmap=plt.cm.copper)
classNames = ['Negative','Positive']
plt.title('Fraud or Not Fraud Confusion Matrix - Test Data')
plt.ylabel('True label')
plt.xlabel('Predicted label')
tick_marks = np.arange(len(classNames))
plt.xticks(tick_marks, classNames, rotation=45)
plt.yticks(tick_marks, classNames)
s = [['TN','FP'], ['FN', 'TP']]
for i in range(2):
    for j in range(2):
        plt.text(j,i, str(s[i][j])+" = "+str(cm[i][j]), color = 'white')
plt.show()


# In[ ]:


from sklearn.metrics import f1_score
f1 = f1_score(y_test, y_test_pred)

print("F1 Score: {}".format(f1))

