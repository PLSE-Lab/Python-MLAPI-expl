#!/usr/bin/env python
# coding: utf-8

# # Neural Network for Fraud detection with SMOTE
# 
# The datasets contains transactions made by credit cards in September 2013 by european cardholders. This dataset presents transactions that occurred in two days, where we have 492 frauds out of 284,807 transactions. The dataset is highly unbalanced, the positive class (frauds) account for 0.172% of all transactions.
# 
# In this notebook for handle the imbalanced data I have used the [SMOTE](https://jair.org/index.php/jair/article/view/10302) technique. 
# 
# I have also writed another notebook where handle this situation using [undersampling](http:// https://www.kaggle.com/davidevegliante/nn-for-fraud-detection) technique.
# 
# **These are my first two notebooks, I hope to receive comment and advice for improve the understanding of the tools I used. **
# 
# 
# 

# In[ ]:


import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import os

# read the dataset and print five rows
original_dataset = pd.read_csv('../input/creditcard.csv')

dataset = original_dataset.copy()


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


# I have decided  don't use all the non-fraud examples but choose only 3000 random example, because I don't want to use SMOTE for generating all remaining examples.

# In[ ]:


# get only 3000 non-fraud example
dataset_no_fraud = dataset.loc[dataset.Class == 0]
dataset_no_fraud_index = np.random.choice(dataset_no_fraud.index, size = 3000, replace = False)

# new dataset of (subset of non-fraud) + fraud example
new_dataset = pd.concat(
    [ dataset.iloc[dataset_no_fraud_index], # subset of non-fraud example
      dataset.iloc[dataset.index[dataset.Class == 1]] # all fraud example
    ])

# split into y and X
y = new_dataset['Class'].copy()
X = new_dataset.drop(['Time', 'Class'], axis = 1, inplace = False)

print('Dataset size: {}'.format(len(X)))


# The next step is to split our data in train and test set.
# The stratify param allow to split also the classes of y with the same ratio. 

# In[ ]:


# split in train and test set
from sklearn.model_selection import train_test_split
X_over_train, X_over_test, y_over_train, y_over_test = train_test_split(X, y, test_size = 0.20 , stratify = y)

print('Expected Fraud example in y_test: {}. Actual: {} '.format( len(y[y == 1]) * 0.2, len(y_over_test[y_over_test == 1]) )) 


# Now that the dataset is splitted into train and test set, it's possible to scale tha amount value.
# 
# **Must fix the warnings**

# In[ ]:


# scale amount
from sklearn.preprocessing import StandardScaler
am_scaler = StandardScaler().fit(X_over_train[['Amount']])

X_over_train['AmountS'] = am_scaler.transform(X_over_train[['Amount']])
X_over_test['AmountS'] = am_scaler.transform(X_over_test[['Amount']])

X_over_train.drop(['Amount'], axis = 1, inplace = True)
X_over_test.drop(['Amount'], axis = 1, inplace = True)

print("{} Train-set examples. Non-Fraud: {} Fraud: {}".format( len(X_over_train), len(y_over_train[y_over_train == 0]), len(y_over_train[y_over_train ==1]) ))
print("{} Test-set examples. Non-Fraud: {} Fraud: {}".format( len(X_over_test), len(y_over_test[y_over_test == 0]), len(y_over_test[y_over_test ==1]) ))


# Here we are :-) It's here that the Smote technique comes into action.

# In[ ]:


from imblearn.over_sampling import SMOTE
sm = SMOTE(ratio = 1.0) # with ratio = 1 we obtain the same # of examples for every class
x_over_train_sm, y_over_train_sm = sm.fit_sample(X_over_train, y_over_train)

print("Now we have {} Train-set examples. Non-Fraud: {} Fraud: {}".format( len(x_over_train_sm), len(y_over_train_sm[y_over_train_sm == 0]), len(y_over_train_sm[y_over_train_sm == 1]) ))
print("{} Test-set examples. Non-Fraud: {} Fraud: {}".format( len(X_over_test), len(y_over_test[y_over_test == 0]), len(y_over_test[y_over_test ==1]) ))


# We're ready for train our Neural Network.
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
    kernel_regularizer=regularizers.l2(0.005)
))
# add a new hidden layer with the same number of neurons
classifier.add(Dense(
    units = 6, # second hidden layer
    kernel_initializer = 'he_normal',
    bias_initializer = 'zeros',
    activation = 'relu', # activation function (rectifier)
    kernel_regularizer=regularizers.l2(0.005)
))
# add the output layer with one neuron
classifier.add(Dense(
    units = 1, # output layer
    kernel_initializer = 'random_uniform',
    bias_initializer = 'zeros',
    activation = 'sigmoid', # activation function (sigmoid)
    kernel_regularizer=regularizers.l2(0.005)
))

# Compiling the ANN
classifier.compile(
    optimizer = 'adam', 
    loss = 'binary_crossentropy', # cost function
    metrics = ['accuracy']
)


# fit the ann to the Training set
history_over = classifier.fit(
    x_over_train_sm, y_over_train_sm,  # training set
    validation_data = (X_over_test, y_over_test),
    batch_size = 100,
    epochs = 150,
    verbose = False
)


# Now we can plot the learning curves

# In[ ]:


# summarize history for accuracy
plt.plot(history_over.history['acc'])
plt.plot(history_over.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='lower right')
plt.show()

# summarize history for loss
plt.plot(history_over.history['loss'])
plt.plot(history_over.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper right')
plt.show()


# And finally the confusion matrix and F1 Score

# In[ ]:


from sklearn.metrics import confusion_matrix
y_over_test_pred = classifier.predict(X_over_test) > 0.5
cm_over = confusion_matrix(y_over_test, y_over_test_pred)

print('Train Accuracy: {}\nTest Accuracy:{}'.format(history_over.history['acc'][-1], history_over.history['val_acc'][-1]))
print(cm_over)


# In[ ]:


plt.clf()
plt.imshow(cm_over, interpolation='nearest', cmap=plt.cm.copper)
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
        plt.text(j,i, str(s[i][j])+" = "+str(cm_over[i][j]), color = 'white')
plt.show()

from sklearn.metrics import f1_score
f1 = f1_score(y_over_test, y_over_test_pred)

print("F1 Score: {}".format(f1))


# In[ ]:




