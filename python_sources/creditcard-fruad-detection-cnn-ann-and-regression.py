#!/usr/bin/env python
# coding: utf-8

# # **Welcome to my exploration of Credit Card fraud detection!**
# 
# I will implement technique an technique called SMOTE, supervised models, semi supervised learning algorithms and a deep learning model.

# Please, if you think that I can do anything in a better way,feedback will be appreciated.

# In Machine Learning and Data Science we often come across a term called Imbalanced Data Distribution, it happens when observations in one of the class are much higher or lower than the other classes.examples such as Fraud Detection, Anomaly Detection etc.

# there are mainly two methods which are useful in the imbalance dataset to convert it in balance dataset
# so here we are using smote which stands for **SMOTE (synthetic minority oversampling technique)**.It aims to balance class distribution by randomly replicating the minority class and make dataset shape same for example if i have two classes 1 and 2 if 2 is minority class then smote will randomly replicate the data and make them same like  1 = 10000 and 2 = 200 then after applying smote the both classes will have same values 1 = 10000 and 2 = 10000.

# if you want to learn more then below link is great. it includes how to use smote on regression and also on classification
# 
# **https://machinelearningmastery.com/smote-oversampling-for-imbalanced-classification/**

# # **BEFORE APPLYING SMOTE**

# In[ ]:


#below code are used directly on the dataset we haven't applied smote for imbalanced data
#becoz the dataset is hugely imbalanced

import pandas as pd
train_df = pd.read_csv("/kaggle/input/creditcard.csv")
X = train_df.drop(columns={'Class'})
y = train_df['Class']

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.fit_transform(X_test)
y_test = y_test.ravel()
y_train = y_train.ravel()


# In[ ]:


X.info()


# In[ ]:


import matplotlib.pyplot as plt
import seaborn as sns
plt.figure(figsize=(20,14))
corr = X.corr()
sns.heatmap(corr)


# without applying smote which is use for balance the dataset

# # 1)logistic regression

# In[ ]:


# fitting logistic regression to the training set
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state=0)
classifier.fit(X_train, y_train)
# predictiing the test result
y_pred = classifier.predict(X_test)
# making the confusion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
import seaborn as sns
sns.heatmap(cm, annot=True)
#find accuracy
from sklearn.metrics import accuracy_score
print('logistic regression:',accuracy_score(y_test,y_pred))
# find classification report
from sklearn.metrics import f1_score , precision_score , recall_score
print('f1_score:',f1_score(y_test,y_pred))
print('precision_score:',precision_score(y_test,y_pred))
print('recall_score:',recall_score(y_test,y_pred))


# # 2)naive bayes

# In[ ]:



# Fitting naive byes classifier to the Training set
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(X_train,y_train)
# Predicting the Test set results
y_pred2 = classifier.predict(X_test)
# making the confusion matrix
from sklearn.metrics import confusion_matrix
cm2 = confusion_matrix(y_test, y_pred2)
import seaborn as sns
sns.heatmap(cm2, annot=True)
#find accuracy
from sklearn.metrics import accuracy_score
print('naive byes:',accuracy_score(y_test,y_pred2))
# find classification report
from sklearn.metrics import f1_score , precision_score , recall_score
print('f1_score:',f1_score(y_test,y_pred2))
print('precision_score:',precision_score(y_test,y_pred2))
print('recall_score:',recall_score(y_test,y_pred2))


# # 3)Decision tree

# In[ ]:



# Fitting Decision Tree Classification to the Training set
from sklearn.tree import DecisionTreeClassifier
classifier = DecisionTreeClassifier(criterion = 'entropy', random_state = 0)
classifier.fit(X_train, y_train)
# Predicting the Test set results
y_pred3 = classifier.predict(X_test)
# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm3 = confusion_matrix(y_test, y_pred3)
import seaborn as sns
sns.heatmap(cm3, annot=True)
#find accuracy
from sklearn.metrics import accuracy_score
print('decision tree:',accuracy_score(y_test,y_pred3))
# find classification report
from sklearn.metrics import f1_score , precision_score , recall_score
print('f1_score:',f1_score(y_test,y_pred3))
print('precision_score:',precision_score(y_test,y_pred3))
print('recall_score:',recall_score(y_test,y_pred3))


# # 4)random forest

# In[ ]:



# Fitting Random Forest Classification to the Training set
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 0)
classifier.fit(X_train, y_train)
# Predicting the Test set results
y_pred4 = classifier.predict(X_test)
# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm4 = confusion_matrix(y_test, y_pred4)
import seaborn as sns
sns.heatmap(cm4, annot=True)
#find accuracy
from sklearn.metrics import accuracy_score
print('random forest:',accuracy_score(y_test,y_pred4))
# find classification report
from sklearn.metrics import f1_score , precision_score , recall_score
print('f1_score:',f1_score(y_test,y_pred4))
print('precision_score:',precision_score(y_test,y_pred4))
print('recall_score:',recall_score(y_test,y_pred4))


# # 5)ANN

# In[ ]:


#ANN
# Importing the Keras libraries and packages
from keras.models import Sequential
from keras.layers import Dense

# Initialising the ANN
classifier = Sequential()
classifier.add(Dense(10, activation = 'relu', input_dim = 30))
classifier.add(Dense(10, activation = 'relu'))
classifier.add(Dense(1,  activation = 'sigmoid'))
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
classifier.summary()
# Fitting the ANN to the Training set
classifier.fit(X_train, y_train, batch_size = 1000, epochs = 20)
# Predicting the Test set results
y_pred5 = classifier.predict(X_test).round()
y_pred5 = (y_pred5 > 0.5)
# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
import seaborn as sns
cm5 = confusion_matrix(y_test, y_pred5)
sns.heatmap(cm5, annot=True)
#find accuracy
from sklearn.metrics import accuracy_score
print('ANN:',accuracy_score(y_test,y_pred5))
# find classification report
from sklearn.metrics import f1_score , precision_score , recall_score
print('f1_score:',f1_score(y_test,y_pred5))
print('precision_score:',precision_score(y_test,y_pred5))
print('recall_score:',recall_score(y_test,y_pred5))


# # **AFTER APPLYING SMOTE**

# now i am applying smote for balance the dataset you can see the difference in classification report.

# In[ ]:


y.value_counts()


# We have a clearly imbalanced data.
# It's very common when treating of frauds.

# In[ ]:


y.describe()


# In[ ]:


fraud = train_df[train_df['Class'] == 1]
valid = train_df[train_df['Class'] == 0]

print("Fraud transaction statistics")
print(fraud["Amount"].describe())
print("\nNormal transaction statistics")
print(valid["Amount"].describe())


# In[ ]:


# describes info about train and test set
print("X_train dataset: ", X_train.shape)
print("y_train dataset: ", y_train.shape)
print("X_test dataset: ", X_test.shape)
print("y_test dataset: ", y_test.shape)


# In[ ]:


print("before applying smote:",format(sum(y_train == 1)))
print("before applying smote:",format(sum(y_train == 0)))


# In[ ]:



# import SMOTE module from imblearn library
# pip install imblearn if you don't have
from imblearn.over_sampling import SMOTE
sm = SMOTE(random_state=2)
X_train, y_train = sm.fit_sample(X_train, y_train)

print('After applying smote X_train: {}\n'.format(X_train.shape))
print('After applying smote y_train: {}\n'.format(y_train.shape))

print("After applying smote label '1': {}\n".format(sum(y_train == 1)))
print("After applying smote label '0': {}\n".format(sum(y_train == 0)))


# # 1)logistic regressoin after applying smote

# In[ ]:


# fitting logistic regression to the training set
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state=0)
classifier.fit(X_train, y_train)
# predictiing the test result
y_pred = classifier.predict(X_test)
# making the confusion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
import seaborn as sns
sns.heatmap(cm, annot=True)
#find accuracy
from sklearn.metrics import accuracy_score
print('SMOTE+LR:',accuracy_score(y_test,y_pred))
# find classification report
from sklearn.metrics import f1_score , precision_score , recall_score , classification_report
print('classification_report:',classification_report(y_test,y_pred))
print('f1_score:',f1_score(y_test,y_pred))
print('precision_score:',precision_score(y_test,y_pred))
print('recall_score:',recall_score(y_test,y_pred))


# # 2)naive bayes after applying smote

# In[ ]:


# Fitting naive byes classifier to the Training set
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(X_train,y_train)
# Predicting the Test set results
y_pred2 = classifier.predict(X_test)
# making the confusion matrix
from sklearn.metrics import confusion_matrix
cm2 = confusion_matrix(y_test, y_pred2)
import seaborn as sns
sns.heatmap(cm2, annot=True)
#find accuracy
from sklearn.metrics import accuracy_score
print('SMOTE+naive byes:',accuracy_score(y_test,y_pred2))
# find classification report
from sklearn.metrics import f1_score , precision_score , recall_score, classification_report
print('classification_report:',classification_report(y_test,y_pred))
print('f1_score:',f1_score(y_test,y_pred2))
print('precision_score:',precision_score(y_test,y_pred2))
print('recall_score:',recall_score(y_test,y_pred2))


# # 3)Decision tree after applying smote

# In[ ]:



# Fitting Decision Tree Classification to the Training set
from sklearn.tree import DecisionTreeClassifier
classifier = DecisionTreeClassifier(criterion = 'entropy', random_state = 0)
classifier.fit(X_train, y_train)
# Predicting the Test set results
y_pred3 = classifier.predict(X_test)
# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm3 = confusion_matrix(y_test, y_pred3)
import seaborn as sns
sns.heatmap(cm3, annot=True)
#find accuracy
from sklearn.metrics import accuracy_score
print('SMOTE+decision tree:',accuracy_score(y_test,y_pred3))
# find classification report
from sklearn.metrics import f1_score , precision_score , recall_score, classification_report
print('classification_report:',classification_report(y_test,y_pred))
print('f1_score:',f1_score(y_test,y_pred3))
print('precision_score:',precision_score(y_test,y_pred3))
print('recall_score:',recall_score(y_test,y_pred3))


# # 4)random forest after applying smote

# In[ ]:



# Fitting Random Forest Classification to the Training set
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 0)
classifier.fit(X_train, y_train)
# Predicting the Test set results
y_pred4 = classifier.predict(X_test)
# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm4 = confusion_matrix(y_test, y_pred4)
import seaborn as sns
sns.heatmap(cm4, annot=True)
#find accuracy
from sklearn.metrics import accuracy_score
print('SMOTE+random forest:',accuracy_score(y_test,y_pred4))
# find classification report
from sklearn.metrics import f1_score , precision_score , recall_score, classification_report
print('classification_report:',classification_report(y_test,y_pred))
print('f1_score:',f1_score(y_test,y_pred4))
print('precision_score:',precision_score(y_test,y_pred4))
print('recall_score:',recall_score(y_test,y_pred4))


# # 5)ANN after applying smote

# In[ ]:


#ANN
# Importing the Keras libraries and packages
from keras.models import Sequential
from keras.layers import Dense

# Initialising the ANN
classifier = Sequential()
classifier.add(Dense(10, activation = 'relu', input_dim = 30))
classifier.add(Dense(10, activation = 'relu'))
classifier.add(Dense(1,  activation = 'sigmoid'))
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
classifier.summary()
# Fitting the ANN to the Training set
classifier.fit(X_train, y_train, batch_size = 1000, epochs = 20)
# Predicting the Test set results
y_pred5 = classifier.predict(X_test).round()
y_pred5 = (y_pred5 > 0.5)
# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
import seaborn as sns
cm5 = confusion_matrix(y_test, y_pred5)
sns.heatmap(cm5, annot=True)
#find accuracy
from sklearn.metrics import accuracy_score
print('SMOTE+ANN:',accuracy_score(y_test,y_pred5))
# find classification report
from sklearn.metrics import f1_score , precision_score , recall_score, classification_report
print('classification_report:',classification_report(y_test,y_pred))
print('f1_score:',f1_score(y_test,y_pred5))
print('precision_score:',precision_score(y_test,y_pred5))
print('recall_score:',recall_score(y_test,y_pred5))


# # **CNN**

# **now i am applying convolutional neural network for which we have to apply a function called reshape**

# In[ ]:


X_train = X_train.reshape(X_train.shape[0] , X_train.shape[1],1)
X_test = X_test.reshape(X_test.shape[0] , X_test.shape[1],1)


# In[ ]:


X_train.shape , X_test.shape


# # Convolutional neural network

# In[ ]:


import tensorflow as tf
from tensorflow.keras.optimizers import Adam
# Initialising the CNN
classifier = tf.keras.models.Sequential()
classifier.add(tf.keras.layers.Convolution1D(32 , 2 , activation='relu',input_shape=X_train[0].shape))
classifier.add(tf.keras.layers.BatchNormalization())
classifier.add(tf.keras.layers.Dropout(0.2))

classifier.add(tf.keras.layers.Convolution1D(64 , 2 , activation='relu'))
classifier.add(tf.keras.layers.BatchNormalization())
classifier.add(tf.keras.layers.Dropout(0.2))

classifier.add(tf.keras.layers.Convolution1D(128 , 2 , activation='relu'))
classifier.add(tf.keras.layers.BatchNormalization())
classifier.add(tf.keras.layers.Dropout(0.2))

classifier.add(tf.keras.layers.Flatten())
classifier.add(tf.keras.layers.Dense(units=256, activation='relu'))
classifier.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))
classifier.compile(optimizer=Adam(lr = 0.0001), loss='binary_crossentropy', metrics=['accuracy'])
classifier.summary()


# In[ ]:


history = classifier.fit(X_train, y_train, batch_size = 100, epochs = 10 , validation_data=(X_test,y_test),verbose=1)


# In[ ]:


# Predicting the Test set results
y_pred = classifier.predict(X_test).flatten().round()
# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
import seaborn as sns
sns.heatmap(cm, annot=True)
#find accuracy
from sklearn.metrics import accuracy_score
print('CNN:',accuracy_score(y_test,y_pred))
# find classification report
from sklearn.metrics import f1_score , precision_score , recall_score , classification_report
print('classification_report:',classification_report(y_test,y_pred))
print('f1_score:',f1_score(y_test,y_pred))
print('precision_score:',precision_score(y_test,y_pred))
print('recall_score:',recall_score(y_test,y_pred))


# In[ ]:


import matplotlib.pyplot as plt
def plot_curve(history , epoch):
    epoch_range = range(1 , epoch+1)
    plt.plot(epoch_range , history.history['accuracy'])
    plt.plot(epoch_range, history.history['val_accuracy'])
    plt.title('model accuracy')
    plt.xlabel('Accuracy')
    plt.ylabel('Epoch')
    plt.legend(['Train','Val'])
    plt.show()

    epoch_range = range(1, epoch + 1)
    plt.plot(epoch_range, history.history['loss'])
    plt.plot(epoch_range, history.history['val_loss'])
    plt.title('model loss')
    plt.xlabel('Loss')
    plt.ylabel('Epoch')
    plt.legend(['Train', 'Val'])
    plt.show()

plot_curve(history , 10)


# # **Please if you have any suggestion to improve my models, let me know!**
