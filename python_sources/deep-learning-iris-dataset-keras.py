#!/usr/bin/env python
# coding: utf-8

# In[244]:


from subprocess import check_output

import numpy as np
import pandas as pd
import seaborn as sns

from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix


# In[245]:


print(check_output(["ls", "../input"]).decode("utf8"))


# In[246]:


dataset = pd.read_csv('../input/Iris.csv')
dataset.head()


# In[247]:


sns.pairplot(dataset.iloc[:,1:6],hue="Species")
# Iris-setosa has quite unique features as it can be seen much separated from other two species


# In[248]:


#Splitting the data into training and test test
X = dataset.iloc[:,1:5].values
y = dataset.iloc[:,5].values

encoder =  LabelEncoder()
y1 = encoder.fit_transform(y)

Y = pd.get_dummies(y1).values

X_train,X_test, y_train,y_test = train_test_split(X,Y,test_size=0.2,random_state=0)


# In[249]:


# Model
model = Sequential()

# first input layer with first hidden layer in a single statement
model.add( Dense(30, input_shape=(4,), activation='relu') )
# 10 is the size(no. of neurons) of first hidden layer, 4 is the no. of features in the input layer
# input_shape=(4,)  can also be written as   input_dim=4

# second hiden layer
model.add(Dense(10,activation='relu')) # 8 = no. of neurons in second hidden layer

# third hiden layer
model.add(Dense(5,activation='relu')) # 6 = no. of neurons in third hidden layer

# ouput layer
model.add(Dense(3,activation='softmax')) # 3 = no. of neurons in output layer as three categories of labels are there

# compile method receives three arguments: "an optimizer", "a loss function" and "a list of metrics"
model.compile(Adam(lr=0.04),'categorical_crossentropy', ['accuracy'])
# we use "binary_crossentropy" for binary classification problems and
# "categorical_crossentropy" for multiclass classification problems
# the compile statement can also be written as:-
# model.compile(optimizer=Adam(lr=0.04), loss='categorical_crossentropy',metrics=['accuracy'])
# we can give more than one metrics like ['accuracy', 'mae', 'mape']

model.summary()


# In[250]:


#fitting the model and predicting 
model.fit(X_train, y_train, epochs=100)
y_pred = model.predict(X_test)


# In[251]:


y_test_class = np.argmax(y_test,axis=1) # convert encoded labels into classes: say [0, 0, 1] -->  [2] i.e Iris-virginica
y_pred_class = np.argmax(y_pred,axis=1) # convert predicted labels into classes: say [0.00023, 0.923, 0.031] -->  [1] i.e. Iris-versicolor

#Accuracy of the predicted values
print(classification_report(y_test_class, y_pred_class)) # Precision , Recall, F1-Score & Support
cm = confusion_matrix(y_test_class, y_pred_class)
print(cm)
# visualize the confusion matrix in a heat map
df_cm = pd.DataFrame(cm)
heatmap = sns.heatmap(df_cm, annot=True, fmt="d")


# In[252]:


score = model.evaluate(X_test, y_test) #evaluate() Returns the loss value & metrics values for the model in test mode
score

