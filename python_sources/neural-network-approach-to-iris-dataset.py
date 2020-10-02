#!/usr/bin/env python
# coding: utf-8

# 

# **An approach of Neural Network to predict Iris Species**
# 
# This kernel uses multilayer perceptrons (Neural Network) to predict the species of the Iris dataset.Neural network is a machine learning algorithm which is inspired by a neuron.
# 
# ![image.png](attachment:image.png)
# 
# A neuron consists of a dendrite and an axon which are responsible for collecting and sending signals. For our artificial neural network, the concept works similar in which a lot of neurons are connected to each layer with its own corresponding weight and biases.
# 
# Although there are currently architecture of neural network, multilayer perceptron is being used as the architecture to prevent overfitting(training accuracy=good but test accuracy=bad)  to the Iris Species due to less feature.

# In[ ]:


#Import required libraries 
import keras #library for neural network
import pandas as pd #loading data in table form  
import seaborn as sns #visualisation 
import matplotlib.pyplot as plt #visualisation
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.preprocessing import normalize #machine learning algorithm library


# In[ ]:


#Reading data 
data=pd.read_csv("../input/Iris.csv")
print("Describing the data: ",data.describe())
print("Info of the data:",data.info())


# In[ ]:


print("10 first samples of the dataset:",data.head(10))
print("10 last samples of the dataset:",data.tail(10))


# **Visualisation of the dataset**
# 
# The coding below shows the visualisation of the dataset in order to understand the data more. It can be seen that every species of the Iris can be segregated into different regions to be predicted.

# In[ ]:


sns.lmplot('SepalLengthCm', 'SepalWidthCm',
           data=data,
           fit_reg=False,
           hue="Species",
           scatter_kws={"marker": "D",
                        "s": 50})
plt.title('SepalLength vs SepalWidth')

sns.lmplot('PetalLengthCm', 'PetalWidthCm',
           data=data,
           fit_reg=False,
           hue="Species",
           scatter_kws={"marker": "D",
                        "s": 50})
plt.title('PetalLength vs PetalWidth')

sns.lmplot('SepalLengthCm', 'PetalLengthCm',
           data=data,
           fit_reg=False,
           hue="Species",
           scatter_kws={"marker": "D",
                        "s": 50})
plt.title('SepalLength vs PetalLength')

sns.lmplot('SepalWidthCm', 'PetalWidthCm',
           data=data,
           fit_reg=False,
           hue="Species",
           scatter_kws={"marker": "D",
                        "s": 50})
plt.title('SepalWidth vs PetalWidth')
plt.show()


# Coding below convert the species into each respective category to be feed into the neural network

# In[ ]:


print(data["Species"].unique())


# In[ ]:


data.loc[data["Species"]=="Iris-setosa","Species"]=0
data.loc[data["Species"]=="Iris-versicolor","Species"]=1
data.loc[data["Species"]=="Iris-virginica","Species"]=2
print(data.head())


# In[ ]:


data=data.iloc[np.random.permutation(len(data))]
print(data.head())


# Converting data to numpy array in order for processing 

# In[ ]:


X=data.iloc[:,1:5].values
y=data.iloc[:,5].values

print("Shape of X",X.shape)
print("Shape of y",y.shape)
print("Examples of X\n",X[:3])
print("Examples of y\n",y[:3])


# **Normalization**
# 
# It can be seen from above that the feature of the first dataset has 6cm in Sepal Length, 3.4cm in Sepal Width, 4.5cm in Petal Length and 1.6cm in Petal Width. However, the range of the dataset may be different. Therefore, in order to maintain a good accuracy, the feature of each dataset must be normalized to a range of 0-1 for processing 

# In[ ]:


X_normalized=normalize(X,axis=0)
print("Examples of X_normalised\n",X_normalized[:3])


# In[ ]:


#Creating train,test and validation data
'''
80% -- train data
20% -- test data
'''
total_length=len(data)
train_length=int(0.8*total_length)
test_length=int(0.2*total_length)

X_train=X_normalized[:train_length]
X_test=X_normalized[train_length:]
y_train=y[:train_length]
y_test=y[train_length:]

print("Length of train set x:",X_train.shape[0],"y:",y_train.shape[0])
print("Length of test set x:",X_test.shape[0],"y:",y_test.shape[0])


# In[ ]:


#Neural network module
from keras.models import Sequential 
from keras.layers import Dense,Activation,Dropout 
from keras.layers.normalization import BatchNormalization 
from keras.utils import np_utils


# In[ ]:


#Change the label to one hot vector
'''
[0]--->[1 0 0]
[1]--->[0 1 0]
[2]--->[0 0 1]
'''
y_train=np_utils.to_categorical(y_train,num_classes=3)
y_test=np_utils.to_categorical(y_test,num_classes=3)
print("Shape of y_train",y_train.shape)
print("Shape of y_test",y_test.shape)


# In[ ]:


model=Sequential()
model.add(Dense(1000,input_dim=4,activation='relu'))
model.add(Dense(500,activation='relu'))
model.add(Dense(300,activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(3,activation='softmax'))
model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])


# In[ ]:


model.summary()


# In[ ]:


model.fit(X_train,y_train,validation_data=(X_test,y_test),batch_size=20,epochs=10,verbose=1)


# In[ ]:


prediction=model.predict(X_test)
length=len(prediction)
y_label=np.argmax(y_test,axis=1)
predict_label=np.argmax(prediction,axis=1)

accuracy=np.sum(y_label==predict_label)/length * 100 
print("Accuracy of the dataset",accuracy )


# An accuracy of **100%** is achieved in this dataset.It can be asserted that for each epoch, the neural network is trying to learn from its existing feature and predict it by its weights and biases. For each epoch, the weights and biases and changed by subtracting its rate to get a better accuracy each time.
# 
# 
# **Further improvement: **
# 
# 1.Adding batch normalization 
# 
# 2.Adding dropout layer to prevent overfitting 

# In[ ]:




