#!/usr/bin/env python
# coding: utf-8

# If you like the content please upvote the kernel. Ask for anydoubt and tell if I have done anything wrong. 

# Before starting, I would like to give an overview of how to structure any machine learning project.
# 
# **Preprocess and load data**- As we have already discussed data is the key for the working of neural network and we need to process it before feeding to the neural network. In this step, we will also visualize data which will help us to gain insight into the data.
# 
# **Define model**- Now we need a neural network model. This means we need to specify the number of hidden layers in the neural network and their size, the input and output size.
# 
# **Loss and optimizer**- Now we need to define the loss function according to our task. We also need to specify the optimizer to use with learning rate and other hyperparameters of the optimizer.
# 
# **Fit model**- This is the training step of the neural network. Here we need to define the number of epochs for which we need to train the neural network.
# 
# After fitting model, we can test it on test data to check whether the case of overfitting. We can save the weights of the model and use it later whenever required.

# The dataset consists of 20 features and we need to predict the price range in which phone lies. These ranges are divided into 4 classes.
# 

# # Data preprocessing

# In[ ]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
#dataset import
dataset = pd.read_csv('../input/train.csv')
dataset.head(10)


# In[ ]:


#name of columns
dataset.columns


# This code as discussed in python module will make two arrays X and y.  X will contain features and y will contain classes.

# In[ ]:


#Changing pandas dataframe to numpy array
X = dataset.iloc[:,:20].values
y = dataset.iloc[:,20:21].values


# **Normalization of dataset**
# 
# The next step is used to normalize the data. Normalization is a technique used to change the values of an array to a common scale, without distorting differences in the ranges of values. It is an important step and you can check the difference in accuracies on our dataset by removing this step. It is mainly required in case the dataset features vary a lot as in our case the value of battery power is in the 1000's and clock speed is less than 3. So if we feed unnormalized data to the neural network, the gradients will change differently for every column and thus the learning will oscillate. Study further from this link.

# In[ ]:


#Normalizing the data
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X = sc.fit_transform(X)
print('Normalized data:')
print(X[0])


# **One hot encoding**
# 
# Next step is to one hot encode the classes. One hot encoding is a process to convert integer classes into binary values. Consider an example, let's say there are 3 classes in our dataset namely 1,2 and 3. Now we cannot directly feed this to neural network so we convert it in the form:<br>
# 1- 1 0 0<br>
# 2- 0 1 0<br>
# 3- 0 0 1
# 
# Now there is one unique binary value for the class. The new array formed will be of shape (n, number of classes), where n is the number of samples in our dataset. We can do this using simple function by sklearn:

# In[ ]:


from sklearn.preprocessing import OneHotEncoder
ohe = OneHotEncoder()
y = ohe.fit_transform(y).toarray()
print('One hot encoded array:')
print(y[0:5])


# Now our dataset is processed and ready to feed in the neural network.
# 
# Generally, it is better to split data into training and testing data. Training data is the data on which we will train our neural network. Test data is used to check our trained neural network. This data is totally new for our neural network and if the neural network performs well on this dataset, it shows that there is no overfitting. Read more about this [here](https://medium.com/r/?url=https%3A%2F%2Ftowardsdatascience.com%2Ftrain-test-split-and-cross-validation-in-python-80b61beca4b6).

# In[ ]:


from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.1)


# This will split our dataset into training and testing. Training data will have 90% samples and test data will have 10% samples. This is specified by the test_size argument.
# 
# Now we are done with the boring part and let's build a neural network.

# # Model
# Keras is a simple tool for constructing a neural network. It is a high-level framework based on tensorflow, theano or cntk backends.
# In our dataset, the input is of 20 values and output is of 4 values. So the input and output layer is of 20 and 4 dimensions respectively.

# In[ ]:


#Dependencies
import keras
from keras.models import Sequential
from keras.layers import Dense

# Neural network
model = Sequential()
model.add(Dense(16, input_dim=20, activation='relu'))
model.add(Dense(12, activation='relu'))
model.add(Dense(4, activation='softmax'))


# In our neural network, we are using two hidden layers of 16 and 12 dimension.
# Now I will explain the code line by line.
# 
# **Sequential** specifies to keras that we are creating model sequentially and the output of each layer we add is input to the next layer we specify.
# 
# **model.add** is used to add a layer to our neural network. We need to specify as an argument what type of layer we want. The **Dense** is used to specify the fully connected layer. The arguments of Dense are output dimension which is 16 in the first case, input dimension which is 20 for input dimension and the activation function to be used which is relu in this case. The second layer is similar, we dont need to specify input dimension as we have defined the model to be sequential so keras will automatically consider input dimension to be same as the output of last layer i.e 16. In the third layer(output layer) the output dimension is 4(number of classes). Now as we have discussed earlier, the output layer takes different activation functions and for the case of multiclass classification, it is softmax.

# Now we need to specify the loss function and the optimizer. It is done using compile function in keras.

# In[ ]:


model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])


# Here loss is cross entropy loss . Categorical_crossentropy specifies that we have multiple classes. The optimizer is Adam. Metrics is used to specify the way we want to judge the performance of our neural network. Here we have specified it to accuracy.
# 
# Now we are done with building a neural network and we will train it.

# In[ ]:


history = model.fit(X_train, y_train, epochs=100, batch_size=64)


# Here we need to specify the input data-> X_train, labels-> y_train, number of epochs(iterations), and batch size. It returns the history of model training. History consists of model accuracy and losses after each epoch. We will visualize it later.
# 
# Usually, the dataset is very big and we cannot fit complete data at once so we use batch size. This divides our data into batches each of size equal to batch_size. Now only this number of samples will be loaded into memory and processed. Once we are done with one batch it is flushed from memory and the next batch will be processed.

# # Test model

# Now we can check the model's performance on test data:

# In[ ]:


y_pred = model.predict(X_test)
#Converting predictions to label
pred = list()
for i in range(len(y_pred)):
    pred.append(np.argmax(y_pred[i]))
#Converting one hot encoded test label to label
test = list()
for i in range(len(y_test)):
    test.append(np.argmax(y_test[i]))


# This step is inverse one hot encoding process. We will get integer labels using this step. We can predict on test data using a simple method of keras, model.predict(). It will take the test data as input and will return the prediction outputs as softmax.

# In[ ]:


from sklearn.metrics import accuracy_score
a = accuracy_score(pred,test)
print('Accuracy is:', a*100)


# ### Validation data:

# We can use test data as validation data and can check the accuracies after every epoch. This will give us an insight into overfitting at the time of training only and we can take steps before the completion of all epochs. We can do this by changing fit function as:

# In[ ]:


#Dependencies
import keras
from keras.models import Sequential
from keras.layers import Dense
#Re initialized to delete trained weights
# Neural network
model = Sequential()
model.add(Dense(16, input_dim=20, activation='relu'))
model.add(Dense(12, activation='relu'))
model.add(Dense(4, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])


# In[ ]:


history = model.fit(X_train, y_train,validation_data = (X_test,y_test), epochs=100, batch_size=64)


# # Visualizing training

# In[ ]:


import matplotlib.pyplot as plt
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()


# In[ ]:


plt.plot(history.history['loss']) 
plt.plot(history.history['val_loss']) 
plt.title('Model loss') 
plt.ylabel('Loss') 
plt.xlabel('Epoch') 
plt.legend(['Train', 'Test'], loc='upper left') 
plt.show()

