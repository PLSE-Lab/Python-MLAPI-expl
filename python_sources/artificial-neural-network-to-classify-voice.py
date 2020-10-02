#!/usr/bin/env python
# coding: utf-8

# **Artificial Neural Network**
# 

# **AIM:**
# Before we dive in let me give a brief of what we are upto. We have a dataset which based on certain paramaters classifies a voice based on gender. How do humans do it?
# > Sound waves travel into the ear canal until they reach the eardrum. The eardrum passes the vibrations through the middle ear bones or ossicles into the inner ear. The inner ear is shaped like a snail and is also called the cochlea. Inside the cochlea, there are thousands of tiny hair cells. Hair cells change the vibrations into electrical signals that are sent to the brain through the hearing nerve. The brain tells you that you are hearing a sound and what that sound is.
# > 
# 
# What happens in the brain is neurons perform certain operations to classify the sound, this is exactly what we will be trying to simulate. We will try to mimmic the functioning (on a much*1000 smaller scale) just to get the basic idea.
# 

# **Step 1**
# 
# Import the basic libraries 
# matplotlib :: To plot graphs
#     numpy  :: To perform operations and manipulate arrays
#     pandas :: To read and manage the data from the file

# In[ ]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# **Step 2**
# 
# We need to separate the dependent and independent variables. Here the first 20 set columns consists of the features and the last coloumn is the dependent variable, which takes to string values i.e Male and Female

# In[ ]:


dataset=pd.read_csv("../input/voice.csv")
X=dataset.iloc[:,0:20]
y=dataset.iloc[:,-1].values


# **Step 3**
# 
# Now since we have the feature set and the set of dependent variables, We observe that the 'y' has strings and in maths we need values so what we will do is encode Male=1 and Female=0, (p.s I have nothing to do with assignments ;-) ). So we will use the LabelEncoder class and let it do its job.

# In[ ]:


from sklearn.preprocessing import LabelEncoder
labelencoder = LabelEncoder()
y = labelencoder.fit_transform(y)


# **Step 4**
# 
# Now if we observe the values in various coloumns we see that there is a problem either the values are extremely close to zero or the all the coloumn are of not the same scale. Why is there a need to Scale stuff might be the question, Well the answer is there is a lot of times where we will need to calculate slopes assume in the denominator two point are really close to zero, subtracting will lead it much more closer to zero and the slope assumes an amazingly huge value, so to prevent this kind of headache we generally use scaling in Neural Networks.

# In[ ]:



from sklearn.preprocessing import StandardScaler
X_sc = StandardScaler()
X= X_sc.fit_transform(X)


# **Step 5**
# 
# Divide the data into training set and test set, One set to train the neural Network and the other set to test the neural network.

# In[ ]:


from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=0)


# **Step 6**
# 
# Now starts the actual game of building the neural network for this we first import the required libraries.

# In[ ]:


import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import History
from keras.utils import plot_model
from keras.optimizers import SGD


# Before we get into the code let us try to understand the neural network structure we are aiming to build. What is the first thing we need to do if we want to understand or break down any sound? Simple we need to hear the sound in the coumputing terms we need to have an input. Here in the data set there are 20 paramaters which can also be called features and these are fed to the nodes on a one-to-one basis that is one node recieves one input.  We will call this the first layer and this is waht this piece of code does.
# > **classifier.add(Dense(output_dim=11,init='uniform',activation='relu',input_dim=20))**
# 
# next we pass this sound to the processing unit the brain where we have a lot of itermediate processing neurons before we actually get the output. In this case we will add just 2 intermediate stages of processing neurons with 11 nodes in each layer.
# > 
# 
# > **classifier.add(Dense(output_dim=11,init='uniform',activation='relu'))**
# 
# 
# > **classifier.add(Dense(output_dim=6,init='uniform',activation='relu'))**
# 
# Now we need to get the output and one node will do the job
# 
# > classifier.add(Dense(output_dim=1,init='uniform',activation='sigmoid'))
# 
# If you are wondering what is relu and sigmoid well these are the functions which are used to calculate the weights/loss etc. Do considering checking out what they stand for.
# 
# In the end we feed out data to the neural Network and wait for the magic to happen.

# In[ ]:


classifier=Sequential()
history = History()

#number of input variables =20
#first layer 
#input_dim is only for the first layer
classifier.add(Dense(output_dim=11,init='uniform',activation='relu',input_dim=20))
#first Hidden layer
classifier.add(Dense(output_dim=11,init='uniform',activation='relu'))
#Second Hidden
classifier.add(Dense(output_dim=6,init='uniform',activation='relu'))
#output layer
classifier.add(Dense(output_dim=1,init='uniform',activation='sigmoid'))
#Running the artificial neural network
classifier.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])
#fitting
classifier.fit(X_train,y_train,batch_size=10,epochs=10,validation_split=0.1,callbacks=[history],shuffle=2)


# **Final Step **
# 
# It is always important to see, what actually is happening and how the model is learning. So with every epoch there is some learning which happens. The model is capable of calculating the loss it is facing from the actual result and then correspondingly adjusts its weight in 

# In[ ]:



import sklearn.metrics as metrics
y_pred=classifier.predict(X_test)
y_pred = np.round(y_pred)

print('Accuracy we are able to achieve with our ANN is',metrics.accuracy_score(y_pred,y_test)*100,'%')

plt.plot(history.history['loss'], color = 'red',label='Variaton Loss over the epochs',)
plt.plot(history.history['acc'],color='cyan',label='Variation in Profit over the epochs')

plt.xlabel('Epochs')
plt.title('Loss/Accuracy VS Epoch')
plt.ylabel('Loss/Accuracy')
plt.legend(loc='best')
plt.show()


# If we observe the graph, Over a period of time it clear that the loss is gradually hitting zero and the Accuracy is increasing at a considerable rate.
