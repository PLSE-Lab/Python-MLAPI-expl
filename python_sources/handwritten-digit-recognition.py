#!/usr/bin/env python
# coding: utf-8

# # Handwritten Digit Recognition in Python with Keras

# ## Preparing the Dataset

# In[2]:


import numpy as np
import pandas as pd
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
import keras


# In[3]:


import os
print(os.listdir("../input"))
train = pd.read_csv("../input/train.csv")
test = pd.read_csv("../input/test.csv")


# In[4]:


print(train.shape)
print(test.shape)


# In[5]:


train.head()


# In[6]:


train['label'].unique()


# In[7]:


columns=[]
for col in train.columns:
    if col != 'label':
        columns.append(col)


# In[8]:


X_train = train[columns]
print(X_train.shape)
y_train = train['label']
print(y_train.shape)


# In[9]:


test.head()


# In[10]:


X_test = test[columns]
print(X_test.shape)


# Since we have to categorize data into 0-9, so using one-hot encoding technique is preffered.

# In[11]:


print('first 5 training labels : ',y_train[:5])

# convert into one-hot encoded vectors using to_categorical function
num =10
y_train = keras.utils.to_categorical(y_train,num)

print('first 5 training labels after one-hot encoding : ',y_train[:5])


# ## Neural Network Architecture

# In[12]:


from keras.layers import Dense # Dense -> fully conected layer
from keras.models import Sequential

image_size = 784
num_classes = 10

model = Sequential()

model.add(Dense(units = 32, activation = 'sigmoid',input_shape =(image_size,)))
model.add(Dense(units = num_classes,activation ='softmax'))
model.summary()


#  training and evaluating the model :
#     
# -  selected a common loss function called categorical cross entropy.
# -  selected one of the simplest optimization algorithms: Stochastic Gradient Descent (SGD).

# In[13]:


import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
batch_size = 128
model.compile(optimizer = 'sgd',loss = 'categorical_crossentropy',metrics =['accuracy'])
history = model.fit(X_train,y_train,batch_size =batch_size,epochs =3,verbose = False, validation_split =0.1)
pred_val1 = model.predict_classes(X_train)
pred_val1 = keras.utils.to_categorical(pred_val1,num)
print(classification_report(y_train, pred_val1))
acc_model1 = accuracy_score(y_train, pred_val1)
print('accuracy score : ',acc_model1)



# In[14]:


plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['training', 'validation'], loc='best')
plt.show()


# We can see the plot of training and validation accuracy over the time. Final accuracy is around 0.86

# In[ ]:





# ## Network depth and layer width

# ### Network Depth 

# The depth of multi-layer perceptron(also known as a fully connected neural network) is determined by it's number of hidden layers. The model we have used has only one hidden layer so it's a shallow network. 
# 
# so, for deep learning, we will experiment with layers of different lengths and see how it affects the performance.

# In[15]:


def create_dense(layer_size):
    model = Sequential()
    model.add(Dense(layer_size[0],activation = 'sigmoid',input_shape =(image_size,)))
    
    for s in layer_size[1:] :
        model.add(Dense(units = s, activation = 'sigmoid'))
    model.add(Dense(units = num_classes,activation ='softmax'))
    
    return model

def evaluate(model,batch_size =128, epochs =5) :
    model.summary()
    model.compile(optimizer = 'sgd',loss = 'categorical_crossentropy',metrics =['accuracy'])
    history = model.fit(X_train,y_train, batch_size = batch_size, epochs = epochs,verbose =False,validation_split =.1)
    pred_val = model.predict_classes(X_train)
    pred_val = keras.utils.to_categorical(pred_val,num)
    print(classification_report(y_train, pred_val))
    acc_model = accuracy_score(y_train, pred_val)
    print('accuracy score : ',acc_model)
    
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['training', 'validation'], loc='best')
    plt.show()

    print()


# In[16]:


for layers in range(1, 5):
    model = create_dense([32] * layers)
    evaluate(model)


# Adding more layers have decreased the accuracy of the model.This is not intuitive. One cause of this may be overfitting. We can check overfitting when we can see that training accuracy is more than test accuracy but this is only in the case of 4 hidden layers. so, problem may be something else.
# Since, every layers as input gets output of previous layer so there may be prevalent information loss.
# Also, neural networks take time to train, so we can try by incresing the time span.
# 

# we try by incresing time span from 5 epochs to 40 epochs for model with 3 hidden layers and see the result.

# In[17]:


model = create_dense([32]*3)
evaluate(model,epochs =40)


# Now, we can see that, model is giving proper accuracy.
# so, we should always experiment on our models to find which of the factors are creating problems.

# ### Layer Width

# Width of a lyer is number of nodes in each layer. Making wider layers tends to scale the number of parameters faster than adding more layers. Every time we add a single node to layer i, we have to give that new node an edge to every node in layer i+1.

# Using create_dense and evaluate functions, we try to compare neural networks with different widths with single hidden layer

# In[18]:


for nodes in [32, 64, 128, 256, 512, 1024, 2048]:
    model = create_dense([nodes])
    evaluate(model)


# From the plots, we observe that increasing number of nodes in a layer is increasing the performance from ~88% in case of 32 nodes to ~95% in acse of 2048 nodes. We can see that, in the last case with 2048 nodes, the training accuracy is somewhat predicted the accuracy of test data - so probabily there is no overfitting.
# 
# The cost of this improvement was increased training time.

# ### combining width and depth

# In[19]:


for nodes_per_layer in [32, 128, 512]:
    for layers in [3, 4, 5]:
        model = create_dense([nodes_per_layer] * layers)
        evaluate(model, epochs=10*layers)


# Highest accuracy achieved is 94% in neural network with 512 nodes and 3 layers.
# Regardless of number of nodes, all networks performed better upto 3 layers. As the layers are increased to 4 and above, accuracy is decreased. In case of 32 nodes and 5 layers, it is worst affected.

# In[21]:


model = create_dense([2048]*3)
model.compile(optimizer = 'sgd',loss = 'categorical_crossentropy',metrics =['accuracy'])
model.fit(X_train,y_train, batch_size = 128, epochs = 30,verbose =False,validation_split =.1)
predictions = model.predict_classes(X_test,verbose =0)
submissions=pd.DataFrame({"ImageId": list(range(1,len(predictions)+1)),"Label": predictions})
submissions.to_csv("out.csv", index=False, header=True)


# In[ ]:




