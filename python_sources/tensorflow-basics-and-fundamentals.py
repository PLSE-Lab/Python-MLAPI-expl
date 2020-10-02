#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# # Basics of tensorflow ?
# 1. ### Fundamentals of tensorflow 
#     1. Data structure in TF
#     1. Data flow graphs 
#     1. tensorflow sessions 
#     1. tensorflow placeholder
#     1. tensorflow variables    
#     
# 1. ### Forward propagation 
#     1. all types of loss function 
#     1. softmax vs cross entropy is same ? quick trick question let's who can explain me the difference 
#     1. gradient descent 
#     1. optimzers
#     1. activation functions 
#     1. Batching  and it's types 
# 1. ### tensorflow for classification and Regression
#     1. Regularisation 
#     1. Batch Normalization 
#     1. Advance training algorithms

# ### import tensorflow 

# In[ ]:


import tensorflow as tf


# # Data structure
#    * In Tensorflow we have tensor(fancy name for arrays with arbitary dimensions) for representing data and they array of varying dimensions
#    * one dimensional array is called as vector in tensorflow it is refferred is 1d tensor same as for 2 dimensional space is called 2d Tensor even for cube it is refferred as 3d tensor so basically dimension is property that describe every tensor
#    
#    
# ![](https://miro.medium.com/max/1752/1*RiQ5LRM0CLfJToI3fk7VLA.png**)   

# #### Q1. now one big question for all of is there any difference between data type vs tensorflow data type if yes then what is that and if no then why we can't define it like other 

# # Dataflow Graph
# *    basically to carry out operation in tensorflow it is done with the help of dataflow graphs / diagrams     (internally) whatever constants variables optimizer weights biases you define that flow within graphs and operations are performed on them when we define sessions and intiate them explicitly
# 
# *    graph contains 2 things node and edges , on node (operations performed) on edge (data-flows)
# 
# ![](https://miro.medium.com/max/1200/1*vPb9E0Yd1QUAD0oFmAgaOw.png)

# #### as you can easily see internally in tensorflow how edges and node are used to carry out dataflow 

# # Dataflow graph of NN 
# 
# ![](https://images.techhive.com/images/article/2016/10/tensorflow-data-flow-100685891-large.idge.png)

# #### Q2 Why dataflow graph came up as idea to tensorflow developers because when we have numpy and we can still build the model using numpy yes we can then why dataflow graphs? ask yourself or wait for live session 

# # Tensorflow Session (switches to control your dataflow graphs)

# #### so big picture in small words see in any tensorflow programme consists of 2 parts
# 1. Data-flow (series of tensorflow operations arranged of graph nodes)
# 1. Running computation graph in a session (encapsulates control and tensorflow runtime)

# In[ ]:


import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()


# In[ ]:


# Create an empty graph, that will be filled with operation nodes later
main_graph=tf.Graph()


# Register this graph as default graph. 
# All operations within this context will become operation nodes in this graph
with main_graph.as_default():
    
    # Create two constants of value 5
    a=tf.constant(5.0)
    b=tf.constant(5.0)
    
    # Multiply the constants with each other
    c=tf.multiply(a,b)
    

# Create a session to execute the dataflow graph
with tf.Session(graph=main_graph) as session:
    
    # Perform the calculation defined in the dataflow graph and get the result
    output=session.run(c)
    print('Result of the multiplication: %d '%output)


# # Tensorflow placeholders

# #### How you will feed python external data ? in simple terms (x_train and x_test) how you will feed that data in dataflow graphs for that we need placeholders 
# #### so with help of placeholders you can define graphs earlier and can feed the data later on that is possible because with the help of placeholders
# #### so we define empty placeholder which will take data later training time and feed this data in data flow graphs
# 
# #### Q3 When python a array becomes a tensor after feeding in neural network or when the values  flows in dataflow graphs ?

# #### now to feed data in place holder we just need a dictionary 

# ![](data:image/png;base64,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)

# In[ ]:


# Create an empty graph, that will be filled with operation nodes later
main_graph=tf.Graph()

# Register this graph as default graph. 
# All operations within this context will become operation nodes in this graph
with main_graph.as_default():
    
    # Define the placeholders that will feed python arrays into the dataflow graph
    a=tf.placeholder(name='a', shape=[5], dtype=tf.float32)
    b=tf.placeholder(name='b', shape=[5], dtype=tf.float32)
    
    c=tf.multiply(a,b)
    
# Create a session to execute the dataflow graph
with tf.Session(graph=main_graph) as session:
    
    # Perform the calculation defined in the dataflow graph and get the result.
    # We must provide the values for the placeholders with "feed_dict" dictionary
    output=session.run(c, feed_dict={a: [5.0,7.0,3.0,9.0,2.0],
                                     b: [1.0,2.0,4.0,8.0,4.0],
                                     })
    print(output)

   


# # Carrying out Forward Propagation (TF variables)
# * for defining weight matrix and biases we need another concept to learn called as tensorflow variables 
# 
# ** Note placeholder and constant can also define tensor same as tf variables 
# but why need it ?
# 
# see till now we haven't considered any trainable paramters which are getting dynamically changes up after every iteration , with help of constants by definition it is constant and place-holder to feed data which again a mapped dict so now need different tensor whose values is always a variable (dynamic change requires) from their concept came up of tensor variables
# 
# 

# #### now here are 2 things first we have tf.Variable() and another is .get_variable() the first one takes up intial values and second one take random values from a probability distribution of our choice for training weights second method is considered more rather than first because we want our weights to be random during initalization
# 

# ### how to initialize any varibale (how to send them) in the dataflow graph
# ### so when dataflow graph is executed in session variables are initialized but for that any variable in data flow graph you wanna intialize is done with tf.global_variables_intializer() 

# In[ ]:


# Import numpy for some array reshaping
import numpy as np

# Create an empty graph, that will be filled with operation nodes later
main_graph=tf.Graph()

# Register this graph as default graph. 
# All operations within this context will become operation nodes in this graph
with main_graph.as_default():
    # Placeholder for data input
    input_=tf.placeholder(dtype=tf.float32, shape=[1,5], name='input')
    
    ###Define the weight matrices###
    # Weight matrix, that connects the input and the 1st hidden layer
    W1=tf.get_variable(name='W1', shape=[5,10], initializer=tf.random_normal_initializer)
    # Weight matrix, that connects the 1st hidden layer and the 2nd hidden layer
    W2=tf.get_variable(name='W2', shape=[10,10], initializer=tf.random_normal_initializer)
    # Weight matrix, that connects the 2nd hidden layer and the 3rd hidden layer
    W3=tf.get_variable(name='W3', shape=[10,10], initializer=tf.random_normal_initializer)
    # Weight matrix, that connects the 3rd hidden layer and the output layer
    W4=tf.get_variable(name='W4', shape=[10,3], initializer=tf.random_normal_initializer)
         
    ####Define the forward propagation operations###
      
    #1st hidden layer
    z1=tf.matmul(input_, W1)
    a1=tf.nn.tanh(z1)
    
    #2nd hidden layer
    z2=tf.matmul(a1, W2)
    a2=tf.nn.tanh(z2)
    
    #3rd hidden layer
    z3=tf.matmul(a2, W3)
    a3=tf.nn.tanh(z3)
    
    #output layer
    z_out=tf.matmul(a3, W4)
    output=tf.nn.tanh(z_out)


# Create a session to execute the dataflow graph   
with tf.Session(graph=main_graph) as sess:
    
    # Initialize the weight matrix
    sess.run(tf.global_variables_initializer())
    
    # Define some random input
    x=np.array([1.0, 2.5, 0.7, 3.0, 9.0]).reshape([1,5])
    
    # Start the forward propagation step
    prediction=sess.run(output, feed_dict={input_: x})
    
    print('\n\n\nResult of the forward propagation: \n')
    print(prediction)


# ### people familiar with forward propagation can understand this code
# 
# ### to make it as you can see that (sess) will initialize everything above it 
# 
# ### to initialze anything we need to use sess.run() (* not to forget point)

# # Moving on 2nd part of this tutorial

# #### So here tutorial is all about components which we need to understand how to call them and setup with feed forward propagation (NN)  

# ## Mse use case 

# In[ ]:


# Import numpy for some array reshaping
import numpy as np

# Create an empty graph, that will be filled with operation nodes later
main_graph=tf.Graph()

# Register this graph as default graph. 
# All operations within this context will become operation nodes in this graph
with main_graph.as_default():
    
    # Placeholder for the input data
    input_= tf.placeholder(dtype=tf.float32, shape=[1,3], name='input')
    # Placeholder for the label
    label = tf.placeholder(dtype=tf.float32, shape=[1,2], name='label')
    
    # Define a 3x2 weight matrix
    W=tf.get_variable(name='weights', shape=[3,2], dtype=tf.float32)
    
    # Compute a forward propagation step
    output=tf.nn.tanh(tf.matmul(input_, W))
    
    # Use build-in function for the mean squared error loss
    loss_op=tf.losses.mean_squared_error(labels=label, predictions=output)
    

# Create a session to execute the dataflow graph

with tf.Session(graph=main_graph) as sess:
    
    # Initialize the weight matrix
    sess.run(tf.global_variables_initializer())
    
    # Define some random input (x) and a label (y)
    # Reshape the arrays into the shape of the placeholders
    x=np.reshape([10.0,3.0,4.0], [1,3])
    y=np.reshape([5.0,2.0], [1,2])

    # Compute the prediction of the network
    nn_output=sess.run(output, feed_dict={input_: x})

    # Compute the mean squared error loss
    mse_loss=sess.run(loss_op, feed_dict={input_: x,
                                         label: y})
    print('\n\n\nOutput of the neural network: \n')
    print(nn_output)
    
    print('\nMean squared error loss: %.2f'%mse_loss)


# ## Cross entropy and sparse cross entropy example 

# In[ ]:


# Import numpy for some array reshaping
import numpy as np

# Create an empty graph, that will be filled with operation nodes later
main_graph=tf.Graph()

# Register this graph as default graph. 
# All operations within this context will become operation nodes in this graph
with main_graph.as_default():
    
    # Placeholder for the input data
    input_= tf.placeholder(dtype=tf.float32, shape=[1,3], name='input')
    # Placeholder for the scalar label
    label = tf.placeholder(dtype=tf.int32, shape=[1], name='label_sparse')
    # Placeholder for the one-hot-encoded label
    label_ohe = tf.placeholder(dtype=tf.float32, shape=[1,5], name='label_one_hot_encoded')

    # Define a 3x5 weight matrix
    W=tf.get_variable(name='weights', shape=[3,5], dtype=tf.float32)
    
    # Forward propagation without an activation function
    logits=tf.matmul(input_, W)
    
    # Cross entropy loss operation, that requires the one-hot-encoded label
    loss_op=tf.nn.softmax_cross_entropy_with_logits_v2(labels=label_ohe, 
                                                       logits=logits, 
                                                       name='cross_entropy_loss')

    # Cross entropy loss operation, that requires the scalar label
    loss_op_sparse=tf.nn.sparse_softmax_cross_entropy_with_logits(labels=label, 
                                                                  logits=logits, 
                                                                  name='sparse_cross_entropy_loss')


# Create a session to execute the dataflow graph
with tf.Session(graph=main_graph) as sess:
    
    # Initialize the weight matrix
    sess.run(tf.global_variables_initializer())
    
    # Define some random input (x)
    x=np.reshape([10.0,3.0,4.0], [1,3])
    
    # Define the scalar label 
    y=np.reshape([2], [1])
    # The same label but one-hot-encoded version of it
    y_ohe=np.reshape([0,0,1,0,0], [1,5])
    
    # Run the one cross entropy loss operation
    loss=sess.run(loss_op, feed_dict={input_:x, 
                                      label_ohe: y_ohe
                                      })
    
    # Run the other cross entropy loss operation
    sparse_loss=sess.run(loss_op_sparse, feed_dict={input_:x,
                                                    label: y
                                                    })
    
    print('\n\n\nCross entropy loss: %.2f'%loss)
    print('Sparse cross entropy loss: %.2f'%sparse_loss)


# ## Gradient descent example

# In[ ]:


# Import numpy for some array reshaping
import numpy as np

# Create an empty graph, that will be filled with operation nodes later
main_graph=tf.Graph()

# Register this graph as default graph. 
# All operations within this context will become operation nodes in this graph
with main_graph.as_default():
    
    # Placeholder for data input
    input_=tf.placeholder(dtype=tf.float32, shape=[1,5], name='input')
    # Placeholder for the label
    labels=tf.placeholder(dtype=tf.float32, shape=[1,1], name='labels')

    # Define a 5x1 weight matrix
    W=tf.get_variable(name='weights', shape=[5,1])
    
    # Forward propagation
    forward=tf.nn.tanh(tf.matmul(input_, W))

    # Mean squared error loss function
    loss=tf.losses.mean_squared_error(labels, forward)
    
    # Define the instance of the class that performs the gradient descent step
    optimizer=tf.train.GradientDescentOptimizer(learning_rate=0.1)
    
    # Get all trainable parameters of the network (here: the weight matrix W)
    trainable_variables=tf.trainable_variables()
    
    # Compute the gradients of the loss function with respect to the weights
    gradients= tf.gradients(loss, trainable_variables)
    
    # Perform the gradient descent step.
    # The input argument are tuples. Each tuple is a pair of the gradient and the weight 
    # that was used to calculate this gradient
    update_step=optimizer.apply_gradients(zip(gradients, trainable_variables))
     
    
    #Alternative Solution 
    update_step_alternative=tf.train.GradientDescentOptimizer(learning_rate=0.001).minimize(loss)
       
    
    
# Create a session to execute the dataflow graph   
with tf.Session(graph=main_graph) as sess:
    
    # Initialize the weight matrix
    sess.run(tf.global_variables_initializer())
    
    # Define some random input and a label
    x=np.array([1,1,0,2,5]).reshape([1,5])
    y=np.array([2]).reshape([1,1]) 
    
    # Perform one single gradient descent step
    sess.run(update_step, feed_dict={input_: x,
                                     labels: y
                                     }) 


# ## Code for Neural network regression must see example

# In[ ]:


import numpy as np
from random import shuffle

# list that will contain our training set
training_data=[]

# How many digits should this binary number have?
n=11
    
#Mini-batch size
batch_size=16

print('\n\nGeneration of Data...\n') 
for i in np.arange(2048):
    
    # Create a binary number of type string
    b = bin(i)[2:]
    l = len(b)
    b = str(0) * (n - l) + b  

    # Convert the binary string number to type float
    features=np.array(list(b)).astype(float)
    
    # Create and normalize the corresponding decimal label
    label=float(i)/2047
    
    # Put the feature-label instance into the list
    training_data.append([features, label])
    
    if (i>=1 and i<11) or (i>=2038):
        print('binary number: %s, decimal number: %d' %(b, i))
        
# shuffle the data
shuffle(training_data)  

# convert the list to np.array     
training_data=np.array(training_data)


#%%

# Get the next mini-batch of training samples
def get_next_batch(n_batch):
    
    # Get the next mini-batch of training samples from the dataset
    features=training_data[n_batch*batch_size:(batch_size*(1+n_batch)),0]
    labels=training_data[n_batch*batch_size:(batch_size*(1+n_batch)),1]
    
    # Reshape the list of arrays into a nxn np.array
    features = np.concatenate(features).reshape([batch_size,11])  
    # Reshape the labels 
    labels=np.reshape(labels, [batch_size,1])
    
    return features, labels
    
features, labels=get_next_batch(n_batch=1)

print(features)
print(labels)
#%%  

# Create the training graph
main_graph=tf.Graph()

with main_graph.as_default():
    
    # Define the placeholders for the features and the labels
    x=tf.placeholder(dtype=tf.float32,shape=[batch_size,11], name='features')
    y=tf.placeholder(dtype=tf.float32, shape=[batch_size,1], name='labels')
           
    # Create the weight matrices and the bias vectors 
    initializer=initializer=tf.random_normal_initializer(mean=0.0, stddev=0.25)
   
    W1=tf.get_variable('W1',shape=[11,50], initializer=initializer)
    W2=tf.get_variable('W2',shape=[50,25], initializer=initializer)
    W3=tf.get_variable('W3',shape=[25,1], initializer=initializer)
    
    b1=tf.get_variable('b1',shape=[50], initializer=initializer)
    b2=tf.get_variable('b2',shape=[25], initializer=initializer)

    ### Define the forward propagation step ###
    
    # First hidden layer
    z1=tf.matmul(x,W1)+b1
    a1=tf.nn.tanh(z1)
    
    # Second hidden layer
    z2=tf.matmul(a1,W2)+b2
    a2=tf.nn.tanh(z2)
    
    # Outputlayer
    predict_op=tf.nn.relu(tf.matmul(a2,W3))
    
    # Define the loss function
    loss_op=tf.losses.mean_squared_error(y,predict_op)
       
    # Perform a gradient descent step
    optimizer=tf.train.GradientDescentOptimizer(learning_rate=0.01)
    trainable_parameters = tf.trainable_variables()
    gradients = tf.gradients(loss_op,trainable_parameters)
    update_step = optimizer.apply_gradients(zip(gradients, trainable_parameters))


print('\n\nStart of the training...\n')
with tf.Session(graph=main_graph) as sess:
    
    sess.run(tf.global_variables_initializer())

    # How many mini-batches in total?
    num_batches=int(2048/batch_size)
    
    loss=0

    #Iterate over the entire training set for 10 times
    for epoch in range(10):
            
        # Iterate over the number of mini-batches
        for n_batch in range(num_batches-1):
            
            # Get the next mini-batches of samples for the training set
            features, labels=get_next_batch(n_batch)
              
            # Perform the gradient descent step on that mini-batch and compute the loss value
            _, loss_=sess.run((update_step, loss_op), feed_dict={x:features, y:labels})   
            loss+=loss_
             
        print('epoch_nr.: %i, loss: %.3f' %(epoch,(loss/num_batches)))
        loss=0 

    # Compute the prediction on the last mini-batch
    prediction=sess.run(predict_op, feed_dict={x:features})
    
    # Iterate over the features and labels from the last mini-batch as well as
    # the predicitons made by the network, and compare them to check the performance
    for f, l, p in zip(features, labels, prediction):
        
        # Rescale the predictions and labels back into their original value range
        p=p*2047
        l=l*2047
        
        print('Binary number: %s,  label: %i, prediciton %i' %(str(f), l,p))


# ## Code for Neural network for classification

# In[ ]:


import numpy as np
from random import shuffle

# list that will contain our training set
training_data=[]

# How many digits should this binary number have?
n=4
    
#Mini-batch size
batch_size=8

print('\n\nGeneration of Data...\n') 
for i in np.arange(0, 10):
    
    # Create a binary number of type string
    b = bin(i)[2:]
    l = len(b)
    b = str(0) * (n - l) + b  

    # Convert binary string number to type float
    features=np.array(list(b)).astype(float)
    # Create the corresponding binary label / class
    label=i
    
    # Put the feature-label pair into the list
    training_data.append([features, label])

    print('binary number: %s, decimal number: %d' %(b, i))
        
    
#%%
# shuffle the data
shuffle(training_data)  

training_data=training_data*1000

# convert the list to np.array     
training_data=np.array(training_data)



#%%

# Get the next mini-batch of training samples
def get_next_batch(n_batch):
    
    # Get the next mini-batch of training samples from the dataset
    features=training_data[n_batch*batch_size:(batch_size*(1+n_batch)),0]
    labels=training_data[n_batch*batch_size:(batch_size*(1+n_batch)),1]
    
    # Reshape the list of arrays into a nxn np.array
    features = np.concatenate(features).reshape([batch_size, 4])  
    # Reshape the labels 
    labels=np.reshape(labels, [batch_size])
    
    return features, labels
    
features, labels=get_next_batch(n_batch=1)

print('\n\nMini-batch of features: \n')
print(features)
print('\n\nMini-batch of labels: \n')
print(labels)

#%%  

# Create the training graph
main_graph=tf.Graph()

with main_graph.as_default():
    
    # Define the placeholders for the features and the labels
    x=tf.placeholder(dtype=tf.float32,shape=[batch_size, 4], name='features')
    y=tf.placeholder(dtype=tf.int32, shape=[batch_size], name='labels')
           
    # Create the weight matrices and the bias vectors 
    initializer=initializer=tf.random_normal_initializer(mean=0.0, stddev=0.25)
   
    W1=tf.get_variable('W1',shape=[4,50], initializer=initializer)
    W2=tf.get_variable('W2',shape=[50,25], initializer=initializer)
    W3=tf.get_variable('W3',shape=[25,10], initializer=initializer)
    
    b1=tf.get_variable('b1',shape=[50], initializer=initializer)
    b2=tf.get_variable('b2',shape=[25], initializer=initializer)

    ### Define the forward propagation step ###
    
    # First hidden layer
    z1=tf.matmul(x,W1)+b1
    a1=tf.nn.tanh(z1)
    
    # Second hidden layer
    z2=tf.matmul(a1,W2)+b2
    a2=tf.nn.tanh(z2)
    
    # Outputlayer, without an activation function (input for the loss function)
    logits=tf.matmul(a2,W3)
       
    # Compute the probability scores after the training)
    probs=tf.nn.softmax(logits)
    
    # Define the loss function
    loss_op=tf.losses.sparse_softmax_cross_entropy(labels=y, logits=logits)
       
    # Perform a gradient descent step
    optimizer=tf.train.GradientDescentOptimizer(learning_rate=0.001)
    trainable_parameters = tf.trainable_variables()
    gradients = tf.gradients(loss_op,trainable_parameters)
    update_step = optimizer.apply_gradients(zip(gradients, trainable_parameters))


print('\n\nStart of the training...\n')
with tf.Session(graph=main_graph) as sess:
    
    sess.run(tf.global_variables_initializer())

    # How many mini-batches in total?
    num_batches=int(10000/batch_size)
    
    loss=0

    #Iterate over the entire training set for 10 times
    for epoch in range(10):
            
        # Iterate over the number of mini-batches
        for n_batch in range(num_batches-1):
            
            # Get the next mini-batches of samples for the training set
            features, labels=get_next_batch(n_batch)
              
            # Perform the gradient descent step on that mini-batch and compute the loss value
            _, loss_=sess.run((update_step, loss_op), feed_dict={x:features, y:labels})   
            
            loss+=loss_
             
        print('epoch_nr.: %i, loss: %.3f' %(epoch,(loss/num_batches)))
        loss=0 
    
    
    print('\n\nTesting the neural network:\n')
    # Compute the probability scores for the last mini-batch
    prob_scores=sess.run(probs, feed_dict={x:features, y:labels})
    
    # Iterate over the features and labels from the last mini-batch as well as
    # the predicitons made by the network, and compare them to check the performance
    for f, l, p in zip(features, labels, prob_scores):
    
        # Get the class with the highest probability score
        predicted_class=np.argmax(p)
        # Get the actual probability score
        predicted_class_score=np.max(p)
     
        print('Binary number: %s, decimal number: %i, predicted_class: %i, predicted_prob_score: %.3f' 
              %(str(f), l, predicted_class, predicted_class_score))


# ## Dropout example

# In[ ]:


import numpy as np
from random import shuffle

# list that will contain our training set
training_data=[]

# How many digits should this binary number have?
n=11
    
#Mini-batch size
batch_size=16

print('\n\nGeneration of Data...\n') 
for i in np.arange(2048):
    
    # Create a binary number of type string
    b = bin(i)[2:]
    l = len(b)
    b = str(0) * (n - l) + b  

    # Convert the binary string number to type float
    features=np.array(list(b)).astype(float)
    
    # Create and normalize the corresponding decimal label
    label=float(i)/2047
    
    # Put the feature-label instance into the list
    training_data.append([features, label])
    
    if (i>=1 and i<11) or (i>=2038):
        print('binary number: %s, decimal number: %d' %(b, i))
        
# shuffle the data
shuffle(training_data)  

# convert the list to np.array     
training_data=np.array(training_data)


#%%

# Get the next mini-batch of training samples
def get_next_batch(n_batch):
    
    # Get the next mini-batch of training samples from the dataset
    features=training_data[n_batch*batch_size:(batch_size*(1+n_batch)),0]
    labels=training_data[n_batch*batch_size:(batch_size*(1+n_batch)),1]
    
    # Reshape the list of arrays into a nxn np.array
    features = np.concatenate(features).reshape([batch_size,11])  
    # Reshape the labels 
    labels=np.reshape(labels, [batch_size,1])
    
    return features, labels
    
features, labels=get_next_batch(n_batch=1)

print(features)
print(labels)
#%%  

# Create the training graph
main_graph=tf.Graph()

with main_graph.as_default():
    
    # Define the placeholders for the features and the labels
    x=tf.placeholder(dtype=tf.float32,shape=[batch_size,11], name='features')
    y=tf.placeholder(dtype=tf.float32, shape=[batch_size,1], name='labels')
           
    # Create the weight matrices and the bias vectors 
    initializer=initializer=tf.random_normal_initializer(mean=0.0, stddev=0.25)
   
    W1=tf.get_variable('W1',shape=[11,50], initializer=initializer)
    W2=tf.get_variable('W2',shape=[50,25], initializer=initializer)
    W3=tf.get_variable('W3',shape=[25,1], initializer=initializer)
    
    b1=tf.get_variable('b1',shape=[50], initializer=initializer)
    b2=tf.get_variable('b2',shape=[25], initializer=initializer)

    ### Define the forward propagation step with Dropout ###
    
    z1=tf.matmul(x,W1)+b1
    a1=tf.nn.tanh(z1)
    a1_dropout= tf.nn.dropout(a1, rate=0.25)
    
    z2=tf.matmul(a1_dropout,W2)+b2
    a2=tf.nn.tanh(z2)
    a2_dropout= tf.nn.dropout(a2, rate=0.25)
    
    # Outputlayer
    predict_op=tf.nn.relu(tf.matmul(a2_dropout,W3))
    
    # Define the loss function
    loss_op=tf.losses.mean_squared_error(y,predict_op)
       
    # Perform a gradient descent step
    optimizer=tf.train.GradientDescentOptimizer(learning_rate=0.01)
    trainable_parameters = tf.trainable_variables()
    gradients = tf.gradients(loss_op,trainable_parameters)
    update_step = optimizer.apply_gradients(zip(gradients, trainable_parameters))


print('\n\nStart of the training...\n')
with tf.Session(graph=main_graph) as sess:
    
    sess.run(tf.global_variables_initializer())

    # How many mini-batches in total?
    num_batches=int(2048/batch_size)
    
    loss=0

    #Iterate over the entire training set for 10 times
    for epoch in range(10):
            
        # Iterate over the number of mini-batches
        for n_batch in range(num_batches-1):
            
            # Get the next mini-batches of samples for the training set
            features, labels=get_next_batch(n_batch)
              
            # Perform the gradient descent step on that mini-batch and compute the loss value
            _, loss_=sess.run((update_step, loss_op), feed_dict={x:features, y:labels})   
            loss+=loss_
             
        print('epoch_nr.: %i, loss: %.3f' %(epoch,(loss/num_batches)))
        loss=0 

    # Compute the prediction on the last mini-batch
    prediction=sess.run(predict_op, feed_dict={x:features})
    
    # Iterate over the features and labels from the last mini-batch as well as
    # the predicitons made by the network, and compare them to check the performance
    for f, l, p in zip(features, labels, prediction):
        
        # Rescale the predictions and labels back into their original value range
        p=p*2047
        l=l*2047
        
        print('Binary number: %s,  label: %i, prediciton %i' %(str(f), l,p))


# ## L2 regularisation example 

# In[ ]:


import numpy as np
from random import shuffle

# list that will contain our training set
training_data=[]

# How many digits should this binary number have?
n=11
    
#Mini-batch size
batch_size=16

print('\n\nGeneration of Data...\n') 
for i in np.arange(2048):
    
    # Create a binary number of type string
    b = bin(i)[2:]
    l = len(b)
    b = str(0) * (n - l) + b  

    # Convert the binary string number to type float
    features=np.array(list(b)).astype(float)
    
    # Create and normalize the corresponding decimal label
    label=float(i)/2047
    
    # Put the feature-label instance into the list
    training_data.append([features, label])
    
    if (i>=1 and i<11) or (i>=2038):
        print('binary number: %s, decimal number: %d' %(b, i))
        
# shuffle the data
shuffle(training_data)  

# convert the list to np.array     
training_data=np.array(training_data)


#%%

# Get the next mini-batch of training samples
def get_next_batch(n_batch):
    
    # Get the next mini-batch of training samples from the dataset
    features=training_data[n_batch*batch_size:(batch_size*(1+n_batch)),0]
    labels=training_data[n_batch*batch_size:(batch_size*(1+n_batch)),1]
    
    # Reshape the list of arrays into a nxn np.array
    features = np.concatenate(features).reshape([batch_size,11])  
    # Reshape the labels 
    labels=np.reshape(labels, [batch_size,1])
    
    return features, labels
    
features, labels=get_next_batch(n_batch=1)

print(features)
print(labels)
#%%  

# Create the training graph
main_graph=tf.Graph()

with main_graph.as_default():
    
    # Define the placeholders for the features and the labels
    x=tf.placeholder(dtype=tf.float32,shape=[batch_size,11], name='features')
    y=tf.placeholder(dtype=tf.float32, shape=[batch_size,1], name='labels')
           
    # Create the weight matrices and the bias vectors 
    initializer=initializer=tf.random_normal_initializer(mean=0.0, stddev=0.25)
   
    W1=tf.get_variable('W1',shape=[11,50], initializer=initializer)
    W2=tf.get_variable('W2',shape=[50,25], initializer=initializer)
    W3=tf.get_variable('W3',shape=[25,1], initializer=initializer)
    
    b1=tf.get_variable('b1',shape=[50], initializer=initializer)
    b2=tf.get_variable('b2',shape=[25], initializer=initializer)

    ### Define the forward propagation step ###
    
    # First hidden layer
    z1=tf.matmul(x,W1)+b1
    a1=tf.nn.tanh(z1)
    
    # Second hidden layer
    z2=tf.matmul(a1,W2)+b2
    a2=tf.nn.tanh(z2)
    
    # Outputlayer
    predict_op=tf.nn.relu(tf.matmul(a2,W3))
    
	
	
	#### L2-Regularization ####
	
	# Collect the weight matrices
    weight_matrices=[var for var in tf.trainable_variables() if 'bias' not in var.name] 
	#Apply the L2 regularization to the weight matrices
    l2_losses=[tf.nn.l2_loss(w) for w in weight_matrices]
    l2_loss = tf.add_n(l2_losses)
    
	# Regularization rate
    alpha=0.1
    
	# Add the L2 regularization to the regular loss function
    loss_op=tf.losses.mean_squared_error(y,predict_op)+alpha*l2_loss
	
	########
       
    # Perform a gradient descent step
    optimizer=tf.train.GradientDescentOptimizer(learning_rate=0.01)
    trainable_parameters = tf.trainable_variables()
    gradients = tf.gradients(loss_op,trainable_parameters)
    update_step = optimizer.apply_gradients(zip(gradients, trainable_parameters))


print('\n\nStart of the training...\n')
with tf.Session(graph=main_graph) as sess:
    
    sess.run(tf.global_variables_initializer())

    # How many mini-batches in total?
    num_batches=int(2048/batch_size)
    
    loss=0

    #Iterate over the entire training set for 10 times
    for epoch in range(10):
            
        # Iterate over the number of mini-batches
        for n_batch in range(num_batches-1):
            
            # Get the next mini-batches of samples for the training set
            features, labels=get_next_batch(n_batch)
              
            # Perform the gradient descent step on that mini-batch and compute the loss value
            _, loss_=sess.run((update_step, loss_op), feed_dict={x:features, y:labels})   
            loss+=loss_
             
        print('epoch_nr.: %i, loss: %.3f' %(epoch,(loss/num_batches)))
        loss=0 

    # Compute the prediction on the last mini-batch
    prediction=sess.run(predict_op, feed_dict={x:features})
    
    # Iterate over the features and labels from the last mini-batch as well as
    # the predicitons made by the network, and compare them to check the performance
    for f, l, p in zip(features, labels, prediction):
        
        # Rescale the predictions and labels back into their original value range
        p=p*2047
        l=l*2047
        
        print('Binary number: %s,  label: %i, prediciton %i' %(str(f), l,p))


# ### so after this we will move forward with advance modules of tensorflow
# ### whole purpose of next module is designing ETL (Extract Transformation Load) so that we can fasten training process to next level to increase cpu gpu parallelization

# In[ ]:




