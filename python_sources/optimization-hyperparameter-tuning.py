#!/usr/bin/env python
# coding: utf-8

# ### Gradient Descent with Momentum
# - Gradient descent with momentum is an improvised version of minibatch gradient descent.
# - This method accelerates the gradient descent steps much faster by eliminating unwanted oscillations.
# - In other words, it smoothens the path taken by gradient descent towards the global minimum.
# - This method makes use of the concept called exponentially weighted moving average when performing parameter update during gradient descent.
# 
# ### RMS Prop
# - Root Mean Square propagation is another technique to boost the gradient descent.
# - This technique aims to damp out oscillation perpendicular to the path of gradient descent and at the same time increase the step size in the direction of the global minimum.
# 
# ### Adam Optimizer
# - Adam optimization is nothing but the combination of gradient descent with momentum and RMS prop.
# - It incorporates the advantages of both of these algorithms by dividing the moving average by the root mean square of derivates
# 
# ## Intro
# - In this handson you will be using the concept of GD with momentum, RMS prop and Adam prop to build optimized deep neural network
# - You will also be implementing minibatch gradient and L2 regularization to train you network
# - Follow the instruction provided for cell to write the code in each cell.
# - Run the below cell for to import necessary packages to read and visualize data

# The data is provided as file named 'data.csv'.  
# Using pandas read the csv file and assign the resulting dataframe to variable 'data'   
# for example if file name is 'xyz.csv' read file as **pd.read_csv('xyz.csv')** 

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors


# The data is provided as file named 'blobs.csv'.  
# Using pandas read the csv file and assign the resulting dataframe to variable 'data'   
# for example if file name is 'xyz.csv' read file as **pd.read_csv('xyz.csv')** 

# In[ ]:



data = pd.read_csv('../input/data.csv')

print(data.head())
print(data['class'].unique())


#  Extract all the feature values from dataframe 'data' and assign it to variable 'X'
# - Extract target variable 'class' and assign it to variable 'y'.  
# Hint:
#  - Use .values to exract values from dataframe

# In[ ]:



X = data.loc[:, data.columns != 'class'].values
y = data['class'].values
###End code

assert X.shape == (10000, 10)
assert y.shape == (10000, )


# - Run the below cell to visualize the data in x-y plane. 
# - The green spots corresponds to target value 0 and green spots corresponds to target value 1
# - Though the data is more than 2 dimension only first two features are considered for visualization

# In[ ]:


colors=['green','blue']
cmap = matplotlib.colors.ListedColormap(colors)
#Plot the figure
plt.figure()
plt.title('Non-linearly separable classes')
plt.scatter(X[:,0], X[:,3], c=y,
           marker= 'o', s=50,cmap=cmap,alpha = 0.5 )
plt.show()


# In[ ]:


from pandas.plotting import scatter_matrix
get_ipython().run_line_magic('matplotlib', 'inline')
color_wheel = {0: "#0392cf", 
               1: "#7bc043", 
            }

colors_mapped = data["class"].map(lambda x: color_wheel.get(x))

axes_matrix = scatter_matrix(data.loc[:, data.columns != 'class'], alpha = 0.2, figsize = (10, 10), color=colors_mapped )


# In[ ]:





# - In order to feed the network the input has to be of **shape (number of features, number of samples)** and target should be of shape **(1, number of samples)**
# - Transpose X and assign it to variable 'X_data'
# - reshape y to have shape (1, number of samples) and assign to variable 'y_data'

# In[ ]:


X_data = X.T
y_data = y.reshape(1,len(y))

assert X_data.shape == (10, 10000)
assert y_data.shape == (1, 10000)


# Define the network dimension to have **10** input features, **two** **hidden layers** with **9** nodes each, one output node at final layer. 

# In[ ]:


layer_dims = [10,9,9,1]


# import tensorflow as tf

# In[ ]:


import tensorflow as tf


# Define a function named placeholders to return two placeholders one for input data as A_0 and one for output data as Y.
# - Set the datatype of placeholders as float64
# - parameters - num_features
# - Returns - A_0 with shape (num_feature, None) and Y with shape(1,None)

# In[ ]:


def placeholders(num_features):
    A_0 = tf.placeholder(dtype = tf.float64, shape = ([num_features,None]))
    Y = tf.placeholder(dtype = tf.float64, shape = ([1,None]))
    return A_0,Y


# define function named initialize_parameters_deep() to initialize weights and bias for each layer
# - Use tf.random_normal() to initialise weights and tf.zeros() to initialise bias. Set datatype as float64
# - Parameters - layer_dims
# - Returns - dictionary of weights and bias

# In[ ]:


def initialize_parameters_deep(layer_dims):
    tf.set_random_seed(1)
    L = len(layer_dims)
    parameters = {}
    for l in range(1,L):
        parameters['W' + str(l)] = tf.get_variable("W" + str(l), shape=[layer_dims[l], layer_dims[l-1]], dtype = tf.float64,
                                   initializer=tf.random_normal_initializer())
                                   
        parameters['b' + str(l)] = tf.get_variable("b"+ str(l), shape = [layer_dims[l], 1], dtype= tf.float64, initializer= tf.zeros_initializer() )
        
    return parameters 


# Define functon named linear_forward_prop() to define forward propagation for a given layer.
# - parameters: A_prev(output from previous layer), W(weigth matrix of current layer), b(bias vector for current layer),activation(type of activation to be used for out of current layer)  
# - returns: A(output from the current layer)
# - Use relu activation for hidden layers and for final output layer return the output unactivated i.e if activation is sigmoid

# In[ ]:


def linear_forward_prop(A_prev,W,b, activation):
    Z = tf.add(tf.matmul(W, A_prev), b)
    if activation == "sigmoid":
        A = Z
    elif activation == "relu":
        A = tf.nn.relu(Z)
    return A


# Define forward propagation for entire network as l_layer_forward()
# - Parameters: A_0(input data), parameters(dictionary of weights and bias)
# - returns: A(output from final layer)  

# In[ ]:


def l_layer_forwardProp(A_0, parameters):
    A = A_0
    L = len(parameters)//2
    for l in range(1,L):
        A_prev = A
        A = linear_forward_prop(A_prev,parameters['W' + str(l)],parameters['b' + str(l)], "relu")     
        #call linear forward prop with relu activation
    A = linear_forward_prop(A, parameters['W' + str(L)], parameters['b' + str(L)], "sigmoid" )                  
    #call linear forward prop with sigmoid activation
    
    return A


# - Define the cost function
# - parameters:
#   - Z_final: output fro final layer
#   - Y: actual output
#   - parameters: dictionary of weigths and bias
#   - regularization : boolean
#   - lambd: regularization parameter
# - First define the original cost using tensoflow's sigmoid_cross_entropy function
# - If **regularization == True** add regularization term to original cost function

# In[ ]:


def final_cost(Z_final, Y , parameters, regularization = False, lambd = 0):
    cost = tf.nn.sigmoid_cross_entropy_with_logits(logits=Z_final,labels=Y)
    if regularization:
        reg_term = 0
        L = len(parameters)//2
        for l in range(1,L+1):
            
            reg_term +=  tf.nn.l2_loss(parameters['W'+str(l)])              #add L2 loss term
            
        cost = cost + (lambd/2) * reg_term
    return tf.reduce_mean(cost)


# Define the function to generate mini-batches.

# In[ ]:


import numpy as np
def random_samples_minibatch(X, Y, batch_size, seed = 1):
    np.random.seed(seed)
    
    m =  X.shape[1]                                          #number of samples
    num_batches = int(m / batch_size )                               #number of batches derived from batch_size
    
    indices =  np.random.permutation(m)                                 # generate ramdom indicies
    shuffle_X = X[:,indices]
    shuffle_Y = Y[:,indices]
    mini_batches = []
    
    #generate minibatch
    for i in range(num_batches):
        X_batch = shuffle_X[:,i * batch_size:(i+1) * batch_size]
        Y_batch = shuffle_Y[:,i * batch_size:(i+1) * batch_size]
        
        assert X_batch.shape == (X.shape[0], batch_size)
        assert Y_batch.shape == (Y.shape[0], batch_size)
        
        mini_batches.append((X_batch, Y_batch))
    
    #generate batch with remaining number of samples
    if m % batch_size != 0:
        X_batch = shuffle_X[:, (num_batches * batch_size):]
        Y_batch = shuffle_Y[:, (num_batches * batch_size):]
        mini_batches.append((X_batch, Y_batch))
    return mini_batches


# In[ ]:


def model(X_train,Y_train, layer_dims, learning_rate, optimizer ,num_iter, mini_batch_size):
    tf.reset_default_graph()
    num_features, num_samples = X_train.shape
    
    A_0, Y = placeholders(num_features)
    #call placeholder function to initialize placeholders A_0 and Y
    parameters =  initialize_parameters_deep(layer_dims)                   
    #Initialse Weights and bias using initialize_parameters
    Z_final = l_layer_forwardProp(A_0, parameters)                      
    #call the function l_layer_forwardProp() to define the final output
    
    cost =  final_cost(Z_final, Y , parameters, regularization = True)
    #call the final_cost function with regularization set TRUE
    
    
    
    if optimizer == "momentum":
        train_net = tf.train.MomentumOptimizer(learning_rate, momentum=0.9).minimize(cost)                 
        #call tensorflow's momentum optimizer with momentum = 0.9
    elif optimizer == "rmsProp":
        train_net = tf.train.RMSPropOptimizer(learning_rate, decay=0.999).minimize(cost)
                   
        #call tensorflow's RMS optimiser with decay = 0.999
    elif optimizer == "adam":
        train_net = tf.train.AdamOptimizer(learning_rate, beta1=0.9, beta2=0.999).minimize(cost)                 
        ##call tensorflow's adam optimizer with beta1 = 0.9, beta2 = 0.999
    
    seed = 1
    num_minibatches = int(num_samples / mini_batch_size)
    init = tf.global_variables_initializer()
    costs = []
    with tf.Session() as sess:
        sess.run(init)
        for epoch in range(num_iter):
            epoch_cost = 0
            
            mini_batches = random_samples_minibatch(X_train, Y_train, mini_batch_size, seed)
            #call random_sample_minibatch to return minibatches
            
            seed = seed + 1
            
            #perform gradient descent for each mini-batch
            for mini_batch in mini_batches:
                
                X_batch, Y_batch = mini_batch            #assign minibatch
                
                _,mini_batch_cost = sess.run([train_net, cost], feed_dict={A_0: X_batch, Y: Y_batch})
                epoch_cost += mini_batch_cost/num_minibatches
            
            if epoch % 2 == 0:
                costs.append(epoch_cost)
            if epoch % 10 == 0:
                print(epoch_cost)
        with open("output.txt", "w+") as file:
            file.write("%f" % epoch_cost)
        plt.ylim(0 ,2, 0.0001)
        plt.xlabel("epoches per 2")
        plt.ylabel("cost")
        plt.plot(costs)
        plt.show()
        params = sess.run(parameters)
    return params


# Call the method model_with_minibatch() with learning rate 0.001, **optimizer = momentum** num_iter = 100 and minibatch 256

# In[ ]:


params_momentum = model(X_data,y_data, layer_dims, learning_rate=0.001, optimizer='momentum' ,num_iter=100, mini_batch_size=256)


# Call the method model_with_minibatch() with learning rate 0.001, **optimizer = rmsProp** num_iter = 100 and minibatch 256

# In[ ]:


params_momentum = model(X_data,y_data, layer_dims, learning_rate=0.001, optimizer='rmsProp' ,num_iter=100, mini_batch_size=256)


# Call the method model_with_minibatch() with learning rate 0.001, **optimizer = adam** num_iter = 100 and minibatch 256

# In[ ]:


params_momentum = model(X_data,y_data, layer_dims, learning_rate=0.001, optimizer='adam' ,num_iter=100, mini_batch_size=256)

