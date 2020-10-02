#!/usr/bin/env python
# coding: utf-8

# <center><h1>Artificial Neural Networks for beginner </h1></center>
# **In this chapter we will cover the following topics**
# * Introduction 
# * Artificial Neural Networks.
# * Multi layer Perceptrons.
# * Deep Neural Networks , with implementation from scratch.
# ---------------------------------------------------------------------------------------------------------------------------------
# ![nnw](https://i1.wp.com/thedatascientist.com/wp-content/uploads/2015/12/cool_neural_network.jpeg?resize=870%2C435)
# 
# ---------------------------------------------------------------------------------------------------------------------------------
#  # Neurons
# ![neurons](https://synapseweb.clm.utexas.edu/sites/default/files/styles/os_files_large/public/synapseweb/files/neuron.gif?m=1484252011&itok=3yWNfD_i)
# * A neuron (also called neurone or nerve cell) is a cell that carries electrical impulses. Neurons are the basic units of the nervous system. Every neuron is made of a cell body (also called a soma), dendrites and an axon. Dendrites and axons are nerve fibres.
#    * Neuron cells are located all over the brain.
#    * Neuron's have an **input wire** called as 'Dendrites'   , and a **cell body** where all the processing work is done  ,  and an  **output wire** are called as 'Axon'.
#    * Axon passes the message to other neurons , the neurons are connected to each other by Axon of one neuron to other neurons Dendrite.
#    * And the message is tranfered with an electric pulse.
#  
# * **Auditory Cortex**
#    * The auditory cortex is the part of the temporal lobe that processes auditory information in humans and other vertebrates. It is a part of the auditory system, performing basic and higher functions in hearing, such as possible relations to language switching.
#    * **If we re-wire it to the eye , the Auditory Cortex can learn to see.**
# * **Somatosensory Cortex**
#    * The somatosensory cortex receives all sensory input from the body. Neurons that sense feelings in our skin, pain, visual, or auditory stimuli, all send their information to the somatosensory cortex for processing.
#    * **If we also re-wire it to the eye , the Somatosensory Cortex can learn to see.**
# ---------------------------------------------------------------------------------------------------------------------------------
# 

# # Artificial Neural Network (ANN)
# * Now , that we know how a neuron works in the brain . 
#     * First the input is passed from **Axon** to the **cell body** for some processing and then it is passed to the final output wire **Dendrites**.
# * Artificial Neural Networks also follows the same principle.
# * Let's see the Architecture of an ANN .
# ![anna](https://drive.google.com/uc?export=view&id=1whPd5r8yhJMd-IdMIEcQDDqVZEfsruRQ)
#  * The round shapes are called as **nodes or artificial neurons or neurons**.
#  * The arrows represent the **output from one neuron to the  input of other neuron**.
# 
# * **Input Layer** :- The first layer is called the **input layer** , each node in the input layer represents a **feature variable** and one **bias variable $x_{0}$** . 
# * **Hidden Layer** :- The middle 2 layer's are called the **hidden layer's** , there can be **n** number of hidden layers and **n** number of nodes or neurons in each hidden layer.  We will understand what the hidden layer does later. In the above figure we have 2 hidden layer . **Note**: just like input layer it should contain a bias term , but it should not be connected to previous layer. 
# * **Output Layer** :- The final layer is called the **output layer** , the number of  nodes in the output layer depends upon the number of classes we have to predict , example in **iris** dataset we have 3 target classes , so we will have 3 nodes in the output layer.
# 
# **Now that we know the architecture of Artificial Neural Netwroks , lets understand the working of the ANN.**
# * We will use **Perceptron and Multi-layer Perceptron** to understand the working of ANN, which is very easy to understand .

# # Perceptron 
# * Perceptron is one of the simplest ANN architectures. 
# * It uses **Linear Threshold Unit (LTU)**.
#    * The **LTU** computes a weighted sum of its input i.e $$ z = w_{0}x_{0} +  w_{1}x_{1} +  w_{2}x_{2} + ...... + w_{n}x_{n}$$  **Note** $ w_{0}x_{0}$ , are the bias term.
#    * Then applies a **step function** to that sum and outputs the result : $h_{w}(x) = step(z) = step(w^T \cdotp x)$ 
# ![pann](https://cdn-images-1.medium.com/max/1600/1*n6sJ4yZQzwKL9wnF5wnVNg.png)
# 
# <center><b>(one of the most simplest ANN architecture)</b></center>
# 
# * The most common step function used in Perceptrons is the **Heaviside step function** :
# ![hz](http://math.feld.cvut.cz/mt/txtb/4/gifa4/pc3ba4lc.gif)
# * You can even use**Sign Step Function** :
# ![sf](https://helloacm.com/wp-content/uploads/2016/10/math-sgn-function-in-cpp.jpg)
# 

# * A single **LTU** can be used for **simple linear binary classification**. It computes a linear combination of inputs and if the result exceeds a threshold , it outputs positive class or else the negative class.
# * A Perceptron is simply composed of a single layer of LTU's, with each neuron connected to all the inputs . These connections are often represented using special pass through neurons called input neurons: they jus output whatever the input they are fed , an extra bias feature is generally added ($x_{0} = 1 $). The bias feature is typically represented using a special type of neuron called a bias neuron , which just outputs 1 all the time.
# 
# ![p3](https://drive.google.com/uc?export=view&id=1qgicvYULvDG1RB5on5HKJrzysnUWz8-R)
# 
# * Above is the diagram of perceptron with two inputs and three outputs . This Perceptron can classify instances simultaneously  into three different binary classes , which makes it a multioutput classifier
# 
# **Note : Perceptron does not have  a hidden layer if it has a hidden layer then it is called as Multi Layer Perceptron**

# # Training a Perceptron
# * It calculates the error made by the network; it does not reinforce connections that lead to the wrong output.
# * More specifically , the Perceptron is fed one training instance at a time, and for each instance it makes its predictions .
# * For every output neuron that produced a wrong prediction, it reinforces the connection weights from the inputs that would have contributed to the correct prediction.
# * Perceptron learning rule equation:
# $$w_{i,j}^{(next step)} = w_{i ,j} + \alpha (y_{j} - \hat{y}_{j})x_{i}$$
#    * $w_{i,j}$ is the connection weight between the $i^{th}$ input neuron and the $j^{th}$ output neuron.
#    * $x_{i}$ is the $i^{th}$ input value of the current training instance.
#    * $\hat{j}_{j}$ is the output of the $j^{th}$ output neuron of the current training instance.
#    * $y_{j}$ is the target output (actual output) of the $j^{th}$ output neuron for the current training instance.
#    * $\alpha$ is the learning rate.
# * The decision boundary of each neuron is linear , therefore **Perceptrons are incapable of learning complex patterns**.
# 
# * Scikit - Learn provides a Perceptron class that implements a single LTU network 

# In[ ]:


'''example of perceptron using scikit learn'''
import numpy as np 
from sklearn.datasets import load_iris
from sklearn.linear_model import Perceptron 

iris = load_iris()
X = iris.data[: ,  (2 , 3)] # petal lenght and petal width
y = (iris.target == 0 ).astype(np.int) # binary classification 

classifier = Perceptron(max_iter = 1000 ,tol = 1e-3 ,random_state = 42)
classifier.fit(X , y)
y_pred = classifier.predict([[1.5 , 0.5] , [3 , 3.2]])

print(y_pred)


# # Multi-layer Perceptron 
# * Some of the limitations of Perceptrons can be eliminated by stacking multiple Perceptrons , the resulting **Artificial Neural Network** is called as a **Multi-Layer Perceptron**. It can solve non linear problems.
# * A Multi layer perceptron architecture consists of :
#      * One input layer.
#      * One or more layers of LTU's called hidden layer.
#      * One final layer of LTU's is called output layer.
#      
#      
# # When an ANN has two or more hidden layer's it is known as DEEP NEURAL NETWORK
# 
# ![mlp](https://miro.medium.com/max/804/1*xxZXeKfVKTRqh54t10815A.jpeg)
# 
# * For training MLP's we use **backpropagation training algorithm**.

# # Backpropagation Training algorithm
# * In simple words we can describe Backpropagation training algorithm as **Gradient Decent using reverse-mode autodiff (automatic differentiation)**.
# * # Working of **Backpropagation Training Algorithm** :-
#     * First for each training instances it makes a prediction.(forward pass)
#     * Then it measures the error.
#     * Then goes through each layer in reverse to measure the error contribution from each connection.(reverse pass)
#     * Finally it tweaks the connection weights to reduce the error.
#     
# * **In order for this algorithm to work properly , the authors made a key change to the MLP's architecture** : they replaced the step function with the **Logistic or Sigmoid Function** $$ \sigma (z)  = \frac{1}{1+ e^{-z}} $$
# * This was important because step function contains only flat segments , Gradient descent cannot move on a flat surface.
# * While **Logistic Function** has a non - zero derivate every where, allowing Gradient descent to make some progress at every step.
# 
# ![lgf](https://www.researchgate.net/profile/Farid_Najafi/publication/268874045/figure/fig1/AS:295410389274629@1447442733860/Graph-of-the-Logistic-function-and-its-derivative-function.png)

# * We can use other activation functions to , instead of logistic function .Two other popular activation functions are :
#      * The hyperbolic tangent function : $$ tanh(z) = \frac{e^{z} - e^{-z}}{e^{z} + e^{-z}}$$
#          * just like the **Logistic Function** it is s-shaped , continuous , differentiable , but its output value in the range from -1 to 1 , which tend to make each layers output more or less normalized at the beginning of training, This often helps speed up the convergence.
#      * The ReLU function : $$ ReLU(z) = max(0 , z)$$
#          * It is continuous but not differentiable at  z = 0 , the slope changes abruptly , which can make Gradient Descent bounce around . It works very well and has the advantage of being fast to compute . Most importantly , the fact that it does not have a maximum output value also helps reduce some issues during Gradient Descent.
#          

# * **MLP is often used for classification , with each output corresponding to a different binary class. When the classes are exclusive (e.g  classes 0 to 9 for digits image classification) the output layer is typically modified by replacing the individual activation functions by a shared SOFTMAX FUNCTION**. $$ \sigma(z) = \frac{exp(z)}{\sum_{j = 1}^{K} exp(z)}$$
#   * K is the number of classes.
# * The output of each neuron corresponds to the estimated probability of the corresponding class.
# * Cross Entropy cost function is used to measure loss in Softmax Function : $$j(\theta) = -\frac{1}{m} \sum_{i = 1}^{m} \sum_{k = 1}^{K}  y_{k}^{(i)}  log (\hat{p}_{k}^{(i)})$$ 
#     * $\hat{p}$ is the predictedt probability
# * Cross entropy gradient Vector for class k:
# $$\bigtriangledown_{\theta^{(k)}} j(\theta) = \frac{1}{m} \sum_{i = 1}^{m} (\hat{p}_{k}^{(i)} - y_{k}^{(i)})  x^{(i)} $$
# **Note that the signal flows only one direction , so this architecture is an example of a FEEDFORWARD NEURAL NETWORK(FNN)**
# ![p23](https://drive.google.com/uc?export=view&id=1iascQ7p-8t2WLWebis1BzD4Ga9B2DR-M)

# # Implementing Deep Nerual Network from Scratch using TensorFlow's lower-level API.
# 
# * We will implement **Mini-batch Gradient Descent** to train DNN on Digits dataset.
# * We will divide this implementation into 2 parts :
#    1. **Construction Phase.**
#    2. **Execution Phase.**
#    
# # Construction Phase 
# 
# * First we will import tensroflow library , the Digits dataset from sklearn.
# * Then we will specify the number of inputs and ouputs , and set the number of hidden neurons in each layer
# (architecture : 1 input layer , 2 hidden layer with 300 and 100 nodes each and a output layer)

# In[ ]:


import tensorflow as tf
import matplotlib
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits 

digits = load_digits()
m , n = digits.data.shape

n_inputs = n 
n_hidden_neurons_1 = 300
n_hidden_neurons_2 = 100
n_outputs = 10 # classes 0 to 9


# * Now we will create two **placeholder nodes** to represent training data and targets.

# In[ ]:


X = tf.placeholder(tf.float32 ,shape = (None , n_inputs) , name = 'X')
y = tf.placeholder(tf.int64 , shape = (None) , name = 'y')


# * Creating a function that creates mini - batches of the dataset.

# In[ ]:


def fetch_batch(epoch, batch_index, batch_size):
    np.random.seed(epoch * n_batches + batch_index)      
    indices = np.random.randint(digits.data[:1700].shape[0] , size = batch_size)  
    X_batch = digits.data[:1700][indices] 
    y_batch = digits.target[:1700][indices] 
    return X_batch, y_batch


# * **Now , we need to create two hidden layers and the output layer.**
# * We will use ReLU activation function for hidden layers.
# * Softmax activation function for output layer.
# 
# **Let's create a neuron_layer() function that we will use to create one layer at a time. Parameters will be the inputs , the number of neurons a,  the activation function and the name of the layer**

# In[ ]:


def neuron_layer(X , n_neurons , name , activation = None):
    with tf.name_scope(name):
        n_inputs = int(X.get_shape()[1])
        stddev = 2 /np.sqrt(n_inputs)
        '''initializing weights randomly , using truncated normal distribution
        with standard deviation of 2/sqrt(n_inputs)'''
        init = tf.truncated_normal((n_inputs , n_neurons) , stddev = stddev)
        W = tf.Variable(init , name = 'weights')
        '''biases'''
        b = tf.Variable(tf.zeros([n_neurons]) , name = 'bias')
        '''weighted sum'''
        Z = tf.matmul(X , W) + b
        '''activation function'''
        if activation is not None:
            return activation(Z)
        else:
            return Z


# **Now lets use neuron_layer() function to create Deep Neural Network**
# * The first hidden layer will take X as input 
# * The second hidden layer will take the output of first hidden layer as its input 
# * The output layer takes the output of the second hidden layer as its input

# In[ ]:


with tf.name_scope('dnn'):
    hidden_layer_1 = neuron_layer(X = X , n_neurons = n_hidden_neurons_1 ,
                                  name = 'hiddenLayer1' , activation = tf.nn.relu)
    hidden_layer_2 = neuron_layer(X = hidden_layer_1 , n_neurons = n_hidden_neurons_2 ,
                                 name = 'hiddenLayer2' , activation = tf.nn.relu)
    logits = neuron_layer(X = hidden_layer_2 , n_neurons = n_outputs , name = 'outputs')


# **Note logits is the output of the nn before going through the softmax activation 
# funtion. we will handle softmax computation futher**

# **Now that we have the neural network ready , we need to define the cost function that we will use to train the network.**
# * We are going to use **cross entropy cost function** to train the neural network.
# * Cross entropy will penalize models that estimate a low probability for the target class.
# 
# **TensorFlow provides several functions to compute cross entropy , we will use sparse_soft_max_cross_entropy_with_logits()**
# * It computes the cross entropy based on the **logits** and it expects labels in the form of integers ranging from 0 to the number of classes minus 1 , in our case , from 0 to 9.
# * It will give us a 1D tensor containing the cross entropy for each instance. We can then use TensorFlow's **reduce_mean()** function to compute the mean cross entropy over all instances.

# In[ ]:


with tf.name_scope('loss'):
    xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels = y , 
                                                              logits = logits)
    loss = tf.reduce_mean(xentropy , name = 'loss')


# * **Now  we need to define a GradientDescentOptimizer that will tweak the model parameters to minimize the cost function.**

# In[ ]:


learning_rate = 0.01 
with tf.name_scope('train'):
    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    training_op = optimizer.minimize(loss)


# **The following code , is used for evaluation of the model.**

# In[ ]:


with tf.name_scope('eval'):
    correct = tf.nn.in_top_k(logits , y , 1)
    accuracy = tf.reduce_mean(tf.cast(correct , tf.float32))


# **The Construction phase is done , how lets create a node to initialize all the variable and we  will also create a saver to save our trained model parameters to disk**

# In[ ]:


init = tf.global_variables_initializer()
saver = tf.train.Saver()


# # Execution Phase

# In[ ]:


n_epochs = 1000
batch_size = 50 
n_batches = int(np.ceil(digits.data.shape[0] / batch_size))

with tf.Session() as session:
    init.run()
    for epoch in range(n_epochs):
        
        for batch_index in range(n_batches):
            X_batch , y_batch = fetch_batch(epoch , batch_index  , batch_size)
            session.run(training_op , feed_dict = {X : X_batch , y : y_batch})
            
        acc_train = accuracy.eval(feed_dict = {X : X_batch , y : y_batch})
        acc_test = accuracy.eval(feed_dict = {X : digits.data[1700:] , y : digits.target[1700:]})
        if epoch % 100 == 0 :
            print('epoch : {0}, Train Acc : {1}  , Test Acc : {2}'.format(epoch ,acc_train , acc_test))
    save_path = saver.save(session , './model_final.ckpt') 


# ![epochvserro](https://drive.google.com/uc?export=view&id=1wkIVHHvzaEmezCYf8lLaqUCe8sgLJxPY)
# * The following code  , opens a TensorFlow sesssion , and it runs the init node , that initializes all the variables.
# * Then it runs the main training loop : at the each epoch , the code iterates through a number of mini - batches that corresponds t the training set size
# * Each mini- batch is fetched from the fetch_batch() function , then the code simply runs  the training operation , feeding it the current mini - batch input data and targets.
# * Next , at the end of each epoch , the code evaluates the mode on the last mini batch and on the full test set , and it prints out the result, Finally the model parameters are saved to the disk

# **now lets predict on unseen data.**

# In[ ]:


with tf.Session() as session:
    saver.restore(session , './model_final.ckpt')
    X_test = digits.data[1780:]
    Z = logits.eval(feed_dict = {X : X_test})
    y_pred = np.argmax(Z , axis = 1)

plt.figure(1 , figsize = (15 , 10))
for n in np.arange(1 , 17):
    plt.subplot(4 , 4 , n )
    plt.subplots_adjust(hspace = 0.3 , wspace=0.3)
    plt.imshow(digits.data[1780:][n].reshape(8 , 8)  ,cmap = matplotlib.cm.binary , interpolation="sinc")
    x_l = "True : {0} , Predicted : {1}".format(digits.target[1780:][n] , y_pred[n])
    plt.xlabel(x_l)
    
plt.show()


# # Implementation of DNN with TensorFlow's High-Level API

# In[ ]:


'''creating an input function which returns X , y'''
'''reason to create this input function is because tf.estimator.DNNClassifier() class
methods train() , eval() and predict() needs a function in the parameter which returns 
X and y'''
def input(df):
    return df.data[:1789] , df.target[:1789].astype(np.int32)

'''tf.feature_column are used to convert some sort of input data feature into continuous 
variables that can be used by a regression or neural network model.'''
'''In the case of a numeric_column, there is no such conversion needed,
so the class basically just acts like a tf.placeholder.'''
feature_col = [tf.feature_column.numeric_column('x' , shape = [8 , 8])] #shape = (8,8) cause digits data images are of 64 pixel


'''creating architecture of DNN with two hidden layer with 300 and 100 neurons 
respectively and a softmax output layer with 10 neurons'''
dnn_clf = tf.estimator.DNNClassifier(hidden_units = [200 , 100] ,
                                     n_classes = 10  ,
                                     feature_columns = feature_col 
                                    ) 


'''Defining the training inputs'''
input_fn_train = tf.estimator.inputs.numpy_input_fn(
    x={"x": input(digits)[0]}, # X
    y=input(digits)[1], # y
    num_epochs=None,
    batch_size=50,
    shuffle=True # randomness
)

'''training the classifier'''
dnn_clf.train(input_fn = input_fn_train , steps = 1000 )

'''Defining the Eval inputs'''
input_fn_eval = tf.estimator.inputs.numpy_input_fn(
    x={"x": input(digits)[0]},
    y=input(digits)[1],
    num_epochs=1,
    shuffle=False
)

'''evaluating'''
metrics = dnn_clf.evaluate(input_fn=input_fn_eval , steps=10)

'''Defining the predict inputs'''
input_fn_predict = tf.estimator.inputs.numpy_input_fn(x = {"x" : digits.data[1788:]} , 
                                                          num_epochs = 1 , 
                                                          shuffle = False
                                                         )
'''predicting'''
predictions = dnn_clf.predict(input_fn= input_fn_predict)

'''Note you should always split the data into train , eval and test data which i
have not done , because this is just an example.'''


# In[ ]:


'''getting the predicted classes'''
cls = [p['classes'] for p in predictions]
'''converting into int array'''
y_pred = np.array(cls , dtype = 'int').squeeze()

'''ploting true value and predicted value'''
plt.figure(1 , figsize = (15 , 10))
for n in np.arange(1 , 9):
    plt.subplot(4 , 4 , n )
    plt.subplots_adjust(hspace = 0.3 , wspace=0.3)
    plt.imshow(digits.data[1788:][n].reshape(8 , 8)  ,cmap = matplotlib.cm.binary , interpolation="sinc")
    x_l = "True : {0} , Predicted : {1}".format(digits.target[1788:][n] , y_pred[n])
    plt.xlabel(x_l)
    
plt.show()

