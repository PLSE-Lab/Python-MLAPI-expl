#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import warnings
warnings.filterwarnings('ignore')


# In[ ]:


# Import tensorflow 
import tensorflow as tf
print("version: "+tf.__version__)


# In[ ]:


# lets see Hello world example
hello = tf.constant("Hello ")
world = tf.constant("world!")


# In[ ]:


# lets print
print(hello+world)


# In[ ]:


# lets have a look on type of hello and world variables
print(type(hello))
print(type(world))


# ![](http://)In order to print **hello world!** we need tf session

# In[ ]:


with tf.Session() as sess:
    print( sess.run(hello+world) ) # concatenation


# In[ ]:


# lets play with numbers
a = tf.constant(5)
b = tf.constant(10)
type(a)


# In[ ]:


with tf.Session() as sess:
    print( sess.run(a+b) )


# #### Lets see how matrix will be declared 
#     Explore matrix functions by placing curser on tf and (shift+tab)

# In[ ]:


# A 4X4 Matrix with all values 10
matA = tf.fill((4,4),10) 
with tf.Session() as sess:
    print( sess.run(matA) )


# In[ ]:


# Zero values matrix
matB = tf.zeros((2,2))
with tf.Session() as sess:
    print( sess.run(matB) )


# In[ ]:


# Normal distribution Matrix
matN = tf.random_normal((2,2),mean=0,stddev=1.0)

with tf.Session() as sess:
    print(sess.run(matN))


# In[ ]:


# Uniform Random Distribution
matU = tf.random_uniform((2,2),minval=10,maxval=100)
with tf.Session() as sess:
    print(sess.run(matU))


# ### To execute tensors we can also create interactive session
# #### tf.InteractiveSession()
# This is the exact same as tf.Session() but is targeted for using IPython and Jupyter Notebooks that allows you to add things and use Tensor.eval() and Operation.run() instead of having to do Session.run() every time you want something to be computed.

# In[ ]:


ISess = tf.InteractiveSession()

print( ISess.run(tf.zeros((5,5)) ) )
# or
ones = tf.ones((5,5))
print( ones.eval() ) 


# In[ ]:


# Simple Matrix example
matS1 = tf.constant([[10,5],
                    [3, 9]  ])
matS2 = tf.constant([[10,5],
                    [3, 9]  ])

print( matS1.eval() )
print( matS1.get_shape())

print(tf.matmul(matS1,matS2).eval())


# #### Graphs in Tensorflow
# A tf.Graph object defines a namespace for the tf.Operation objects it contains.
# 
# A tf.Graph contains two relevant kinds of information:
# 
# **Graph structure** The nodes and edges of the graph, indicating how individual operations are composed together, but not prescribing how they should be used. The graph structure is like assembly code: inspecting it can convey some useful information, but it does not contain all of the useful context that source code conveys.
# 
# **Graph collections** TensorFlow provides a general mechanism for storing collections of metadata in a tf.Graph. The tf.add_to_collection function enables you to associate a list of objects with a key (where tf.GraphKeys defines some of the standard keys), and tf.get_collection enables you to look up all objects associated with a key. Many parts of the TensorFlow library use this facility: for example, when you create a tf.Variable, it is added by default to collections representing "global variables" and "trainable variables". When you later come to create a tf.train.Saver or tf.train.Optimizer, the variables in these collections are used as the default arguments.

# In[ ]:


tf.get_default_graph


# In[ ]:


print( tf.constant(0) )
print( tf.constant(0, name="c") )


# In[ ]:


# Other than default graph
d = tf.get_default_graph
g = tf.Graph()
g


# In[ ]:


g is tf.get_default_graph


# In[ ]:


d is tf.get_default_graph


# #### Variables
# A TensorFlow variable is the best way to represent shared, persistent state manipulated by your program.
# Variables are manipulated via the **tf.Variable class**
# 
# Variable needs to be initialized.
# 
# Can hold the values of weights and biases through out the session. 

# In[ ]:


tensorVar = tf.Variable( initial_value=tf.zeros((2,2)) )

# initialize the variables
init = tf.global_variables_initializer()
init.run()

print( tensorVar )
print(tensorVar.eval())


# #### Placeholders
# They are initially empty and used to feed in the actual training examples. But they need to be declared with datatype and expected shape.

# In[ ]:


ph = tf.placeholder(tf.float32) 
ph


# In[ ]:


matph1 = tf.placeholder(tf.int32, shape=(3,3))
matph2 = tf.placeholder(tf.int32, shape=(None,3))
# None is for no. of rows or examples in dataset

print(matph1)
print(matph2)


# ### Basic Neural Network with Tensorflow 

# In[ ]:


import numpy as np #for linear algebra 
np.random.seed(101) # It can be called again to re-seed the generator
tf.set_random_seed(101)


# In[ ]:


rand_a = np.random.uniform(0,100,(5,5))
rand_b = np.random.uniform(0,100,(5,1))

a = tf.placeholder(tf.float32)
b = tf.placeholder(tf.float32)

add_op = a+b # tf.add(a,b)
mult_op = a*b #tf.multiply(a,b)


# **Running Sessions to create Graphs with Feed Dictionaries**

# In[ ]:


with tf.Session() as sess:
    s = sess.run( add_op, feed_dict={ a:rand_a, b:rand_b} )
    print(s)
    
    m = sess.run( mult_op, feed_dict={a:rand_a, b:rand_b } )
    print(m)


# In[ ]:


n_features = 10
n_dense_neurons = 3


# In[ ]:


# Placeholder for x
x = tf.placeholder(tf.float32,(None,n_features))

# Variables for w and b
b = tf.Variable(tf.zeros([n_dense_neurons]))

W = tf.Variable(tf.random_normal([n_features,n_dense_neurons]))


# In[ ]:


# Activation func
xW = tf.matmul(x,W)
z = tf.add(xW,b)
# tf.nn.relu() or tf.tanh()
a = tf.sigmoid(z)

# Variable Intialization
init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    print(sess.run(z,feed_dict={x : np.random.random([1,n_features])}))
    layer_out = sess.run(a,feed_dict={x : np.random.random([1,n_features])})

print(layer_out)


# #### Full Neural Network Example
# **${y = mx + b}$**
# 

# In[ ]:


np.linspace(1,10,10)


# In[ ]:


# Artificial Data
xdata = np.linspace(0,10,10) + np.random.uniform(-1.5,1.5,10)
print(xdata)
ydata = np.linspace(0,10,10) + np.random.uniform(-1.5,1.5,10)
print(ydata)


# In[ ]:


m = tf.Variable(0.39)
b = tf.Variable(0.2)


# In[ ]:


# Cost or Residual 

error = 0
for x,y in zip(xdata,ydata):
    y_ = m*x + b
    error += (y-y_)**2

print(error)


# In[ ]:


# optimizer
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
train = optimizer.minimize(error)
train


# In[ ]:


init = tf.global_variables_initializer()

with tf.Session() as sess:
    
    sess.run(init)
    
    epochs = 100
    
    for i in range(epochs):
        
        sess.run(train)
        

    # Fetch Back Results
    final_slope , final_intercept = sess.run([m,b])


# In[ ]:


print(final_slope)
print(final_intercept)


# In[ ]:


import matplotlib.pyplot as plt

x_test = np.linspace(-1,11,10)
y_pred_plot = final_slope*x_test + final_intercept

plt.plot(x_test,y_pred_plot,'r')

plt.plot(xdata,ydata,'*')
plt.show()

