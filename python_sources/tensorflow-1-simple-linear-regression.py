#!/usr/bin/env python
# coding: utf-8

# ## Simple Linear Regression with Tensorflow
# 
# Following is a simple example of performing Linear Regression with Tensorflow.  By following the steps below, we can find parameters such as slope and interecept which are required for making a prediction.
# Tensorflow is a popular Deep Learning framework by Google.

# In[ ]:


import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import tensorflow as tf


# **Generation of Data**
# The data is generated using linspace function of numpy which generates data following a linear pattern. To make things a bit difficult, we add some noise to lineary generated data as well.
# 
# The data follows a linear pattern, which can be represented as:
# > y = mX + b 
# 
# Where, y is the dependant variable,  X is the independent variable and m and b are constants. Our goal is to find m and b so that prediction of unknown points can be performed.
# 

# In[ ]:


x_data = np.linspace(0,10,10000) # X values 
noise = np.random.randn(len(x_data))


# Now we are creating response variable y, which follows a linear relationship with X.  We are also adding some noise (error) to the response, so that we make it tough for algorithm to guess the parameters. As you can see from the equation, slope is kept to 0.5 and intercept is kept to 5. You can choose to take any value as per your liking.

# In[ ]:


y_true = .5 * x_data + 5 + noise # Added Noise, m is taken as .5 and b is 5, these are arbitraty values which can be changed.
x_df = pd.DataFrame(data = x_data, columns = ["X Data"])
y_df = pd.DataFrame(data = y_true, columns = ["Y"])
my_data = pd.concat([x_df, y_df], axis = 1) # DataFrame created with X and Y values.
my_data.sample(250).plot(kind = 'scatter', x = 'X Data', y = 'Y')


# The above graph demonstrates that there is indeed a linear pattern which exists between x and y, even after we have added some noise.

# ## Tensorflow functions ##
# Now we will use Tensorflow related APIs to build the linear model.  **m** and **b** are the ones, which will be found out by the algorithm hence they are kept in th **tf.Variable** container, whereas x and y are placeholders because there values will not change.
# Following steps are done:
# * m and b are initialized. This is the initial model.
# * y_model is determined for above m and b.
# * error is calculated as mean square error.
# * an optimizer - Gradient Descent Optimizer is used to reduce the error.
# * number of batches are defined which are used to train the model a number of times.
# * final value of m and b is extracted.
# 

# In[ ]:


batch_size = 8 # Requirement by Tensorflow for how many records are trained in one go.
rnd = np.random.randn(2)

m = tf.Variable(rnd[0], ) # purely arbitrary values.
b = tf.Variable(rnd[1], )

xph = tf.placeholder(tf.float32, [batch_size]) # placeholder equal to batch size.
yph = tf.placeholder(tf.float32, [batch_size])

y_model = tf.cast(m, tf.float32) * xph + tf.cast(b, tf.float32)

error = tf.reduce_sum(tf.square(y_model - yph)) # Error function, notice that tensorflow functions are used tf.square.
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001) # Gradient descent optimizer is used.
train = optimizer.minimize(error)
init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    batches = 1000 # How many batches to train.
    for i in range(batches):
        rand_ind = np.random.randint(len(x_data), size = batch_size) # Data Selection.
        feed = {xph:x_data[rand_ind], yph:y_true[rand_ind]}
        sess.run(train, feed_dict = feed)
    model_m, model_b = sess.run([m,b])

print(model_m, model_b) # Final params learned


# We see that final values of m and b are very close to the ones which have been used to define the model. Hence the model has performed adequately.

# In[ ]:


y_predicted = model_m * x_data + model_b


# In[ ]:


my_data.sample(250).plot(kind = 'scatter', x = 'X Data', y = 'Y')
plt.plot( x_data, y_predicted, 'r')


# With the help of above plot we see that our model has done a good job of finding the linear equation, which can be defined by two parameters, m and b. 
# **Happy Model Building!**
