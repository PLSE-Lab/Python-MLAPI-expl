#!/usr/bin/env python
# coding: utf-8

# # Tensorflow: Create your first regression model @ TensorFlow!!
# 
# With simple examples, here I am here trying to elaborate the workflow for creating a basic tensorflow regression model -
# * Manual: without using tensorflow estimator APIs
# * TF Estimator: with tensorflow estimator APIs
# 
# ***
# 
# # Regression Example (manual)
# 
# Our first **Objective** is to create a regression model without using tensorflow estimator APIs. So we will be creating some dummy data which has some linear trend, and will fit a regression line over it. We would be creating all this using tensorflow objects and operation graphs.

# ## 1. Import the libraries/modules

# In[ ]:


import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
# This is important to display plots inline in notebook

import tensorflow as tf


# ## 2. Data Generation

# In[ ]:


# create x data with one feature
x_data = np.linspace(0.0, 10.0, 1000000)
x_data


# In[ ]:


# Some noise to be added in the data. Same data points as x data
noise = np.random.randn(len(x_data))
noise


# We have the x feature now, and to fit below regression formula we need corresponding slope and intercept.
# 
# y = mx + b
# 
# Let's assume slope & intercept as a random value, and add some noise to it. 
# 
# * m = 0.5
# * b = 5
# 
# This will give us the y labels for this regression problem. Now we need to fit the regression line to find the slope & intercept as close as possible to above assumed values.

# In[ ]:


y_true = (0.5 * x_data) + 5 + noise 
y_true


# In[ ]:


# Merge data to create pd dataframe
x_df = pd.DataFrame(data=x_data, columns=["X_Data"])
y_df = pd.DataFrame(data=y_true, columns=["Y"])

my_data = pd.concat([x_df, y_df], axis = 1)
my_data.head()


# ### Let's visualize our data by creating a plot.
# 
# But plotting a million data point may freeze out our notbook kernal here, so it's better to visualize some sample data. We're visualizing 250 random data points here to see the trend.

# In[ ]:


my_data.sample(250).plot(x = "X_Data", y = "Y", kind = "scatter")


# ## 3. Fit the regression line
# 
# We want tensorflow to the fit the regression line here which can be used for prediction. Let's see how to make this happen.
# 
# First point is that we have a million data points. More data is better for trainnig complex models, but we can't just feed a million sample to neural network. We rather need to create batches of data, and feed one batch at a time.

# In[ ]:


# We are here taking a batch size of 8.
# There is no good, bad or optimal batch size defined, it depends on the size of data you're dealing with.
batch_size = 8


# In[ ]:


# Take two random numbers to initiate the slope and intercept variables.
# Numbers really doesn't matter here, as we're going to imrove these using grdient descent method.
np.random.randn(2)


# In[ ]:


m = tf.Variable(0.63)
b = tf.Variable(0.61)


# #### Create placeholders:
# 
# Once variables are created for slope and intercept, we need to create the placeholders for x and y features for creating operation graphs for training. 
# 
# x & y will be placeholders not variables because we would be feeding various data points at later stage.

# In[ ]:


# We're sure here of the size of the placeholder. 
# It would contain 8 data points at a time for this example, as our batch size is 8.
xph = tf.placeholder(tf.float32, [batch_size])
yph = tf.placeholder(tf.float32, [batch_size])


# >Workflow:
# 1. create variables
# 2. create placeholders
# 3. define operation/graph

# In[ ]:


# Create model/operation graph
y_model = m * xph + b


# In[ ]:


# Loss function - sum of squared errors
error = tf.reduce_sum(tf.square(yph - y_model))


# In[ ]:


# Create optimizer
# Creating a Gradient Descent Optimizer to train and minimize the error.
optimizer = tf.train.GradientDescentOptimizer(learning_rate = 0.001)
train = optimizer.minimize(error)


# In[ ]:


# initiate the global initializer
init = tf.global_variables_initializer()


# In[ ]:


#Execute the operation graphs now
with tf.Session() as sess:
    
    sess.run(init)
    
    # No of epocs/no of batches to train upon
    batches = 1000
    
    for i in range(batches):
        
        # create random indices to sample out from training data
        # Creating integers upto number of rows in training data and then sampling out batch of 8 random indices
        rand_index = np.random.randint(len(x_data), size = batch_size)
        
        # Create feed dictionary
        feed = {xph: x_data[rand_index], yph: y_true[rand_index]}
        
        # run train optimizer operation graph
        sess.run(train, feed_dict = feed)
        
    model_m, model_b = sess.run([m,b])
        


# In[ ]:


model_m


# In[ ]:


model_b


# **Here, we may see improved results if retrain with higher number of batches.**
# 
# Though optimal epoch may vary for different datasets/samples.

# In[ ]:


#Execute the operation graphs now
with tf.Session() as sess:
    
    sess.run(init)
    
    # No of epocs/no of batches to train upon
    batches = 5000
    
    for i in range(batches):
        
        # create random indices to sample out from training data
        # Creating integers upto number of rows in training data and then sampling out batch of 8 random indices
        rand_index = np.random.randint(len(x_data), size = batch_size)
        
        # Create feed dictionary
        feed = {xph: x_data[rand_index], yph: y_true[rand_index]}
        
        # run train optimizer operation graph
        sess.run(train, feed_dict = feed)
        
    model_m, model_b = sess.run([m,b])
        


# In[ ]:


model_m


# In[ ]:


model_b


# We can try predicting corresponding y values for our x data using above coefficients:

# In[ ]:


y_hat = x_data * model_m + model_b


# ### Evaluate:
# 
# Visualize the prediction line vs actuals

# In[ ]:


my_data.sample(250).plot(x = "X_Data", y = "Y", kind = "scatter")
plt.plot(x_data, y_hat, 'r')


# ***
# ***
# 
# 
# ## Regression Example (TF Estimator)
# 
# 

# To use estimator APIs we do following:
# * Define features
# * Create estimator model
# * Create data input function
# * train, evaluate & predict methods on estimator object.
# 
# Our manual example above was to understand the workflow of creating a regression model using tensorflow objects. However doing same with tensorflow estimator APIs are lot easier to use.

# ### Define feature

# In[ ]:


feature_col = [tf.feature_column.numeric_column('x', shape = [1])]


# ### Create estimator

# In[ ]:


estimator = tf.estimator.LinearRegressor(feature_columns = feature_col)

# Ignore the warning below


# In[ ]:


# Train-test split the data
from sklearn.model_selection import train_test_split


# In[ ]:


x_train, x_test, y_train, y_test = train_test_split(x_data, y_true,
                                                   test_size = 0.3, random_state = 123)


# In[ ]:


# Check out the shape for train and test datasets
x_train.shape


# In[ ]:


x_test.shape


# ### Create input function

# In[ ]:


# This can take inputs from both numpy arrays and pandas dataframes
# input_fns = tf.estimator.inputs.pandas_input_fn()  # Example

# As we are using numpy arrays, so we'll be using below input fn
input_fns = tf.estimator.inputs.numpy_input_fn({'x': x_train}, y_train,
                                              batch_size = 8, num_epochs = None, shuffle = True)


# **We need to create two more input function for evaluation:**

# In[ ]:


train_input_fns= tf.estimator.inputs.numpy_input_fn({'x': x_train}, y_train,
                                              batch_size = 8, num_epochs = 1000, shuffle = False)


# In[ ]:


test_input_fns = tf.estimator.inputs.numpy_input_fn({'x': x_test}, y_test,
                                              batch_size = 8, num_epochs = 1000, shuffle = False)


# ### Train the estimator:

# In[ ]:


estimator.train(input_fn=input_fns, steps=1000)


# ### Evaluation Matrix:
# 
# We have trained the estimator, now it's time to create evaluation matrix. Tensorflow estomator API also have such methods.

# In[ ]:


train_eval_matrix = estimator.evaluate(input_fn=train_input_fns, steps=1000)


# In[ ]:


test_eval_matrix =  estimator.evaluate(input_fn=test_input_fns, steps=1000)


# In[ ]:


print("Training data matrix:")
print(train_eval_matrix)


# In[ ]:


print("Test data matrix:")
print(test_eval_matrix)


# > A high variance in train and test evaluation matrix is good indicator of overfitting. Both being close is good sign a model being well trained, however it does not correlates to accuracy of the model but capability to predict over unseen data.

# ### Predict:
# 
# How to use this model to predict on new data?

# In[ ]:


new_data = np.linspace(0,10, 10)
new_data


# So we have the new data which our model has never seen before.

# In[ ]:


predict_input_fn = tf.estimator.inputs.numpy_input_fn({'x': new_data}, shuffle=False)


# In[ ]:


estimator.predict(input_fn=predict_input_fn)


# *If you noticed above output, it returned a generator object. So to extract the values from it we can either convert it to a list or iterate through it.*

# In[ ]:


list(estimator.predict(input_fn=predict_input_fn))


# In[ ]:


y_pred = []

for y_hat in estimator.predict(input_fn=predict_input_fn):
    y_pred.append(y_hat['predictions'])
    
y_pred


# ### Visualize the predictions vs actuals

# In[ ]:


my_data.sample(n = 250).plot(kind = 'scatter',  x = 'X_Data', y = 'Y')
plt.plot(new_data, y_pred, 'r')


# So we can see here that our predictions are quite inlined with training data, and the coefficients are also been captured close to our assumed values with a basic regression model created here.
# 
# Thanks for visiting my kernal, hope you found this helpful for getting handy with tensorflow. Will connect again soon. Till then, happy coding.
# 
# *Please hit upvote if you think this kernal is useful. Thanks again.*
# 
# **Manish**

# In[ ]:




