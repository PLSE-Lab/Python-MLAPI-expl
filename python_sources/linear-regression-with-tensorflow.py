# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

'''
This is a simple example of Regression using Tensorflow .The data is self generated and methods are either 
self explainatory or explained as per requirement
'''
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import tensorflow as tf

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.

# creating random data with noise to be added to it 
x_data = np.linspace(0.0,10.0,1000000)
noise = np.random.randn(len(x_data))
#  The desired graph : y = mx + b
# b = 5
# m = 0.5

y_true = (0.5 * x_data) + 5 + noise
# creating the X dataFrame
x_df = pd.DataFrame(data=x_data,columns=['X data'])
# creating the Y dataFrame
y_df = pd.DataFrame(data=y_true,columns=['Y'])
# concatinating the dataFrames X and Y
my_data = pd.concat([x_df,y_df],axis=1)
# plotting the new DataFrame
my_data.sample(n=250).plot(kind = 'scatter',x='X data',y = 'Y')
# using random values for Variables m and b ....you can use any value 
batch_size = 8
m = tf.Variable(0.81)
b = tf.Variable(0.17)
# using placeholder for x and y 
xph = tf.placeholder(tf.float32,[batch_size])
yph = tf.placeholder(tf.float32,[batch_size])
# generating the model
y_model = m*xph + b
# checking the loss incurred
loss = tf.reduce_sum(tf.square(yph-y_model))
# optimising the result and training the data 
optimizer  = tf.train.GradientDescentOptimizer(learning_rate=0.001)
train = optimizer.minimize(loss)
# though the varaibles have been created we need to initialise them to access them 
init = tf.global_variables_initializer()
# the sessionn is used to do evaluation here 
with tf.Session() as sess:
    sess.run(init)
    
    batches = 1000
    
    for i in range(batches):
        
        rand_ind = np.random.randint(len(x_data),size = batch_size)
        
        feed = {xph:x_data[rand_ind],yph:y_true[rand_ind]}
        
        sess.run(train,feed_dict = feed)
        
        model_m , model_b = sess.run([m,b])
# printing the value of m and b as predicted by the regresssion method 
# output values for m = 0.598 and b  = 4.92 and the input values for m was 0.5 and b was 5
print(model_m)
print(model_b)

y_hat = x_data*model_m +model_b
# plotting the linear regression output
my_data.sample(250).plot(kind='scatter',x= 'X data',y = 'Y')
plt.plot(x_data,y_hat,'r')