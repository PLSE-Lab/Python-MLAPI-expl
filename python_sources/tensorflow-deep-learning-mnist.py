#!/usr/bin/env python
# coding: utf-8

# # TensorFlow: implementing a deep learning neural network wit MNIST dataset
# 
# This is a simple kernel to implement a deep learning neural network with de MSNIT dataset to multiclass classification
# 
# Annotations:
# * 784 neurons into the input layer (pixels from image 28x28)
# * 3 hidden layers with 397 neurons into the hidden layers
# * 10 neurons into the output layer
# * 55000 images into the MNIST dataset 

# In[ ]:


import tensorflow as tf
import numpy as np


# # Loading data

# In[ ]:


from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('mnist/', one_hot=True)


# # Tensorflow implementation

# ### Train and Test data

# In[ ]:


x_train = mnist.train.images
y_train = mnist.train.labels
x_test = mnist.test.images
y_test = mnist.test.labels

x_train.shape


# In[ ]:


y_test.shape


# ### Showing a sample image

# In[ ]:


import matplotlib.pyplot as plt
import numpy as np
get_ipython().run_line_magic('matplotlib', 'inline')
plt.imshow(x_train[102].reshape((28,28)), cmap='gray')
plt.title('Classe: ' + str(np.argmax(y_train[102])))


# ### Defining batch size 

# In[ ]:


x_batch, y_batch = mnist.train.next_batch(128)
x_batch.shape


# # Layers definitions
# 
# Neural Network Layers
# 
# >        input layer -> hidden_layer1 -> hidden_layer2 -> hidden_layer3 -> output layer
# 
# Weights for each layer:
# >        784      ->        397           ->         397          ->           397         ->  10
#       
# 

# In[ ]:


neurons_input = x_train.shape[1] # 784 pixels converted from a 28x28 image
print('Input Layer Neurons: ', neurons_input)

neurons_hidden1 = neurons_hidden2 = neurons_hidden3 = int((x_train.shape[1] + y_train.shape[1]) / 2) # (784+10)/2 = 397
print('Hidden1 Layer Neurons: ', neurons_hidden1)
print('Hidden2 Layer Neurons: ', neurons_hidden2)
print('Hidden3 Layer Neurons: ', neurons_hidden3)

neurons_output = y_train.shape[1] # 10 of target classifications
print('Output Layer Neurons: ', neurons_output)


# In[ ]:


weights = {
    'hidden1': tf.Variable(tf.random_normal([neurons_input, neurons_hidden1])),
    'hidden2': tf.Variable(tf.random_normal([neurons_hidden1, neurons_hidden2])),
    'hidden3': tf.Variable(tf.random_normal([neurons_hidden2, neurons_hidden3])),
    'output': tf.Variable(tf.random_normal([neurons_hidden3, neurons_output])),
}


# In[ ]:


bias = {
    'hidden1': tf.Variable(tf.random_normal([neurons_hidden1])),
    'hidden2': tf.Variable(tf.random_normal([neurons_hidden2])),
    'hidden3': tf.Variable(tf.random_normal([neurons_hidden3])),
    'output': tf.Variable(tf.random_normal([neurons_output]))
}


# In[ ]:


xph = tf.placeholder('float', [None, neurons_input])
yph = tf.placeholder('float', [None, neurons_output])


# ### Processing

# In[ ]:


def run_process(x, weights, bias):
    hidden_layer1 = tf.nn.relu(tf.add(tf.matmul(x, weights['hidden1']), bias['hidden1']))
    hidden_layer2 = tf.nn.relu(tf.add(tf.matmul(hidden_layer1, weights['hidden2']), bias['hidden2']))
    hidden_layer3 = tf.nn.relu(tf.add(tf.matmul(hidden_layer2, weights['hidden3']), bias['hidden3']))
    output_layer = tf.add(tf.matmul(hidden_layer3, weights['output']), bias['output'])
    return output_layer


# In[ ]:


# train model functions
model = run_process(xph, weights, bias)
error = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=model, labels=yph))
optimizer = tf.train.AdamOptimizer(learning_rate=0.0001).minimize(error)

# prediction result
predictions = tf.nn.softmax(model)
predictions_final = tf.equal(tf.argmax(predictions, 1), tf.argmax(yph,1))

# score function
score = tf.reduce_mean(tf.cast(predictions_final, tf.float32))


# ### Executing on TensorFlow session

# In[ ]:


with tf.Session() as s:
    
    # require to initialize the TensorFlow Variables
    s.run(tf.global_variables_initializer())
    
    # running the trainning for 5000 epochs
    for epoch in range (5000):
        x_batch, y_batch = mnist.train.next_batch(128)
        _, cost = s.run([optimizer, error], feed_dict = { xph: x_batch, yph: y_batch })
        if epoch % 100 == 0:
            acc = s.run([score], feed_dict = {xph: x_batch, yph: y_batch})
            print('Epoch: '+ str(epoch+1) + ' - Error: ' + str(cost) + ' - Accuracy: ' + str(acc))
    
    print('Trained.')
    
    # evaluate the accuracy using our test data
    print(s.run(score, feed_dict = { xph: x_test, yph: y_test }))

