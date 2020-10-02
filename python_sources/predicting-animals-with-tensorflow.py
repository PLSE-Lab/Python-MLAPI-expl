import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import tensorflow as tf
import numpy as np
import pandas as pd


dataset = pd.read_csv('../input/zoo.csv', header=0)
dataset = pd.get_dummies(dataset, columns=['animal_name'])

values = list(dataset.columns.values)
Y = dataset[values[-100:]]
Y = np.array(Y, dtype=np.float32)
X = dataset[values[0:-100]]
X = np.array(X, dtype=np.float32)
# Session
sess = tf.Session()
# Interval / Epochs
interval = 100
epoch = 1500


#Initialize Neural Network
X_data = tf.placeholder(dtype=np.float32, shape=[None, 17])
Y_target = tf.placeholder(dtype=np.float32, shape=[None, 100])

hidden_layer_nodes = 16

w1 = tf.Variable(tf.random_normal(shape=[17, hidden_layer_nodes]))
b1 = tf.Variable(tf.random_normal(shape=[hidden_layer_nodes]))
w2 = tf.Variable(tf.random_normal(shape=[hidden_layer_nodes, 100]))
b2 = tf.Variable(tf.random_normal(shape=[100]))

hidden_output = tf.nn.relu(tf.add(tf.matmul(X_data, w1), b1))
final_output = tf.nn.softmax(tf.add(tf.matmul(hidden_output, w2), b2))

#loss = tf.reduce_mean(-tf.reduce_sum(Y_target * tf.log(final_output), axis=0))
loss = tf.reduce_mean(-tf.reduce_sum(Y_target * tf.log(final_output + 1e-10), axis=0))

# Optimizer
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(loss)
#optimizer = tf.train.AdamOptimizer().minimize(loss)

# Initialize variables
init = tf.global_variables_initializer()
sess.run(init)
# Training
print('Training the model...')
for i in range(1, (epoch + 1)):
    sess.run(optimizer, feed_dict={X_data: X, Y_target: Y})
    if i % interval == 0:
        print('Epoch', i, '|', 'Loss:', sess.run(loss, feed_dict={X_data: X, Y_target: Y}))
# Prediction
print("\nTrying to predict Buffolo (Index 6) ...")
flower = np.array([[1,0,0,1,0,0,0,1,1,1,0,0,4,1,0,1,1]], np.float32)
print(np.rint(sess.run(final_output, feed_dict={X_data: flower})))