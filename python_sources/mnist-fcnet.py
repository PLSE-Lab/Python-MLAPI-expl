import numpy as np
import tensorflow as tf

learning_rate = 0.001
steps = 10000
batch_size = 128
n_input = 784
n_output = 10

n_fc1 = 500

x = tf.placeholder(tf.float32, [None, n_input])
y = tf.placeholder(tf.float32, [None,n_output])

def net(x, w, b):
    fc1 = tf.nn.relu(tf.matmul(x,w['w1'] +b['b1']))
    out = tf.matmul(fc1, w['out']) + b['out']
    return tf.nn.softmax(out)
    
weights = {
    'w1': tf.Variable(tf.random_normal([n_input, n_fc1])),
    'out': tf.Variable(tf.random_normal([n_fc1, n_output])),
    }
biases = {
    'b1': tf.Variable(tf.random_normal([n_fc1])),
    'out':tf.Variable(tf.random_normal([n_output])),
    }

pred = net(x, weights, biases)
cost = tf.reduce_mean(-y*tf.log(tf.clip_by_value(pred, 1e-10, 1.0)))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
correct_pred = tf.equal(tf.argmax(y,1), tf.argmax(pred,1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

init = tf.initialize_all_variables()

def load_data():
    global train_data, test_data, val_data
    train_data = np.genfromtxt("../input/train.csv", delimiter=",")
    print(train_data)
load_data()
    
    