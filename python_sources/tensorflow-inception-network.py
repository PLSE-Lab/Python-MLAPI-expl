import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import tensorflow as tf
import matplotlib.pyplot as plt
import os

traindata = pd.read_csv("../input/train.csv")
testdata = pd.read_csv("../input/test.csv")
indx = list(testdata.index)
indx = [i+1 for i in indx]

data = traindata.sample(frac=1).reset_index(drop=True)

y_labels = data['label'].values
del data['label']
y_labels = np.eye(10)[y_labels.reshape(-1)]

data = data.values
data = data.reshape([-1, 28, 28, 1])
testdata = testdata.values
testdata = testdata.reshape([-1, 28, 28, 1])


m = data.shape[0]
m_train = int(m*0.9)

train  = (data[:m_train,:,:,:] - 128.)/255.
val = (data[m_train:,:,:,:] - 128.)/255.
y_train = y_labels[:m_train,:]
y_val = y_labels[m_train:,:]
test = (testdata[:,:,:,:] - 128.)/255.

print("Shapes:", train.shape, test.shape, y_train.shape)

X = tf.placeholder(shape=[None, 28, 28, 1], dtype = tf.float32)
Y = tf.placeholder(shape=[None, 10], dtype = tf.float32)
r = tf.placeholder(shape=[], dtype = tf.float32)

W1 = tf.get_variable("W1", [3, 3, 1, 18], initializer = tf.contrib.layers.xavier_initializer(seed=0))
W2 = tf.get_variable("W2", [5, 5, 1, 8], initializer = tf.contrib.layers.xavier_initializer(seed=0))
W3 = tf.get_variable("W3", [7, 7, 1, 6], initializer = tf.contrib.layers.xavier_initializer(seed=0))
W4 = tf.get_variable("W4", [5, 5, 32, 32], initializer = tf.contrib.layers.xavier_initializer(seed=0))
W5 = tf.get_variable("W5", [2, 2, 32, 64], initializer = tf.contrib.layers.xavier_initializer(seed=0))

params = {'W1':W1, 'W2':W2, 'W3':W3, 'W4':W4, 'W5':W5}

def forward(X, parameters):
    W1 = parameters['W1']
    W2 = parameters['W2']
    W3 = parameters['W3']
    W4 = parameters['W4']
    W5 = parameters['W5']
    
    T1 = tf.nn.conv2d(X, W1, strides=[1,1,1,1], padding='SAME')
    T2 = tf.nn.conv2d(X, W2, strides=[1,1,1,1], padding='SAME')
    T3 = tf.nn.conv2d(X, W3, strides=[1,1,1,1], padding='SAME')
    Z1 = tf.concat([T1, T2, T3], 3)
    ZA1 = tf.contrib.layers.batch_norm(Z1)
    A1 = tf.nn.relu(ZA1)
    P1 = tf.nn.max_pool(A1, ksize = [1,2,2,1], strides = [1,2,2,1], padding = 'SAME')
    Z2 = tf.nn.conv2d(P1, W4, strides=[1,1,1,1], padding='VALID')
    ZA2 = tf.contrib.layers.batch_norm(Z2)
    A2 = tf.nn.relu(Z2)
    
    Z2t = tf.nn.conv2d(A2, W5, strides=[1,1,1,1], padding='SAME')
    A2t = tf.nn.relu(Z2t)
    
    P2 = tf.nn.max_pool(A2t, ksize = [1,2,2,1], strides = [1,2,2,1], padding = 'SAME')
    P3 = tf.contrib.layers.flatten(P2)
    P3 = tf.contrib.layers.batch_norm(P3)
    
    Z3 = tf.contrib.layers.fully_connected(P3, num_outputs=128, weights_initializer = tf.contrib.layers.xavier_initializer(seed=0), \
        weights_regularizer = tf.contrib.layers.l2_regularizer(scale=0.001), activation_fn=None)
    Z3n = tf.contrib.layers.batch_norm(Z3)
    A3 = tf.nn.relu(Z3n)
    Z4 = tf.contrib.layers.fully_connected(A3, num_outputs=64, weights_initializer = tf.contrib.layers.xavier_initializer(seed=0), \
        weights_regularizer = tf.contrib.layers.l2_regularizer(scale=0.001), activation_fn=None)
    Z4n = tf.contrib.layers.batch_norm(Z4)
    A4 = tf.nn.relu(Z4n)
    Z5 = tf.contrib.layers.fully_connected(A4, weights_initializer = tf.contrib.layers.xavier_initializer(seed=0), num_outputs=10, activation_fn=None)
    
    print(Z1.shape, P1.shape, Z2.shape, P2.shape, P3.shape, A3.shape, A4.shape, Z5.shape)
    return Z5
    
    
def compute_cost(Z, Y):
    reg_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits = Z, labels = Y)) +\
        0.0001*tf.nn.l2_loss(W1) + 0.0001*tf.nn.l2_loss(W2) + 0.0001*tf.nn.l2_loss(W3) + 0.0002*tf.nn.l2_loss(W4) + \
        0.00001*tf.nn.l2_loss(W5) + sum(reg_losses)
    
    return cost

Z5 = forward(X, params)
cost = compute_cost(Z5, Y)
optimizer = tf.train.AdamOptimizer(learning_rate = r).minimize(cost)
init = tf.global_variables_initializer()
predict_op = tf.argmax(Z5,1)
correct_prediction = tf.equal(predict_op, tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
num_epochs = 150
minibatch_size = 16
val_acc = 0.986

with tf.Session(config=tf.ConfigProto(log_device_placement=True)) as sess:
    sess.run(init)
    print("start epochs")
    for epoch in range(num_epochs):
        
        epoch_cost = 0.
        num_minibatches = int(m_train / minibatch_size)
        
        # ------------ adaptive rate ------------- #
        rate = 0.0005/pow(1.3,max(val_acc-0.986,0)/0.001)
        if val_acc > 0.991:
            rate = 0.5*rate
        # ---------------------------------------- #
            
        print("Epoch: "+str(epoch) + " Rate: "+str(rate))
        
        for i in range(num_minibatches):
            
            minibatch_X = train[i*minibatch_size:(i+1)*minibatch_size,:,:,:]
            minibatch_Y = y_train[i*minibatch_size:(i+1)*minibatch_size,:]
            
            _, minibatch_cost = sess.run([optimizer, cost], feed_dict={X:minibatch_X, Y:minibatch_Y, r:rate})
            epoch_cost += minibatch_cost / num_minibatches
        
        print("Cost is = "+str(epoch_cost))
        val_acc = accuracy.eval({X: val, Y: y_val})
        
        if epoch%5 == 0:
            train_acc = accuracy.eval({X: train, Y: y_train})
            print ("Train Accuracy:", train_acc)
            print ("Validation Accuracy:", val_acc)
        
        if val_acc > 0.9935 and epoch_cost < 0.005:
            print("reached threshold")
            break
        
    out = list(predict_op.eval({X: test}))
    out = pd.DataFrame({'ImageId':indx,'Label':out})
    out.to_csv("submission.csv", index=False)