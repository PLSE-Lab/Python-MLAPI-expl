import numpy as np 
import pandas as pd 
import tensorflow as tf
import matplotlib.pyplot as plt
import os

traindata = pd.read_csv("../input/train.csv")
traindata = traindata.sample(frac=1).reset_index(drop=True) # shuffles

y_labels = traindata['label'].values
y_labels = np.eye(10)[y_labels.reshape(-1)]

del traindata['label']
traindata = traindata.values
traindata = traindata.reshape([-1, 28, 28, 1])

train_size = 0.8
m = traindata.shape[0]
m_train = int(m*train_size)

train  = (traindata[:m_train,:,:,:] - 128.)/255.
validation = (traindata[m_train:,:,:,:] - 128.)/255.
y_train = y_labels[:m_train,:]
y_val = y_labels[m_train:,:]

# -------------------------- For submission on the provided test set -------------------------- #
#testdata = pd.read_csv("../input/test.csv")
#indx = list(testdata.index)
#indx = [i+1 for i in indx]
#testdata = testdata.values
#testdata = testdata.reshape([-1, 28, 28, 1])
#test = (testdata[:,:,:,:] - 128.)/255.
# --------------------------------------------------------------------------------------------- #


X = tf.placeholder(shape=[None, 28, 28, 1], dtype = tf.float32)
Y = tf.placeholder(shape=[None, 10], dtype = tf.float32)

# Filters
W1 = tf.get_variable("W1", [4, 4, 1, 18], initializer = tf.contrib.layers.xavier_initializer(seed=0))
W2 = tf.get_variable("W2", [2, 2, 18, 13], initializer = tf.contrib.layers.xavier_initializer(seed=0))

params = {'W1':W1, 'W2':W2}
def forward(X, parameters):
    W1 = parameters['W1']
    W2 = parameters['W2']
    
    Z1 = tf.nn.conv2d(X, W1, strides=[1,1,1,1], padding='SAME')
    A1 = tf.nn.relu(Z1)
    P1 = tf.nn.max_pool(A1, ksize = [1,2,2,1], strides = [1,2,2,1], padding = 'SAME')
    Z2 = tf.nn.conv2d(P1, W2, strides=[1,1,1,1], padding='SAME')
    A2 = tf.nn.relu(Z2)
    P2 = tf.nn.max_pool(A2, ksize = [1,2,2,1], strides = [1,2,2,1], padding = 'SAME')
    P2 = tf.contrib.layers.flatten(A2)
    Z3 = tf.contrib.layers.fully_connected(P2, num_outputs=10, activation_fn=None)
    
    return Z3
    
    
def compute_cost(Z3, Y):
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits = Z3, labels = Y)) + \
        0.01*tf.nn.l2_loss(W1) + 0.01*tf.nn.l2_loss(W2)
    return cost

# -------------------------- The tensorflow graph -------------------------------- #
    
Z3 = forward(X, params)
cost = compute_cost(Z3, Y)
optimizer = tf.train.AdamOptimizer(learning_rate = 0.0005).minimize(cost)
init = tf.global_variables_initializer()
predict_op = tf.argmax(Z3,1)
correct_prediction = tf.equal(predict_op, tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

# -------------------------------------------------------------------------------- #

num_epochs = 70
minibatch_size = 32

with tf.Session(config=tf.ConfigProto(log_device_placement=True)) as sess:
    sess.run(init)
    for epoch in range(num_epochs):
        print("epoch : "+str(epoch))
        epoch_cost = 0.
        num_minibatches = int(m_train / minibatch_size)
        for i in range(num_minibatches):
            
            minibatch_X = train[i*minibatch_size:(i+1)*minibatch_size,:,:,:]
            minibatch_Y = y_train[i*minibatch_size:(i+1)*minibatch_size,:]
            
            _, minibatch_cost = sess.run([optimizer, cost], feed_dict={X:minibatch_X, Y:minibatch_Y})
            epoch_cost += minibatch_cost / num_minibatches
        
        if epoch%5 == 0:
            print("Cost = "+str(epoch_cost))
            print ("Train Accuracy:", accuracy.eval({X: train, Y: y_train}))
            print ("Validation Accuracy:", accuracy.eval({X: validation, Y: y_val}))
    
    
    print ("\nFINAL\nTrain Accuracy:", accuracy.eval({X: train, Y: y_train}))
    print ("Validation Accuracy:", accuracy.eval({X: validation, Y: y_val}))
    
    # ------------------------- submission ----------------------------- #
    #out = list(predict_op.eval({X: test}))
    #out = pd.DataFrame({'ImageId':indx,'Label':out})
    #out.to_csv("submission.csv", index=False)

# --------------------------------------- X --------------------------------------- #