# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import tensorflow as tf
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
oh = OneHotEncoder()
import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
train = pd.read_csv("../input/fashion-mnist_train.csv")
test = pd.read_csv("../input/fashion-mnist_test.csv")


trainx, trainy = train.iloc[:,1:].values, train.iloc[:,0].values
testx, testy = test.iloc[:,1:].values, test.iloc[:,0].values
sc = StandardScaler()
trainx = sc.fit_transform(trainx)
testx = sc.fit_transform(testx)
trainx = np.reshape(trainx, (trainx.shape[0], 28, 28, 1))
testx = np.reshape(testx, (testx.shape[0], 28, 28, 1))

trainy = np.asarray(trainy, dtype = np.float32).reshape(-1,1)
testy = np.asarray(testy, dtype = np.float32).reshape(-1,1)

trainy = oh.fit_transform(trainy).toarray()
testy = oh.fit_transform(testy).toarray()


def random_minibatches(x,y,size=64, seed = 0):
	#x = tf.shuffle(x, seed)
	#y = tf.shuffle(y, seed)
	m = x.shape[0]
	nums = m//size
	batches = []
	for k in range(nums):
		mx = x[k*size:(k+1)*size,:,:,:]
		my = y[k*size:(k+1)*size,:]
		batches.append((mx,my))
	mx = x[nums*size:m,:,:,:]
	my = y[nums*size:m,:]
	batches.append((mx,my))
	return batches

def generate_batch(features, labels, batch_size):
    batch_indexes = np.random.random_integers(0, len(features) - 1, batch_size)
    batch_features = features[batch_indexes]
    batch_labels = labels[batch_indexes]
    
    return (batch_features, batch_labels)

def create_placeholders(nh0, nw0, nc0, ny):
    x = tf.placeholder('float', shape = (None, nh0 ,nw0, nc0))
    y = tf.placeholder('float', shape = (None, ny))
    return x,y

def init_weights():
    w1 = tf.get_variable('w1', [5,5,1,32],initializer = tf.contrib.layers.xavier_initializer_conv2d(seed = 2))
    w2 = tf.get_variable('w2', [5,5,32,64], initializer = tf.contrib.layers.xavier_initializer_conv2d(seed = 2))
    params = {"w1":w1, "w2":w2}
    return params


def forward_prop(x, params):
    w1 = params["w1"]
    w2 = params["w2"]
    z1 = tf.nn.conv2d(x,w1,[1,1,1,1], padding = 'SAME')
    a1 = tf.nn.relu(z1)
    p1 = tf.nn.max_pool(a1, ksize = [1,4,4,1], strides = [1,4,4,1], padding = 'SAME')
    
    z2 = tf.nn.conv2d(p1, w2, [1,1,1,1], padding = "SAME")
    a2 = tf.nn.relu(z2)
    p2 = tf.nn.max_pool(a2, ksize = [1,4,4,1], strides = [1,4,4,1], padding = "SAME")
    
    fc1 = tf.contrib.layers.flatten(p2)
    fc2 = tf.contrib.layers.fully_connected(fc1, num_outputs = 1024, activation_fn = tf.nn.relu)
    dp1 = tf.nn.dropout(fc2, 0.6)
    fc3 = tf.contrib.layers.fully_connected(dp1, num_outputs = 10, activation_fn = None)
    return fc3

def loss(fc3, y):
    cost = tf.nn.softmax_cross_entropy_with_logits(logits = fc3, labels = y)
    cost = tf.reduce_mean(cost)
    return cost

def model(trainx, trainy, testx, testy, learning_rate = 0.009, epochs = 35):
    (m,nh0,nw0,nc0) = trainx.shape
    (m,ny) = trainy.shape
    seed = 2
    x,y = create_placeholders(nh0,nw0,nc0,ny)
    costs = []
    params = init_weights()
    fc3 = forward_prop(x, params)
    cost = loss(fc3,y)
    optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)
    init = tf.global_variables_initializer()
    nums = trainx.shape[0]//128
    with tf.Session() as sess:
        sess.run(init)
        
        for i in range(epochs):
            minibatches = random_minibatches(trainx,trainy, size = 256)
            lcost = 0
            cnt = 0
            for minibatch in minibatches:
                cnt += 1
                mx,my = minibatch
                _, temp = sess.run([optimizer, cost], feed_dict = {x:mx, y:my})
                lcost += temp
                if cnt%5 == 0:
                    #print(temp)
            lcost = lcost / nums
            if i%10 == 0:
                print(i, lcost)
                costs.append(lcost)
        ypred = tf.argmax(fc3, axis = 1)
        yreal = tf.argmax(y, axis = 1)
        correct = tf.equal(ypred, yreal)
        acc = tf.reduce_mean(tf.cast(correct, 'float'))
        train_acc = 0
        minibatches = random_minibatches(trainx, trainy,size = 128)
        for minibatch in minibatches:
            mx, my = minibatch
            tr_ac = acc.eval({x:mx, y:my})
            train_acc += tr_ac
        train_acc = train_acc/ nums
        #train_acc = acc.eval({x:trainx, y:trainy})
        test_acc = 0
        minibatches = random_minibatches(testx, testy, size = 128)
     
        test_acc = acc.eval({x:testx, y:testy})
        print(train_acc, test_acc)
        return train_acc, test_acc, params
    
_,_,params = model(trainx, trainy,testx, testy)








