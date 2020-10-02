import pandas as pd
import numpy as np
import tensorflow as tf

###############################################
#TREINANDO
###############################################
train = pd.read_csv("../input/train.csv").as_matrix()
print("Training set has {0[0]} rows and {0[1]} columns".format(train.shape))

###############################################
#CONSTRUINDO O MODELO
###############################################

print(pd.get_dummies(train[0:100,0]))

x = tf.placeholder(tf.float32, [None, 784])
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))
y = tf.nn.softmax(tf.matmul(x, W) + b)

y_ = tf.placeholder(tf.float32, [None, 10])

cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))

train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

sess = tf.InteractiveSession()
tf.global_variables_initializer().run()
###############################################
#Treinando
###############################################
for _ in range(1):
    batch_xs = train[:,1:]
    batch_ys = pd.get_dummies(train[:,0]).as_matrix()
    sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})
    
exit()

###############################################
#TESTANDO
###############################################
test  = pd.read_csv("../input/test.csv")
print("Test set has {0[0]} rows and {0[1]} columns".format(test.shape))

# Any files you write to the current directory get shown as outputs 