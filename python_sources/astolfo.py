import pandas as pd
import numpy as np
import tensorflow as tf

# Aprender

# carregando dados
train = pd.read_csv("../input/train.csv").as_matrix()
print("Training set has {0[0]} rows and {0[1]} columns".format(train.shape))
print(pd.get_dummies(train[0:100,0]))

# Construir modelo
x = tf.placeholder(tf.float32, [None, 784])
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))

# sa�da real
y = tf.nn.softmax(tf.matmul(x, W) + b)

# sa�da desejada
y_ = tf.placeholder(tf.float32, [None, 10])

cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))

train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

#iniciar sess�o  
sess = tf.InteractiveSession()
#iniciar vari�veis
tf.global_variables_initializer().run()

# treinar o modelo
for _ in range(1000):
    batch_xs = train[:,1:]
    batch_ys = pd.get_dummies(train[:,0]).as_matrix()
    sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

exit()
# Testar
test  = pd.read_csv("../input/test.csv")
print("Test set has {0[0]} rows and {0[1]} columns".format(test.shape))