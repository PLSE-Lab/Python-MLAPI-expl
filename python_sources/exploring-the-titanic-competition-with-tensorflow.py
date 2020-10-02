import numpy as np
import pandas as pd
import tensorflow as tf

#Print you can execute arbitrary python code
train = pd.read_csv("../input/train.csv", dtype={"Age": np.float64}, )
test = pd.read_csv("../input/test.csv", dtype={"Age": np.float64}, )

#Print to standard output, and see the results in the "log" section below after running your script
print("\n\nTop of the training data:")
print(train.head())

print("\n\nSummary statistics of training data")
print(train.describe())

train = train.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis = 1)
train = train.replace(['male', 'female'], [0, 1])
train = train.replace(['C', 'Q', 'S'], [0, 1, 2])
train = train.fillna(0)
train = (train-train.min())/(train.max()-train.min())
train.to_csv('cleaned_training_data.csv', index=False)

ids = test[['PassengerId']]
test = test.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis = 1)
test = test.replace(['male', 'female'], [0, 1])
test = test.replace(['C', 'Q', 'S'], [0, 1, 2])
test = test.fillna(0)
test = (test-test.min())/(test.max()-test.min())
test.to_csv('cleaned_test_data.csv', index=False)

print(train.head())

#Any files you save will be available in the output tab below
#train.to_csv('copy_of_the_training_data.csv', index=False)

x = tf.placeholder(tf.float32, [None, 7])

#Wh1 = tf.Variable(tf.zeros([7, 7]))
#bh1 = tf.Variable(tf.zeros([7]))
Wh2 = tf.Variable(tf.zeros([7, 7]))
bh2 = tf.Variable(tf.zeros([7]))
Wy = tf.Variable(tf.zeros([7, 2]))
by = tf.Variable(tf.zeros([2]))
#W = tf.Variable(tf.random_uniform([7, 1]))
#b = tf.Variable(tf.random_uniform([1]))

#h1 = tf.nn.sigmoid(tf.matmul(x, Wh1) + bh1)
h2 = tf.nn.sigmoid(tf.matmul(x, Wh2) + bh2)
y = tf.nn.softmax(tf.matmul(h2, Wy) + by)
y_ = tf.placeholder(tf.float32, [None, 2])

#cross_entropy = tf.reduce_mean(-y_ * tf.log(y) - (1 - y_) * tf.log(1 - y))
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))

train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

sess = tf.InteractiveSession()
tf.global_variables_initializer().run()

for _ in range(1):
    batch = train.sample(n = 100)
    batch_xs = batch.drop(['Survived'], axis = 1)
    batch_ys = batch[['Survived']]
    batch_ys['Killed'] = 1 - batch[['Survived']]
    sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

#print(sess.run(h1, feed_dict={x: test}))

predictions = sess.run(1 - tf.argmax(y, 1), feed_dict={x: test})
print(predictions)
predictions = pd.DataFrame(predictions, columns = ['Survived'])
#predictions = predictions.round().apply(np.int64)
predictions['PassengerId'] = ids
print(predictions.head())
predictions.to_csv('predictions.csv', index=False)
