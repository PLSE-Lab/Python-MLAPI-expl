import numpy as np
import pandas as pd
import tensorflow as tf
import math

#Print you can execute arbitrary python code
data = pd.read_csv("../input/train.csv", dtype={"Age": np.float64,}, )
test = pd.read_csv("../input/test.csv", dtype={"Age": np.float64}, )

#Reserve validation set
valid = data.sample(frac = 0.0)
#Separate training set
train = data[~data.index.isin(valid.index)]

#Preprocess the data: specifically, convert gender to binary and implement label as vector
def preprocess(df):
    y_temp = df['Survived'].values
    x = df[['Sex', 'Pclass']].values
    y = [None] * len(x)
    for i in range(len(x)):
        if y_temp[i] == 1:
            y[i] = [0, 1]
        else:
            y[i] = [1, 0]
        if x[i][0] == 'male':
            x[i][0] = 0
        else:
            x[i][0] = 1
    return (np.asarray(x), np.asarray(y))

x_v, y_v = preprocess(valid)
x_t, y_t = preprocess(train)

#print(np.shape(x_v), np.shape(y_v), np.shape(x_t), np.shape(y_t))
#The parameters: important ones are step count and batch size
NUM_CLASSES = 2
NUM_FEAT = 2
TRAIN_STEPS = 1000
BATCH_SIZE = 100
LEARNING_RATE = 0.03

sess = None
def ResetSession():
    tf.reset_default_graph()
    global sess
    if sess is not None: sess.close()
    sess = tf.InteractiveSession()

ResetSession()

x = tf.placeholder(tf.float32, [None, NUM_FEAT], name='features')
y = tf.placeholder(tf.float32, [None, NUM_CLASSES], name = 'labels')
b = tf.Variable(tf.truncated_normal(shape = [NUM_CLASSES], dtype = np.float32, stddev = 1.0 / math.sqrt(float(NUM_FEAT))), name='bias')
w = tf.Variable(tf.truncated_normal(shape = [NUM_FEAT, NUM_CLASSES], dtype = np.float32, stddev = 1.0 / math.sqrt(float(NUM_FEAT))),  name = "weights")
h = tf.matmul(x, w) + b

loss = tf.reduce_mean(tf.losses.mean_squared_error(labels=y_t, predictions=h))
tr = tf.train.GradientDescentOptimizer(LEARNING_RATE).minimize(loss)

tf.global_variables_initializer().run()
counter = 0
for step in range(TRAIN_STEPS):
    _, loss_val = sess.run([tr, loss], feed_dict = {x: x_t, y: y_t})
    #val_val = sess.run(loss, feed_dict = {x: x_v, y: y_v})
    #print(val_val)
    counter += 1
    if counter % (TRAIN_STEPS // 10) == 0 or counter == 1:
        print("iteration %d: loss %f" %(counter, loss_val))
    #print("loss %f -- val %f" %(loss_val, val_val)) 
    

#Print to standard output, and see the results in the "log" section below after running your script
print("\n\nTop of the training data:")
print(train.head())

print("\n\nSummary statistics of training data")
print(train.describe())
pred = [0] * len(test)
#Any files you save will be available in the output tab below
data.to_csv('copy_of_the_training_data.csv', index=False)

#output file
x_test = test[['Sex', 'Pclass']].values
for i in range(len(test)):
    if x_test[i][0] == 'male':
        x_test[i][0] = 0
    else:
        x_test[i][0] = 1
result = sess.run(h, feed_dict = {x: x_test})
for i in range(len(test)):
    if result[i][0] > result[i][1]:
        pred[i] = 0
    else:
        pred[i] = 1
#print(pred)
submission = pd.DataFrame({'PassengerId': test['PassengerId'], 'Survived': pred})
submission.to_csv('submission.csv', index = False)