import numpy as np
import tensorflow as tf

# The competition datafiles are in the directory ../input
# Read competition data files:
train = np.loadtxt("../input/train.csv", delimiter=',', skiprows=1)

def n2b(n):
    ret = [item for item in format(2**int(n[0]), '010b')]
    ret.reverse()
    return np.array(ret)
    
trX = train[:, 1:]
totalY = train[:, np.newaxis, 0]
trY = np.array([item for item in map(n2b, totalY)])

trainX, validX, trainY, validY = trX[:-5000], trX[-5000:], trY[:-5000], trY[-5000:]

# Any files you write to the current directory get shown as outputs
print(trainX.shape, validX.shape, trainY.shape, validY.shape)

lr = 0.001
bs = 100
total_epochs = 10000
display_step = 200

n_input = 784
n_hidden1 = 200
n_hidden2 = 100
n_classes = 10

X = tf.placeholder('float', [None, n_input])
Y = tf.placeholder('float', [None, n_classes])

weights = {
    'h1': tf.Variable(tf.random_normal([n_input, n_hidden1])),
    'h2': tf.Variable(tf.random_normal([n_hidden1, n_hidden2])),
    'out': tf.Variable(tf.random_normal([n_hidden2, n_classes]))
}

biases = {
    'h1': tf.Variable(tf.random_normal([n_hidden1])),
    'h2': tf.Variable(tf.random_normal([n_hidden2])),
    'out': tf.Variable(tf.random_normal([n_classes]))
}


def mlp(X, weights, biases):
    h1 = tf.matmul(X, weights['h1']) + biases['h1']
    h1 = tf.nn.relu(h1)
    h2 = tf.matmul(h1, weights['h2']) + biases['h2']
    h2 = tf.nn.relu(h2)
    out = tf.matmul(h2, weights['out']) + biases['out']
    return out


pred = mlp(X, weights, biases)
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(pred, Y)/bs)

optimizer = tf.train.AdamOptimizer(lr).minimize(cost)

init = tf.initialize_all_variables()

with tf.Session() as sess:
    sess.run(init)

    last_costs = [np.divide(1, 0)]
    for e in range(total_epochs):
        idx = np.random.randint(trX.shape[0]-bs)
        sess.run(optimizer, feed_dict={
            X: trainX[idx: idx+bs], Y: trainY[idx: idx+bs]
        })
        if (e+1) % display_step == 0:
            vc = sess.run(cost, feed_dict={X: validX, Y: validY})
            if last_costs[-1] > vc:
                last_costs.append(vc)
            else:
                break
            print(e+1, 'Validation cost = ', last_costs[-1])

            precision = tf.equal(tf.argmax(pred, 1), tf.argmax(Y, 1))
            acc = tf.reduce_mean(tf.cast(precision, tf.float32))
            print('Accuracy:', acc.eval({X: validX, Y: validY}))