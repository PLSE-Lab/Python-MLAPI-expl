import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split

import tensorflow as tf

def load_data():
    # Load the data
    data = pd.read_csv('../input/iris.data', header=None, names=['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'type'])
    
    # Seperate features and labels from data
    y = data['type']
    x = data.drop('type', axis=1)
    
    # One hot encode y
    y = pd.get_dummies(y)
    
    # Split the data into training and testing data
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3,shuffle=True)
    
    # Convert to arrays
    x_train = x_train.values
    y_train = y_train.values
    x_test = x_test.values
    y_test = y_test.values

    return x_train, x_test, y_train, y_test
    
# Create a Neural Network
def neural_net(weights, biases):
    
    hidden_layer = tf.add(tf.matmul(x, weights['w1']), biases['b1'])
    hidden_layer = tf.nn.relu(hidden_layer)
    
    output_layer = tf.add(tf.matmul(hidden_layer, weights['w2']), biases['b2'])
    
    return output_layer


# Set learning rate, epochs and batch size
lr = 0.05
epochs = 100
bs = 24

# Load the data
x_train, x_test, y_train, y_test = load_data()

n_features = x_train.shape[1]
n_classes = y_train.shape[1]
n_hidden = 8

x = tf.placeholder(tf.float32, [None, n_features])
y = tf.placeholder(tf.float32, [None, n_classes])

# Initialize weights and biases
weights = {'w1':tf.Variable(tf.random_normal([n_features, n_hidden])),
    'w2': tf.Variable(tf.random_normal([n_hidden, n_classes]))
}

biases = {'b1':tf.Variable(tf.random_normal([n_hidden])),
    'b2':tf.Variable(tf.random_normal([n_classes]))
}

predictions = neural_net(weights, biases)

loss_fn = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=predictions, labels=y))
optimizer = tf.train.AdamOptimizer(learning_rate=lr).minimize(loss_fn)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for epoch in range(epochs):

        avg_loss = 0.0

        # Split the data into batches
        total_batch = int(len(x_train) / bs)

        xb = np.array_split(x_train, total_batch)
        yb = np.array_split(y_train, total_batch)

        for i in range(total_batch):

            _, loss = sess.run([optimizer, loss_fn], feed_dict={x:xb[i], y:yb[i]})
            avg_loss += loss / total_batch

        if (epoch + 1) % 10 == 0:
            print('Epoch:', epoch+1,'/',epochs,'\tLoss:',avg_loss)

        
    pred = tf.equal(tf.argmax(predictions, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(pred, 'float'))
    print('Train accuracy:', accuracy.eval({x:x_train, y:y_train}))
    print('Test accuracy:', accuracy.eval({x:x_test, y:y_test}))