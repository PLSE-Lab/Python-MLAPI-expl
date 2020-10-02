""" This script is pretty much the same as the TensorFlow deepMNIST tutorial, albeit unlike 
    in the tutorial we get the datasets from csv files for which we use pandas and numpy to perform
    some pre-processing on the data. This is not as didactic as it could be so I plan on turning it 
    into a notebook any of this days"""
import tensorflow as tf
import numpy as np
import pandas as pd

# Some useful function definitions
def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


def conv2d(x, w):
    return tf.nn.conv2d(x, w, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

# Encodes a dense vector onto a one-hot matrix
def dense_to_one_hot(labels_dense, num_classes):
    num_labels = labels_dense.shape[0]
    index_offset = np.arange(num_labels) * num_classes
    labels_one_hot = np.zeros((num_labels, num_classes))
    labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1
    return labels_one_hot


# Helper Class to create the mini-batches for the training. You can definetely do this
# Without a class, instead declaring next_batch as a function and all the object atributes
# as global variables, I just find it to be more organized and elegant this way
class TrainBatcher(object):

    # Class constructor
    def __init__(self, examples, labels):
        self.labels = labels
        self.examples = examples
        self.index_in_epoch = 0
        self.num_examples = examples.shape[0]

    # mini-batching method
    def next_batch(self, batch_size):
        start = self.index_in_epoch
        self.index_in_epoch += batch_size

        # When all the training data is ran, shuffles it
        if self.index_in_epoch > self.num_examples:
            perm = np.arange(self.num_examples)
            np.random.shuffle(perm)
            self.examples = self.examples[perm]
            self.labels = self.labels[perm]
            # Start next epoch
            start = 0
            self.index_in_epoch = batch_size
            assert batch_size <= self.num_examples
        end = self.index_in_epoch
        
        return self.examples[start:end], self.labels[start:end]

# Declaring some constants
# Filepaths
# On your local environment set path to the folder containing the CSVs
PATH = '../input/'
TRAIN_SET = 'train.csv'
TEST_SET = 'test.csv'

# Some hyper-parameters
LEARNING_RATE = 2e-4
# On local environment set to 20000 to obtain ~99.2% accuracy
MAX_ITERATIONS = 1200       
BATCH_SIZE = 100
VALIDATION_SIZE = 2000

# Extract training data from CSV file
print('Reading data from CSV...')
data = pd.read_csv(PATH + TRAIN_SET)
print('Data read successfully')

# Pre-processing on the data
images = data.iloc[:, 1:].values
images = images.astype(np.float)

# Scaling to 0.0 - 1.0 values
images = np.multiply(images, 1.0 / 255.0)

# loading the labels and encoding them in a one-hot vector
labels_flat = data.iloc[:, 0].values.ravel()
labels = dense_to_one_hot(labels_flat, 10)

#Divides dataset into training and validation set
cv_images = images[:VALIDATION_SIZE]
cv_labels = labels[:VALIDATION_SIZE]

train_images = images[VALIDATION_SIZE:]
train_labels = labels[VALIDATION_SIZE:]

# Now we start to get TensorFlowy
x = tf.placeholder(tf.float32, shape=[None, 784])
y_ = tf.placeholder(tf.float32, shape=[None, 10])

# ===First convolutional layer===
# Applies convolution to 5x5 patches, with 1 input channel and 32 output channels
W_conv1 = weight_variable([5, 5, 1, 32])
b_conv1 = bias_variable([32])

# Each example is initially a 28*28 = 784 vector, here we reshape it into a 28x28 image
x_image = tf.reshape(x, [-1, 28, 28, 1])

# After applying the convolution we apply the ReLU activation to it
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
# simple 2x2 max_pooling, that is we take tha maximum of each 2x2 patch, this reduces
# the output of this layer to 14x14 shape 
h_pool1 = max_pool_2x2(h_conv1)


# ===Second convolutional layer===
# Analogous to first layer, but this time we got 32 input channels and 64 output
W_conv2 = weight_variable([5, 5, 32, 64])
b_conv2 = bias_variable([64])

# Same as previous Layer
h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

# ===Densely connected Layer===
W_fc1 = weight_variable([7 * 7 * 64, 1024])
b_fc1 = bias_variable([1024])

h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

# ===Dropout Layer===
# Dropout is a regularization method, we pretty much assign a random probability of discarding an example
# According to the TensorFlow tutorial on small networks like this dropout's influence is negligible so it is
# not really necessary
keep_prob = tf.placeholder('float')
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

# ===Readout Layer===
W_fc2 = weight_variable([1024, 10])
b_fc2 = bias_variable([10])

y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

# Cost Function - We use the softmax cross-entropy as our loss. This is the numerically stable implementation
cross_entropy = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv))

# Optimization Method
train_step = tf.train.AdamOptimizer(LEARNING_RATE).minimize(cross_entropy)

# Evaluation
correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
predict = tf.argmax(y_conv, 1)
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# Create a batcher object to provide the data mini-batches to our trainer
mnist = TrainBatcher(train_images, train_labels)

# Start TensorFlow session
sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())

print('Starting training...')
for i in range(MAX_ITERATIONS):
    batch_xs, batch_ys = mnist.next_batch(BATCH_SIZE)

    # Logs every 100 iterations
    if i % 100 == 0:
        train_accuracy = accuracy.eval(feed_dict={
            x: batch_xs, y_: batch_ys, keep_prob: 1.0})
        print("step %d, training accuracy %g" % (i, train_accuracy))

    train_step.run(feed_dict={x: batch_xs, y_: batch_ys, keep_prob: 0.5})

print('Training finished')

# Evaluates accuracy on CV set
cv_accuracy = accuracy.eval(feed_dict={
    x: cv_images, y_: cv_labels, keep_prob: 1.0})

print('validation_accuracy => %.4f'%cv_accuracy)
sess.close()
