import tensorflow as tf
import numpy as np
import pandas as pd
import random

LEARNING_RATE = 0.5
GRADIENT_STEPS = 1000

############ HELPER METHODS #######################
def split_data(train):
    num_samples = train.shape[0]
    train_size = (int) (num_samples * 0.9)
    train_samples = train[:train_size]
    validation_samples = train[train_size:]
    train_x = train_samples.drop("label", axis=1)
    train_y = train_samples["label"]
    validation_x = validation_samples.drop("label", axis=1)
    validation_y = validation_samples["label"]
    return train_x, train_y.values, validation_x, validation_y.values

def get_random_batch(train_x, train_y, size):
    start = random.randint(0, train_x.shape[0] - size - 1)
    batch_x = train_x[start:start+size]
    batch_y = train_y[start:start+size]
    #print("NUMBER OF SAMPLES:", train_x.shape[0])
    #print("TRAIN X TYPE:", type(train_x))
    #print("TRAIN X SHAPE:", train_x.shape)
    #print("TRAIN Y TYPE:", type(train_y))
    #print("TRAIN Y SIZE:", train_y.size)
    #print("BATCH X:", batch_x)
    #print("BATCH X TYPE:", type(batch_x))
    #print("BATCH X SHAPE:", batch_x.shape)
    #print("BATCH Y:", batch_y)
    #print("BATCH Y TYPE:", type(batch_y))
    #print("BATCH Y SIZE:", batch_y.size)
    return batch_x, batch_y

def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)

def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')

def main():
    ########### INPUT CODE##################
    train = pd.read_csv("../input/train.csv")
    test_x  = pd.read_csv("../input/test.csv")

    #print(train.head()) ###
    #print(train.head()["label"]) ###

    print("Training set has {0[0]} rows and {0[1]} columns".format(train.shape))
    print("Test set has {0[0]} rows and {0[1]} columns".format(test_x.shape))
    print(test_x.shape[0])

    train_x, train_labels, validation_x, validation_labels = split_data(train)
    
    # One hot encode labels
    train_y = np.zeros(shape=(train_labels.size, 10))
    for i in range(train_labels.size):
        train_y[i, train_labels[i]] = 1

    #print("Training features has {0[0]} rows and {0[1]} columns".format(train_x.shape))
    #print("Training labels has {0[0]} rows and {0[1]} columns".format(train_y.shape))

    validation_y = np.zeros(shape=(validation_labels.size, 10))
    for i in range(validation_labels.size):
        validation_y[i, validation_labels[i]] = 1
    #print("Validation features has {0[0]} rows and {0[1]} columns".format(validation_x.shape))
    #print("Validation labels has {0[0]} rows and {0[1]} columns".format(validation_y.shape))
    
    
    ########### TF CODE##################
    # Model code
    x = tf.placeholder(tf.float32, [None, 784])
    W = tf.Variable(tf.zeros([784, 10]))
    b = tf.Variable(tf.zeros([10]))
    y = tf.matmul(x, W) + b
        
    # Define loss and optimizer
    y_ = tf.placeholder(tf.float32, [None, 10])
    #cross_entropy = tf.reduce_mean(
    #tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))
    #train_step = tf.train.GradientDescentOptimizer(LEARNING_RATE).minimize(cross_entropy)
        
    sess = tf.InteractiveSession()
    #tf.global_variables_initializer().run()
    
    # First Convolutional Layer
    print("FIRST CONVOLUTION LAYER...")
    W_conv1 = weight_variable([5, 5, 1, 32])
    b_conv1 = bias_variable([32])
    x_image = tf.reshape(x, [-1,28,28,1])
    h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
    h_pool1 = max_pool_2x2(h_conv1)
    
    # Second Convolutional Layer
    print("SECOND CONVOLUTION LAYER...")
    W_conv2 = weight_variable([5, 5, 32, 64])
    b_conv2 = bias_variable([64])
    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
    h_pool2 = max_pool_2x2(h_conv2)

    # Densely Connected Layer
    print("DENSE CONVOLUTION LAYER...")
    W_fc1 = weight_variable([7 * 7 * 64, 1024])
    b_fc1 = bias_variable([1024])
    h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
    
    # Dropout
    print("DROPOUT...")
    keep_prob = tf.placeholder(tf.float32)
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)
    
    # Readout
    print("READOUT...")
    W_fc2 = weight_variable([1024, 10])
    b_fc2 = bias_variable([10])
    y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv))
    train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
    correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    sess.run(tf.global_variables_initializer())

    # Training
    print("STARTING TRAINING...")
    for i in range(3500):
        #sess.run(train_step, feed_dict={x: train_x, y_: train_y})
        batch_x, batch_y = get_random_batch(train_x, train_y, 50)
        if i % 100 == 0:
            print("STEP ACCURACY", i, ":", accuracy.eval(feed_dict={x:batch_x, y_:batch_y, keep_prob: 1.0}))
        train_step.run(feed_dict={x: batch_x, y_: batch_y, keep_prob: 0.5})
    print("TRAINING DONE!")
        
    # Evaluation
    #correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
    #accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    #print("TRAINING RESULTS!", sess.run(accuracy, feed_dict={x: train_x, y_: train_y}))
    #print("VALIDATION RESULTS!", sess.run(accuracy, feed_dict={x: validation_x, y_: validation_y}))
    #print("training test accuracy %g"%accuracy.eval(feed_dict={x: train_x, y_: train_y, keep_prob: 1.0}))
    #print("validation test accuracy %g"%accuracy.eval(feed_dict={x: validation_x, y_: validation_y, keep_prob: 1.0}))

    # Testing + output
    output_batch_size = 100
    output_index = 1
    num_batches = int(test_x.shape[0] / output_batch_size)
    prediction = tf.argmax(y_conv,1)
    f = open("output.csv", 'w')
    f.write("ImageId,Label\n")
    for batch_index in range(num_batches):
        batch_start = output_batch_size * batch_index
        #print("BATCH START:", batch_start)
        batch_test_x = test_x[batch_start:batch_start+output_batch_size]
        #print("BATCH SHAPE:", batch_test_x.shape)
        #print("BATCH DTYPE:", batch_test_x.dtypes)
        #print("BATCH:",batch_test_x)
        output = prediction.eval(feed_dict={x: batch_test_x, keep_prob: 1.0})
        for i in range(len(output)):
            #print("OUTPUT:", str(output_index), ",", str(output[i]))
            f.write(str(output_index) + "," + str(output[i]) + "\n")
            output_index += 1
    
    # Wrap up
    sess.close()
    print("NUMBER OF OUTPUT LINES:", output_index - 1)
    print("DONE!!!")

if __name__ == '__main__':
    main()